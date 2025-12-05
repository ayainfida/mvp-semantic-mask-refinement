# experiments/sam_compare.py

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator


# -----------------------------
# SAM helpers
# -----------------------------
def build_mask_generator(sam_model):
    return SamAutomaticMaskGenerator(
        sam_model,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
    )


def tensor_to_rgb_uint8(img_tensor, mean_rgb, std_rgb):
    img = img_tensor.cpu().clone()
    for c in range(3):
        img[c] = img[c] * std_rgb[c] + mean_rgb[c]
    img = img.clamp(0.0, 1.0)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


def run_sam_auto(image_uint8, mask_generator):
    masks = mask_generator.generate(image_uint8)
    masks = sorted(masks, key=lambda x: x["area"], reverse=True)
    return [m["segmentation"] for m in masks]


def binary_iou(pred_bool, gt_bool):
    inter = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum() + 1e-6
    return inter / union


# -----------------------------
# Boundary F1 helpers
# -----------------------------
def binary_erosion(mask, kernel_size=3):
    pad = kernel_size // 2
    inv = 1.0 - mask
    eroded_inv = F.max_pool2d(inv, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = 1.0 - eroded_inv
    return eroded


def seg_to_boundary(mask, thickness=1):
    eroded = binary_erosion(mask, kernel_size=3)
    boundary = mask - eroded
    boundary = (boundary > 0.5).float()

    if thickness > 1:
        pad = thickness
        boundary = F.max_pool2d(
            boundary,
            kernel_size=2 * thickness + 1,
            stride=1,
            padding=pad,
        )
        boundary = (boundary > 0.5).float()

    return boundary


def boundary_f1_single(gt, pred, num_classes, tolerance=2, ignore_background=True):
    gt = gt.to(torch.long)
    pred = pred.to(torch.long)
    class_range = range(1, num_classes) if ignore_background else range(num_classes)

    f1_list = []
    for c in class_range:
        gt_mask   = (gt == c).float().unsqueeze(0).unsqueeze(0)    # (1,1,H,W)
        pred_mask = (pred == c).float().unsqueeze(0).unsqueeze(0)

        if gt_mask.sum() == 0 and pred_mask.sum() == 0:
            continue

        gt_boundary   = seg_to_boundary(gt_mask,   thickness=1)
        pred_boundary = seg_to_boundary(pred_mask, thickness=1)

        pad    = tolerance
        gt_dil = F.max_pool2d(gt_boundary,   kernel_size=2*tolerance+1,
                              stride=1, padding=pad)
        pr_dil = F.max_pool2d(pred_boundary, kernel_size=2*tolerance+1,
                              stride=1, padding=pad)

        tp_prec = (pred_boundary * gt_dil).sum()
        pred_pix = pred_boundary.sum()

        tp_rec = (gt_boundary * pr_dil).sum()
        gt_pix = gt_boundary.sum()

        eps = 1e-6
        precision = tp_prec / (pred_pix + eps)
        recall    = tp_rec  / (gt_pix   + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        f1_list.append(f1.item())

    if not f1_list:
        return 0.0
    return float(np.mean(f1_list))


# -----------------------------
# Main visualisation API
# -----------------------------
@torch.no_grad()
def visualize_unet_vs_sam_subset(
    unet_model,
    mask_generator,
    loader,
    device,
    mean_rgb,
    std_rgb,
    num_classes,
    person_label=1,
    max_examples=8,
):
    unet_model.eval()

    examples = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.squeeze(1)  # [B,H,W]
        logits = unet_model(images)
        preds_unet = logits.argmax(1).cpu()
        labels_cpu = labels.cpu()

        B = images.size(0)
        for i in range(B):
            if len(examples) >= max_examples:
                break

            gt = labels_cpu[i]
            if (gt == person_label).sum() == 0:
                continue

            unet_pred = preds_unet[i]

            gt_np   = gt.numpy()
            unet_np = unet_pred.numpy()
            gt_bool   = (gt_np == person_label)
            unet_bool = (unet_np == person_label)

            iou_unet = binary_iou(unet_bool, gt_bool)
            bf1_unet = boundary_f1_single(gt, unet_pred,
                                          num_classes=num_classes,
                                          tolerance=2,
                                          ignore_background=True)

            rgb = tensor_to_rgb_uint8(images[i], mean_rgb, std_rgb)
            sam_masks = run_sam_auto(rgb, mask_generator)
            best_iou  = 0.0
            best_mask = np.zeros_like(gt_bool, dtype=bool)
            if sam_masks:
                for m in sam_masks:
                    iou = binary_iou(m, gt_bool)
                    if iou > best_iou:
                        best_iou = iou
                        best_mask = m

            sam_pred = torch.from_numpy(best_mask.astype(np.int64))
            bf1_sam  = boundary_f1_single(gt, sam_pred,
                                          num_classes=num_classes,
                                          tolerance=2,
                                          ignore_background=True)

            examples.append({
                "rgb": rgb,
                "gt": gt_np,
                "unet_pred": unet_np,
                "sam_pred": best_mask.astype(np.int64),
                "iou_unet": iou_unet,
                "iou_sam": best_iou,
                "bf1_unet": bf1_unet,
                "bf1_sam": bf1_sam,
            })

        if len(examples) >= max_examples:
            break

    if not examples:
        print("No examples with person_label found.")
        return

    print("\nPer-example IoU / Boundary-F1 (person class):")
    for idx, ex in enumerate(examples):
        print(
            f"Example {idx}: "
            f"IoU U-Net={ex['iou_unet']:.3f}, SAM={ex['iou_sam']:.3f} | "
            f"BF1 U-Net={ex['bf1_unet']:.3f}, SAM={ex['bf1_sam']:.3f}"
        )

    N = len(examples)
    plt.figure(figsize=(4*4, 4*N))

    for i, ex in enumerate(examples):
        row = i * 4

        plt.subplot(N, 4, row+1)
        plt.imshow(ex["rgb"])
        plt.axis("off")
        plt.title(f"Input #{i}")

        plt.subplot(N, 4, row+2)
        plt.imshow(ex["gt"], cmap="viridis")
        plt.axis("off")
        plt.title("GT mask")

        plt.subplot(N, 4, row+3)
        plt.imshow(ex["unet_pred"], cmap="viridis")
        plt.axis("off")
        plt.title(f"U-Net\nIoU={ex['iou_unet']:.3f}\nBF1={ex['bf1_unet']:.3f}")

        plt.subplot(N, 4, row+4)
        plt.imshow(ex["sam_pred"], cmap="viridis")
        plt.axis("off")
        plt.title(f"SAM\nIoU={ex['iou_sam']:.3f}\nBF1={ex['bf1_sam']:.3f}")

    plt.tight_layout()
    plt.show()