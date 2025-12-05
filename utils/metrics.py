import numpy as np
import torch
import torch.nn.functional as F

def binary_erosion(mask, kernel_size=3):
    # min-pool via max-pool on inverted mask
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
        # thicken boundary by dilation
        pad = thickness
        boundary = F.max_pool2d(boundary, kernel_size=2*thickness+1,
                                stride=1, padding=pad)
        boundary = (boundary > 0.5).float()

    return boundary


def boundary_f1_single(gt, pred, num_classes, tolerance=2, ignore_background=True):
    gt = gt.to(torch.long)
    pred = pred.to(torch.long)

    class_range = range(1, num_classes) if ignore_background else range(num_classes)

    f1_list = []

    for c in class_range:
        gt_mask = (gt == c).float().unsqueeze(0).unsqueeze(0)    # (1,1,H,W)
        pred_mask = (pred == c).float().unsqueeze(0).unsqueeze(0)

        if gt_mask.sum() == 0 and pred_mask.sum() == 0:
            # no object of this class in GT or prediction -> skip
            continue

        gt_boundary = seg_to_boundary(gt_mask, thickness=1)
        pred_boundary = seg_to_boundary(pred_mask, thickness=1)

        # dilate boundaries for tolerance
        pad = tolerance
        gt_dil = F.max_pool2d(gt_boundary, kernel_size=2*tolerance+1,
                              stride=1, padding=pad)
        pred_dil = F.max_pool2d(pred_boundary, kernel_size=2*tolerance+1,
                                stride=1, padding=pad)

        # Precision: predicted boundary vs dilated GT boundary
        tp = (pred_boundary * gt_dil).sum()
        pred_pixels = pred_boundary.sum()

        # Recall: GT boundary vs dilated predicted boundary
        tp_recall = (gt_boundary * pred_dil).sum()
        gt_pixels = gt_boundary.sum()

        eps = 1e-6
        precision = tp / (pred_pixels + eps)
        recall = tp_recall / (gt_pixels + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1_list.append(f1.item())

    if len(f1_list) == 0:
        return 0.0

    return float(np.mean(f1_list))


@torch.no_grad()
def boundary_f1_score(model, loader, num_classes, device, tolerance=2, ignore_background=True):
    model.eval()
    total_f1 = 0.0
    count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).squeeze(1)  # (B,H,W)
        logits = model(images)
        preds = logits.argmax(1)               # (B,H,W)

        for b in range(images.size(0)):
            f1 = boundary_f1_single(labels[b], preds[b],
                                    num_classes=num_classes,
                                    tolerance=tolerance,
                                    ignore_background=ignore_background)
            total_f1 += f1
            count += 1

    return total_f1 / max(count, 1)
