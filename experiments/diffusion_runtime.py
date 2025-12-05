# diffusion_runtime.py
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.metrics import boundary_f1_single

# ------------------------------------------------------------
#  Multi-step refinement wrapper
# ------------------------------------------------------------
def refine_with_multiple_steps(
    diffusion_model,
    baseline_model,
    images,
    q_sample,
    T,
    num_steps=1,
    t_level=0,
    device="cuda",
):
    diffusion_model.eval()
    baseline_model.eval()

    images = images.to(device)

    logits = baseline_model(images)   # [B,C,H,W]
    logits_base = logits.clone()

    t_level = int(max(0, min(T, t_level)))
    B = images.size(0)
    t = torch.full((B,), t_level, device=device, dtype=torch.long)

    for _ in range(num_steps):
        probs = F.softmax(logits, dim=1)
        x_t = q_sample(probs, t)

        B_, C, H, W = x_t.shape
        t_norm = t.float() / max(T, 1)
        t_img = t_norm.view(B_, 1, 1, 1).expand(-1, 1, H, W)

        cond_probs  = probs
        cond_logits = logits
        img_input   = images

        x_in = torch.cat([x_t, t_img, cond_probs, cond_logits, img_input], dim=1)
        delta_logits = diffusion_model(x_in)
        logits = logits + delta_logits

    return logits_base, logits


@torch.no_grad()
def measure_baseline_runtime_and_quality(model, loader, num_classes, device):
    model.eval()
    confusion = torch.zeros(num_classes ** 2, dtype=torch.int64)
    total_time = 0.0
    total_images = 0

    total_bf1 = 0.0
    count_bf1 = 0

    for images, labels in tqdm(loader, desc="Baseline runtime", leave=False):
        images = images.to(device)
        labels = labels.to(device).squeeze(1)

        start = time.perf_counter()
        logits = model(images)
        end = time.perf_counter()

        total_time += (end - start)
        total_images += labels.size(0)

        preds = logits.argmax(1)
        entries = preds * num_classes + labels
        counts = torch.bincount(entries.view(-1), minlength=num_classes ** 2)
        confusion += counts.cpu()

        for b in range(labels.size(0)):
            f1 = boundary_f1_single(
                labels[b],
                preds[b],
                num_classes=num_classes,
                tolerance=2,
                ignore_background=True,
            )
            total_bf1 += f1
            count_bf1 += 1

    confusion = confusion.view(num_classes, num_classes)
    acc = 100.0 * confusion.diag().sum().float() / confusion.sum().float()

    eps = 1e-6
    intersection = confusion.diag().float()
    union = confusion.sum(0).float() + confusion.sum(1).float() - intersection
    iou = (intersection / (union + eps)).tolist()
    miou = sum(iou) / len(iou)

    runtime_per_image = total_time / max(total_images, 1)
    bf1 = total_bf1 / max(count_bf1, 1)

    return runtime_per_image, acc.item(), iou, miou, bf1


@torch.no_grad()
def evaluate_refiner_runtime_quality(
    diffusion_model,
    baseline_model,
    loader,
    q_sample,
    T,
    num_steps_list,
    t_level,
    num_classes,
    device,
):
    results = {}
    diffusion_model.eval()
    baseline_model.eval()

    for num_steps in num_steps_list:
        print(f"\n=== Evaluating diffusion refiner: num_steps={num_steps}, t_level={t_level} ===")

        confusion = torch.zeros(num_classes ** 2, dtype=torch.int64)
        total_time = 0.0
        total_images = 0

        total_bf1 = 0.0
        count_bf1 = 0

        for images, labels in tqdm(loader, desc=f"Refiner steps={num_steps}", leave=False):
            images = images.to(device)
            labels = labels.to(device).squeeze(1)
            B = labels.size(0)

            start = time.perf_counter()
            _, refined_logits = refine_with_multiple_steps(
                diffusion_model, baseline_model, images,
                q_sample=q_sample,
                T=T,
                num_steps=num_steps,
                t_level=t_level,
                device=device
            )
            end = time.perf_counter()

            total_time += (end - start)
            total_images += B

            preds = refined_logits.argmax(1)

            entries = preds * num_classes + labels
            counts = torch.bincount(entries.view(-1), minlength=num_classes ** 2)
            confusion += counts.cpu()

            for b in range(B):
                f1 = boundary_f1_single(
                    labels[b],
                    preds[b],
                    num_classes=num_classes,
                    tolerance=2,
                    ignore_background=True,
                )
                total_bf1 += f1
                count_bf1 += 1

        confusion = confusion.view(num_classes, num_classes)
        acc = 100.0 * confusion.diag().sum().float() / confusion.sum().float()

        eps = 1e-6
        intersection = confusion.diag().float()
        union = confusion.sum(0).float() + confusion.sum(1).float() - intersection
        iou = (intersection / (union + eps)).tolist()
        miou = sum(iou) / len(iou)

        runtime_per_image = total_time / max(total_images, 1)
        bf1 = total_bf1 / max(count_bf1, 1)

        print(f"  Runtime per image: {runtime_per_image:.6f} s")
        print(f"  Accuracy: {acc:.2f}%")
        print(f"  IoU per class: {iou}")
        print(f"  mIoU: {miou:.4f}")
        print(f"  Boundary F1 (non-bg, tol=2 px): {bf1:.4f}")

        results[num_steps] = (runtime_per_image, acc.item(), iou, miou, bf1)

    return results