# experiments/diffusion_ablation.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------------------------
# 1) Core refiner: multiple diffusion steps + ablations
# ------------------------------------------------
@torch.no_grad()
def refine_with_multiple_steps(
    diffusion_model,
    baseline_model,
    images,
    q_sample,
    T,
    num_steps=1,
    t_level=0,
    mode="full",   # "full", "no_image", "no_logits", "no_probs"
    device="cuda",
):
    diffusion_model.eval()
    baseline_model.eval()

    images = images.to(device)

    # 1) baseline prediction
    logits = baseline_model(images)          # [B,C,H,W]
    logits_base = logits.clone()             # keep a copy for comparison

    # clamp t_level to [0, T]
    t_level = int(max(0, min(T, t_level)))
    B = images.size(0)
    t = torch.full((B,), t_level, device=device, dtype=torch.long)

    for _ in range(num_steps):
        probs = F.softmax(logits, dim=1)     # [B,C,H,W]
        x_t = q_sample(probs, t)             # [B,C,H,W]

        B_, C, H, W = x_t.shape
        t_norm = t.float() / max(T, 1)
        t_img = t_norm.view(B_, 1, 1, 1).expand(-1, 1, H, W)

        cond_probs  = probs
        cond_logits = logits
        img_input   = images

        # Ablation: zero out some conditioning signals
        if mode == "no_image":
            img_input = torch.zeros_like(images)
        elif mode == "no_logits":
            cond_logits = torch.zeros_like(logits)
        elif mode == "no_probs":
            cond_probs = torch.zeros_like(probs)
        # "full" => nothing zeroed

        x_in = torch.cat([x_t, t_img, cond_probs, cond_logits, img_input], dim=1)

        delta_logits = diffusion_model(x_in)
        logits = logits + delta_logits       # refined logits

    return logits_base, logits


# ------------------------------------------------
# 2) Metrics helper (Accuracy, IoU, mIoU, optional BF1)
# ------------------------------------------------
@torch.no_grad()
def evaluate_model_from_logits_fn(
    logits_fn,
    loader,
    num_classes,
    device,
    description="Model",
    boundary_f1_single=None,
    bf1_tolerance=2,
    bf1_ignore_background=True,
):
    confusion = torch.zeros(num_classes ** 2, dtype=torch.int64)
    total_bf1 = 0.0
    count_bf1 = 0

    for images, labels in tqdm(loader, desc=description, leave=False):
        labels = labels.to(device).squeeze(1)   # [B,H,W]
        preds = logits_fn(images, labels)       # [B,H,W]

        entries = preds * num_classes + labels
        counts = torch.bincount(entries.view(-1), minlength=num_classes ** 2)
        confusion += counts.cpu()

        # Boundary F1 if available
        if boundary_f1_single is not None:
            for b in range(labels.size(0)):
                f1 = boundary_f1_single(
                    labels[b],
                    preds[b],
                    num_classes=num_classes,
                    tolerance=bf1_tolerance,
                    ignore_background=bf1_ignore_background,
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

    if boundary_f1_single is not None and count_bf1 > 0:
        bf1 = total_bf1 / count_bf1
    else:
        bf1 = None

    return confusion, float(acc.item()), iou, float(miou), bf1


# ------------------------------------------------
# 3) Top-level Task 4 wrapper: quantitative + ablation
# ------------------------------------------------
def run_task4_quant_ablation(
    baseline_model,
    diffusion_model,
    test_loader,
    num_classes,
    device,
    q_sample,
    T,
    boundary_f1_single=None,
    steps=(1, 2, 3),
    modes=("full", "no_image", "no_logits", "no_probs"),
):
    # ---- baseline ----
    def baseline_logits_fn(images, labels=None):
        images = images.to(device)
        logits = baseline_model(images)
        return logits.argmax(1)

    conf_base, acc_base, iou_base, miou_base, bf1_base = evaluate_model_from_logits_fn(
            logits_fn=baseline_logits_fn,
            loader=test_loader,
            num_classes=num_classes,
            device=device,
            description="Baseline (U-Net)",
            boundary_f1_single=boundary_f1_single,
        )

    baseline_metrics = {
        "confusion": conf_base,
        "accuracy": acc_base,
        "iou": iou_base,
        "miou": miou_base,
        "bf1": bf1_base,
    }

    print("\nBaseline (U-Net)")
    print(f"  Accuracy:   {acc_base:.2f}%")
    print(f"  IoU / class:{iou_base}")
    print(f"  mIoU:       {miou_base:.4f}")
    if bf1_base is not None:
        print(f"  Boundary F1:{bf1_base:.4f}")

    # ---- refined (full conditioning) ----
    refined_results_full = {}
    rows_full = []

    for k in steps:
        def refiner_logits_fn(images, labels=None, steps_=k):
            _, refined_logits = refine_with_multiple_steps(
                diffusion_model=diffusion_model,
                baseline_model=baseline_model,
                images=images,
                q_sample=q_sample,
                T=T,
                num_steps=steps_,
                t_level=0,
                mode="full",
                device=device,
            )
            return refined_logits.argmax(1)

        conf_ref, acc_ref, iou_ref, miou_ref, bf1_ref = evaluate_model_from_logits_fn(
            logits_fn=lambda imgs, lbls, steps_=k: refiner_logits_fn(imgs, lbls, steps_),
            loader=test_loader,
            num_classes=num_classes,
            device=device,
            description=f"Refiner (full, steps={k})",
            boundary_f1_single=boundary_f1_single,
        )

        refined_results_full[k] = {
            "confusion": conf_ref,
            "accuracy": acc_ref,
            "iou": iou_ref,
            "miou": miou_ref,
            "bf1": bf1_ref,
        }

        print(f"\nRefined (full conditioning, steps={k})")
        print(f"  Accuracy:   {acc_ref:.2f}%")
        print(f"  IoU / class:{iou_ref}")
        print(f"  mIoU:       {miou_ref:.4f}")
        if bf1_ref is not None:
            print(f"  Boundary F1:{bf1_ref:.4f}")

        # For the DataFrame
        row = {
            "Method": "Refined_full",
            "Steps": k,
            "Accuracy (%)": acc_ref,
            "mIoU": miou_ref,
        }
        for i, val in enumerate(iou_ref):
            row[f"IoU_class_{i}"] = val
        if bf1_ref is not None:
            row["Boundary_F1"] = bf1_ref
        rows_full.append(row)

    # add baseline row
    base_row = {
        "Method": "Baseline",
        "Steps": 0,
        "Accuracy (%)": acc_base,
        "mIoU": miou_base,
    }
    for i, val in enumerate(iou_base):
        base_row[f"IoU_class_{i}"] = val
    if bf1_base is not None:
        base_row["Boundary_F1"] = bf1_base
    rows_full.append(base_row)

    df_full_refinement = pd.DataFrame(rows_full)
    df_full_refinement = df_full_refinement.sort_values(by=["Method", "Steps"])

    # ---- ablation over modes x steps ----
    print("\n=== Ablation on conditioning + steps (t_level=0) ===")
    ablation_results = {}
    rows_ablation = []

    for mode in modes:
        for k in steps:
            def refiner_logits_fn_mode(images, labels=None, steps_=k, mode_=mode):
                _, refined_logits = refine_with_multiple_steps(
                    diffusion_model=diffusion_model,
                    baseline_model=baseline_model,
                    images=images,
                    q_sample=q_sample,
                    T=T,
                    num_steps=steps_,
                    t_level=0,
                    mode=mode_,
                    device=device,
                )
                return refined_logits.argmax(1)

            desc = f"Refiner (mode={mode}, steps={k})"
            _, acc_m, iou_m, miou_m, bf1_m = evaluate_model_from_logits_fn(
                logits_fn=lambda imgs, lbls, steps_=k, mode_=mode:
                    refiner_logits_fn_mode(imgs, lbls, steps_, mode_),
                loader=test_loader,
                num_classes=num_classes,
                device=device,
                description=desc,
                boundary_f1_single=boundary_f1_single,
            )

            ablation_results[(mode, k)] = {
                "accuracy": acc_m,
                "iou": iou_m,
                "miou": miou_m,
                "bf1": bf1_m,
            }

            print(f"{desc}: Accuracy={acc_m:.2f}%, mIoU={miou_m:.4f}"
                  + (f", BF1={bf1_m:.4f}" if bf1_m is not None else ""))

            row = {
                "Mode": mode,
                "Steps": k,
                "Accuracy (%)": acc_m,
                "mIoU": miou_m,
            }
            for i, val in enumerate(iou_m):
                row[f"IoU_class_{i}"] = val
            if bf1_m is not None:
                row["Boundary_F1"] = bf1_m
            rows_ablation.append(row)

    df_ablation = pd.DataFrame(rows_ablation).sort_values(by=["Mode", "Steps"])

    print("\n=== Ablation summary (t_level=0) ===")
    print(f"Baseline mIoU: {miou_base:.4f}")
    for mode in modes:
        for k in steps:
            r = ablation_results[(mode, k)]
            print(
                f"Mode={mode:8s}, steps={k}: "
                f"mIoU={r['miou']:.4f}"
                + (f", BF1={r['bf1']:.4f}" if r['bf1'] is not None else "")
            )

    return baseline_metrics, df_full_refinement, df_ablation, refined_results_full, ablation_results


# ------------------------------------------------
# 4) Qualitative visualisation
# ------------------------------------------------
def denormalize_image(tensor, mean_rgb, std_rgb):
    mean = torch.tensor(mean_rgb).view(3, 1, 1)
    std = torch.tensor(std_rgb).view(3, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)


@torch.no_grad()
def visualize_baseline_vs_diffusion_steps(
    baseline_model,
    diffusion_model,
    loader,
    q_sample,
    T,
    mean_rgb,
    std_rgb,
    steps_to_show=(1, 2, 3),
    t_level=0,
    mode="full",
    num_samples=3,
    device="cuda",
):

    baseline_model.eval()
    diffusion_model.eval()

    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device).squeeze(1)

    logits_base = baseline_model(images)
    base_probs = F.softmax(logits_base, dim=1)
    base_preds = base_probs.argmax(1)

    num_samples = min(num_samples, images.size(0))
    n_cols = 3 + len(steps_to_show)     # Input, GT, Baseline, then each step
    fig, axes = plt.subplots(num_samples, n_cols, figsize=(4 * n_cols, 3 * num_samples))

    if num_samples == 1:
        axes = axes[None, :]

    for i in range(num_samples):
        img = denormalize_image(images[i].cpu(), mean_rgb, std_rgb).permute(1, 2, 0).numpy()
        gt = labels[i].cpu().numpy()

        # 1) Input
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")

        # 2) GT
        axes[i, 1].imshow(gt, interpolation="nearest")
        axes[i, 1].set_title("GT")
        axes[i, 1].axis("off")

        # 3) Baseline
        axes[i, 2].imshow(base_preds[i].cpu(), interpolation="nearest")
        axes[i, 2].set_title("Baseline")
        axes[i, 2].axis("off")

        # 4..N) refined
        col = 3
        for k in steps_to_show:
            _, refined_logits = refine_with_multiple_steps(
                diffusion_model=diffusion_model,
                baseline_model=baseline_model,
                images=images[i:i+1],
                q_sample=q_sample,
                T=T,
                num_steps=k,
                t_level=t_level,
                mode=mode,
                device=device,
            )
            ref_pred = refined_logits.argmax(1)[0].cpu().numpy()
            axes[i, col].imshow(ref_pred, interpolation="nearest")
            axes[i, col].set_title(f"Refined (steps={k})")
            axes[i, col].axis("off")
            col += 1

    plt.tight_layout()
    plt.show()