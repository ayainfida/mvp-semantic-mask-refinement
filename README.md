# Semantic Mask Refinement with Diffusion Models

## Overview

This repository contains a novel approach to semantic segmentation refinement.

### Key Features

- **Residual Refiner**: A lightweight model that refines baseline U-Net predictions
- **Multi-condition Architecture**: Leverages baseline probabilities, logits, RGB images, and timestep embeddings
- **Boundary-aware Metrics**: Implements boundary F1 score for fine-grained evaluation
- **Multiple Datasets**: Tested on Oxford-IIIT Pets and COCO Humans datasets
- **Comprehensive Experiments**: Includes ablation studies, runtime analysis, and SAM comparisons

## Architecture

The refinement pipeline consists of:

1. **Baseline U-Net**: Standard semantic segmentation model
2. **Refiner**: Small U-Net head that:
   - Adds controlled noise to baseline predictions during training
   - Learns to refine masks at test time
   - Conditions on: baseline probabilities, logits, RGB image, and timestep
   - Uses cross-entropy loss against ground truth

## Repository Structure

```
mvp-semantic-mask-refinement/
├── models/
│   ├── unet.py                    # Baseline U-Net architecture
│   └── diffusion_refiner.py       # A refiner model
├── dataloaders/
│   ├── oxford_dataset.py          # Oxford-IIIT Pets dataset loader
│   └── coco_dataset.py            # COCO Humans dataset loader
├── data/                          # This will be automatically created by the notebook
│   ├── oxford-pets/               # Oxford-IIIT Pets dataset
        ├── train_images
        ├── test_images
        ├── train_masks
        ├── tests_masks
│   └── coco-dataset/              # COCO Humans dataset
        ├── annotations
        ├── images
├── experiments/
│   ├── diffusion_ablation.py      # Ablation studies (no_image, no_logits, etc.)
│   ├── diffusion_runtime.py       # Runtime analysis experiments
│   └── sam_compare.py             # Comparison with SAM baseline
├── utils/
│   └── metrics.py                 # Evaluation metrics (IoU, Boundary F1)
├── 01_oxford_pets_baseline_diffusion.ipynb
├── 02_coco_humans_baseline_diffusion.ipynb
└── 03_sota_sam_baseline_unet_comparison.ipynb
```

## Installation

```bash
# Clone the repository
git clone https://github.com/ayainfida/mvp-semantic-mask-refinement.git
cd mvp-semantic-mask-refinement

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy matplotlib pandas pillow tqdm
pip install segment-anything  # For SAM comparisons
```

## Datasets

### Oxford-IIIT Pets
- 3-class segmentation (background, pet, border)
- Train/test split provided
- Images normalized with dataset-specific mean/std

### COCO Humans
- 3-class segmentation (background, person, border)
- 85/15 train/validation split
- Custom dataset loader with automatic mask generation

## Usage

### Training

#### 1. Train Baseline U-Net

```python
from models.unet import UNet
from dataloaders.oxford_dataset import PetsDataset

# Initialize model
baseline_model = UNet(in_channels=3, num_classes=3)

# Train baseline model (see notebooks for full training loop)
# ...
```

#### 2. Train Diffusion Refiner

```python
from models.diffusion_refiner import train_diffusion_refiner

# Train refiner on baseline predictions
train_diffusion_refiner(
    baseline_model=baseline_model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    learning_rate=5e-4,
    T=300  # Number of diffusion timesteps
)
```

### Inference

```python
from models.diffusion_refiner import refine_mask

# Refine baseline predictions
refined_mask = refine_mask(
    diffusion_model=diffusion_model,
    baseline_model=baseline_model,
    image=image,
    num_steps=5,      # Number of refinement steps
    t_level=150       # Noise level (0=clean, T=max noise)
)
```

### Evaluation

```python
from utils.metrics import boundary_f1_batch

# Compute boundary F1 score
bf1 = boundary_f1_batch(
    gt_masks=ground_truth,
    pred_masks=predictions,
    num_classes=3,
    tolerance=2
)
```

## Experiments

### 1. Oxford Pets Baseline + Diffusion
**Notebook**: `01_oxford_pets_baseline_diffusion.ipynb`
- Train baseline U-Net on Oxford Pets
- Train diffusion refiner
- Evaluate refinement improvements
- Visualize results
- Rutime vs Quality analysis
- Ablation Studies

### 2. COCO Humans Baseline + Diffusion
**Notebook**: `02_coco_humans_baseline_diffusion.ipynb`
- Train on COCO-Human dataset
- Train diffusion refiner
- Evaluate refinement improvements
- Visualize results
- Rutime vs Quality analysis
- Ablation Studies

### 3. SAM Baseline Comparison
**Notebook**: `03_sota_sam_baseline_unet_comparison.ipynb`
- Compare baseline U-Net vs. Segment Anything Model (SAM)
- Evaluate on a small subset of COCO-Human dataset
- Visualize the segmentation

### Ablation Studies

Run ablation experiments to understand model components:

```python
from experiments.diffusion_ablation import run_ablation_study

results = run_ablation_study(
    modes=["full", "no_image", "no_logits", "no_probs"],
    diffusion_model=diffusion_model,
    baseline_model=baseline_model,
    test_loader=test_loader
)
```

**Ablation Modes**:
- `full`: All conditioning inputs
- `no_image`: Without RGB image conditioning
- `no_logits`: Without logit conditioning
- `no_probs`: Without probability conditioning

### Runtime Analysis

```python
from experiments.diffusion_runtime import measure_runtime

runtime_results = measure_runtime(
    diffusion_model=diffusion_model,
    num_steps=[1, 3, 5, 10, 20],
    device="cuda"
)
```

## Evaluation Metrics

### Mean Intersection over Union (mIoU)
Standard segmentation metric measuring overlap between prediction and ground truth.

### Boundary F1 Score
Specialized metric for evaluating boundary accuracy:
- Computes precision/recall on boundary pixels
- Configurable tolerance (default: 2 pixels)
- More sensitive to edge quality than IoU

### Accuracy
Per-class and overall accuracy measuring the percentage of correctly classified pixels.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Oxford-IIIT Pets dataset: [https://www.robots.ox.ac.uk/~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- COCO dataset: [https://cocodataset.org/](https://cocodataset.org/)
- Segment Anything Model (SAM): Meta AI Research
