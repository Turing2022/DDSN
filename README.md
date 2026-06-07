# DDSN: Dual-Domain Spiking Network for Low-Light Image Enhancement

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](#environment)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-ee4c2c.svg)](#environment)
[![Status](https://img.shields.io/badge/Status-Research%20Code-brightgreen.svg)](#overview)

> Official PyTorch implementation of **DDSN**, a dual-domain low-light image enhancement framework that combines frequency-guided priors, spiking-inspired adaptive thresholding, and segmentation-aware supervision.

## Overview

Low-light image enhancement requires both faithful illumination restoration and reliable structure preservation. DDSN addresses this challenge with a dual-domain design that couples:

- **frequency-domain priors** for illumination and structure disentanglement,
- **spiking-inspired adaptive threshold modulation** for dynamic feature filtering,
- **deformable spatial modeling** for context-aware enhancement, and
- **segmentation-aware auxiliary supervision** for preserving task-relevant structures.

In the current implementation, amplitude-related priors are used to guide adaptive threshold modulation, while phase-related priors are used to preserve geometric and structural information.

## Highlights

- Dual-domain enhancement architecture with spatial-frequency interaction
- Frequency prior modeling through amplitude and phase decomposition in deep features
- Spiking-inspired attention with illumination-adaptive ternary thresholding
- Optional segmentation-aware training for structure preservation
- Energy analysis utility for sparsity-aware efficiency estimation

## Method

The main model is implemented in [DDSN.py](/D:/BaiduNetdiskDownload/DDSN/DDSN.py).

Key components include:

- `FreMLP`: extracts amplitude-enhanced illumination priors and phase-based structure priors
- `SpikingMSDeformableAttention`: injects frequency guidance into deformable attention
- `BipolarNeuron`: performs adaptive threshold modulation based on amplitude priors
- `SpikingEnhancementBlock`: fuses frequency and spatial enhancement streams
- `VMUNet` / `net`: reconstructs the final enhanced image with residual learning

The final enhanced result is produced by residual reconstruction:

```python
final = out_unet + inputs
```

## Visualization

You can use this section to place an architecture teaser or qualitative comparisons.

```markdown
![DDSN teaser](./assets/teaser.png)
```

Suggested visual materials for this repository:
- overall architecture figure
- qualitative enhancement comparisons
- amplitude/phase prior visualization
- threshold modulation heatmaps
- downstream segmentation preservation examples

If you do not yet have figures ready, keeping this section in place still helps structure the project page for later updates.

## Results

A suggested quantitative table format is shown below. Replace the placeholder values with your final paper or benchmark numbers.

| Dataset | PSNR | SSIM | LPIPS | Notes |
| --- | ---: | ---: | ---: | --- |
| LOL-v1 | TBA | TBA | TBA | Main benchmark |
| LOL-v2 | TBA | TBA | TBA | Optional |
| VE-LOL / custom | TBA | TBA | TBA | Optional |

You may also add a short qualitative summary here, for example:

> DDSN improves illumination restoration while preserving fine structures under challenging low-light conditions.

## Repository Structure

```text
DDSN/
├─ DDSN.py                 # main network definition and energy analysis
├─ train.py                # training script
├─ test.py                 # folder-based inference script
├─ requirements.txt        # Python dependencies
├─ datasets/               # optional local dataset folder
├─ lib/                    # dataloaders, transforms, utilities, SSIM
├─ utils/                  # logging, losses, metrics, optimization helpers
├─ segmodel/               # segmentation model package
├─ FR/                     # segmentation-related subproject and configs
├─ weight/                 # optional pretrained weights
└─ tensorboard/            # training logs
```

## Environment

Recommended setup:
- Python 3.8+
- PyTorch 2.0.1
- CUDA-enabled GPU for training and fast inference

Install dependencies with:

```bash
pip install -r requirements.txt
```

Since `requirements.txt` contains many packages from the original research environment, a lighter installation may also be sufficient for most users:

```bash
pip install torch torchvision torchaudio timm einops pillow opencv-python scikit-image tqdm tensorboard thop ruamel.yaml bunch lpips
```

## Dataset Preparation

The training pipeline expects paired low-light images, normal-light references, and optional segmentation masks.

Expected directory structure:

```text
datasets/
├─ train/
│  ├─ low/
│  ├─ high/
│  └─ seg/
└─ test/
   ├─ low/
   └─ high/
```

Folder description:
- `low/`: low-light input images
- `high/`: corresponding normal-light ground truth images
- `seg/`: segmentation masks used during training

Important notes:
- file names should align across `low`, `high`, and `seg`
- the custom dataset loader in `train.py` assumes equal numbers of files in these folders

## Training

Run training with:

```bash
python train.py \
  --trainset /path/to/datasets/train \
  --testset /path/to/datasets/test \
  --modelname DDSN_experiment \
  --deviceid 0
```

Useful arguments from the current training script:
- `--trainset`: training set path
- `--testset`: validation/test set path used during evaluation
- `--output`: output directory for generated images
- `--modelname`: experiment name
- `--deviceid`: GPU id
- `--lr`: initial learning rate
- `--lr_min`: minimum learning rate for cosine decay
- `--batchSize`: batch size
- `--nEpochs`: number of training epochs
- `--patch_size`: crop size
- `--seg_weight_path`: pretrained segmentation weight path
- `--seg_config_path`: segmentation config path
- `--use_gt_seg`: whether to use ground-truth segmentation masks

Training artifacts are saved to:
- `models/<modelname>/best.pth`
- `models/<modelname>/last.pth`
- `output/<modelname>/...`
- `tensorboard/<modelname>/...`

## Inference

The provided [test.py](/D:/BaiduNetdiskDownload/DDSN/test.py) performs folder-based inference and saves enhanced images.

Before running inference, update the hard-coded paths in `test.py`:
- `self.weights_path`
- `self.input_folder`
- `self.output_folder`
- `self.device_id`

Then run:

```bash
python test.py
```

### Important Note

The current inference script imports the model as:

```python
from Best_module.BEE1 import net
```

If the public release of this repository is centered on [DDSN.py](/D:/BaiduNetdiskDownload/DDSN/DDSN.py), it is recommended to change that line to:

```python
from DDSN import net
```

This will make the repository self-contained and easier for others to reproduce.

## Segmentation-Aware Supervision

This codebase optionally uses an auxiliary segmentation model during training:

- the segmentation network is loaded from `segmodel/`
- configuration is read from `FR/config.yaml`
- the enhanced output is passed through the segmentation branch
- IoU and Dice based losses are used as structural supervision

This auxiliary objective is intended to encourage the enhanced images to remain semantically and structurally faithful.

## Energy Analysis

[DDSN.py](/D:/BaiduNetdiskDownload/DDSN/DDSN.py) includes an `EnergyMeter` utility for approximate sparsity and energy analysis.

Run:

```bash
python DDSN.py
```

This routine will:
- construct the model
- run a dummy forward pass
- estimate spiking sparsity
- report approximate FLOPs and energy consumption

## Todo

- Release cleaned training and inference scripts without hard-coded paths
- Add architecture and qualitative result figures
- Provide pretrained checkpoints and download links
- Add benchmark tables for LOL-v1 / LOL-v2 or other target datasets
- Refactor inference into a command-line interface
- Add threshold-visualization and amplitude-phase exchange analysis scripts

## Citation

If you find this repository useful, please cite your paper when available.

```bibtex
@article{ddsn2026,
  title   = {DDSN: Dual-Domain Spiking Network for Low-Light Image Enhancement},
  author  = {Author, First and Author, Second and Author, Third},
  journal = {arXiv / Conference / Journal},
  year    = {2026}
}
```

## Acknowledgements

This repository uses or builds upon ideas and tooling related to:
- [PyTorch](https://pytorch.org/)
- [timm](https://github.com/huggingface/pytorch-image-models)
- [einops](https://github.com/arogozhnikov/einops)
- FR-UNet style segmentation utilities under `FR/`

## Notes for Open-Sourcing

Before publishing this repository on GitHub, it is recommended to:
- replace all local absolute paths in `train.py` and `test.py`
- make the inference import path consistent with the released code layout
- simplify `requirements.txt` to a minimal reproducible environment
- remove IDE metadata such as `.idea/` if not needed
- add a root-level `LICENSE` file

## License

No root-level license file is currently included in this repository.
If you plan to release the project publicly, adding a license such as MIT, Apache-2.0, or GPL is strongly recommended.
