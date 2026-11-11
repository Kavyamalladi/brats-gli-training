# BraTS 2023 GLI segmentation training

This repository contains a Swin UNETR training script for the BraTS 2023 Adult Glioma (GLI) dataset. It uses four MRI modalities (T1, contrast-enhanced T1, T2, and FLAIR) and predicts exclusive voxel classes: background, NCR/NET, edema, and enhancing tumor. ET, TC, and WT Dice scores are derived from those predictions.

## Data

Obtain the BraTS 2023 GLI training data through the official Synapse project and follow its access terms: https://www.synapse.org/Synapse:syn51156910

Extract the archive so this path contains the individual `BraTS-GLI-*` subject folders:

```text
data/BraTS-GLI/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
```

Each subject folder should contain `-t1n.nii.gz`, `-t1c.nii.gz`, `-t2w.nii.gz`, `-t2f.nii.gz`, and `-seg.nii.gz` files named with the subject ID prefix. The dataset itself is not included in this repository.

## Setup (Windows PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For NVIDIA GPU training, install the CUDA-enabled PyTorch build that matches your driver using the official selector at https://pytorch.org/get-started/locally/.

## Check data and train

```powershell
.\.venv\Scripts\python.exe -m datasets.test_loader --data-dir "data\BraTS-GLI\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
.\.venv\Scripts\python.exe train.py --smoke-test
.\.venv\Scripts\python.exe train.py --epochs 5 --batch-size 1 --max-subjects 100
```

The trainer currently uses 64 x 64 x 64 patches, feature size 24, and mixed precision on CUDA to fit a 4 GB laptop GPU. Smaller patches and model width reduce memory use and may reduce accuracy. Training logs are written to `outputs/training_log.csv`; the best checkpoint is written to `checkpoints/best_model.pth`. These generated files are ignored by Git by default.

## Recorded quick run

A local quick run completed for 5 epochs using 100 subjects. Best mean validation Dice was 0.7660. This is a subset experiment, not a full-dataset result. The checkpoint and raw dataset are not included here.

## Project status

`train.py` contains the implemented training loop. `evaluate.py` and `infer.py` remain placeholders and need implementation before this is a complete end-to-end segmentation application.
