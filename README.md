# BraTS 2023 GLI segmentation training

A Windows-ready Swin UNETR training project for BraTS 2023 Adult Glioma (GLI). It loads four MRI modalities (T1, T1 contrast-enhanced, T2, and FLAIR) and predicts background, NCR/NET, edema, and enhancing tumor. ET, TC, and WT Dice scores are computed from those predictions.

## Dataset access

The BraTS 2023 GLI dataset is hosted on Synapse: https://www.synapse.org/Synapse:syn51156910. Each collaborator must obtain their own authorized access and accept the applicable data terms. The raw scans are not included in this GitHub repository.

After downloading and extracting the archive, put the subject folders here:

```text
data/BraTS-GLI/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
```

Each subject folder should contain `-t1n.nii.gz`, `-t1c.nii.gz`, `-t2w.nii.gz`, `-t2f.nii.gz`, and `-seg.nii.gz` files using the subject ID as the filename prefix. Dataset attribution: BraTS 2023 Challenge, Synapse ID `syn51156910`.

## Setup (Windows PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For NVIDIA GPU training, install the CUDA-enabled PyTorch build compatible with your driver from the official selector: https://pytorch.org/get-started/locally/.

## Check the data and train

```powershell
.\.venv\Scripts\python.exe -m datasets.test_loader --data-dir "data\BraTS-GLI\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
.\.venv\Scripts\python.exe train.py --smoke-test
.\.venv\Scripts\python.exe train.py --epochs 5 --batch-size 1 --max-subjects 100
```

The trainer uses 64 x 64 x 64 patches, feature size 24, and mixed precision on CUDA to fit a 4 GB laptop GPU. These memory-saving settings can affect accuracy.

## Included training record

`training_summary.csv` and `outputs/training_log.csv` contain the five-epoch run results. This experiment used a 100-subject subset; it is not a full-dataset result. Best mean validation Dice was 0.7660 (epoch 3). The trained model checkpoint was saved locally during the run, but is not included in this repository.

| Epoch | Train loss | ET Dice | TC Dice | WT Dice |
|---:|---:|---:|---:|---:|
| 1 | 0.9690 | 0.6111 | 0.5844 | 0.7090 |
| 2 | 0.8687 | 0.6325 | 0.5398 | 0.7123 |
| 3 | 0.8027 | 0.7837 | 0.6985 | 0.8158 |
| 4 | 0.7649 | 0.7646 | 0.6826 | 0.7922 |
| 5 | 0.7555 | 0.7363 | 0.6998 | 0.8155 |

## Project status

`train.py` contains the training loop. `evaluate.py` and `infer.py` are placeholders, so evaluation and prediction workflows still need implementation before this is a complete end-to-end segmentation application.
