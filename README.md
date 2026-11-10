# AngularNet

Research artifacts for the AngularNet NeurIPS paper. Release contents are being
prepared from the ggd-vision research project.

This repository contains the evaluation code and checkpoints for the trained GGM models:

- ResNet-18 / ResNet-34 on the TinyImageNet-200 and ImageNet-1k test sets
- ResNet-18 and VGG-16 on the CIFAR-10 test set
- Medformer on the ADFTD (EEG) and PTB-XL (ECG) test sets
- BERT-base (BiBERT setup) on the SST-2 and QNLI dev sets of GLUE

All commands below are run from the repository root.

## Setup

Windows or Linux (including WSL2), Python 3.11 or newer.

1. Get the code:

   ```bash
   git clone https://github.com/CACTuS-AI/AngularNet.git
   cd AngularNet
   ```

2. Create an environment, with conda:

   ```bash
   conda create -n AngularNet python=3.11 -y
   conda activate AngularNet
   ```

   or with a plain virtual environment (on Ubuntu/Debian, `sudo apt install python3-venv` may
   be needed first):

   ```bash
   python3 -m venv .venv            # Linux
   source .venv/bin/activate

   python -m venv .venv             # Windows
   .venv\Scripts\activate
   ```

3. Install the packages, including PyTorch. This one command installs a PyTorch build that uses an
   NVIDIA GPU when one is available and the CPU otherwise:

   ```bash
   pip install -r requirements.txt
   ```

4. Check the installation (on a GPU machine this should print `GPU: True`):

   ```bash
   python -c "import torch; print(torch.__version__, 'GPU:', torch.cuda.is_available())"
   ```

   If it prints `GPU: False` on a machine with an NVIDIA GPU, the NVIDIA driver is older than the
   installed PyTorch build needs. Update the driver, or install `torch` and `torchvision` with the
   command for your driver from https://pytorch.org/get-started/locally/ and run step 3 again.
   Without a GPU, evaluation runs on the CPU, only more slowly.

## Quick start

One command per dataset and model (1-bit, N = 1). The command format is
`python evaluate.py <dataset> <model>`.

**TinyImageNet-200** (the test set is downloaded automatically on first use):

```bash
python evaluate.py tinyimagenet resnet18_ggm_1bit_N1
python evaluate.py tinyimagenet resnet34_ggm_1bit_N1
```

**ImageNet-1k** (the validation set has to be downloaded by hand first, see
[Test sets](#test-sets)):

```bash
python evaluate.py imagenet resnet18_ggm_1bit_N1
python evaluate.py imagenet resnet34_ggm_1bit_N1
```

**CIFAR-10** (the test set is downloaded automatically on first use):

```bash
python evaluate.py cifar10 resnet18_ggm_1bit_N1
python evaluate.py cifar10 vgg16_ggm_1bit_N1
```

**ADFTD** (the dataset is downloaded automatically on first use, 2.6 GB):

```bash
python evaluate.py adftd medformer_ggm_1bit
```

**PTB-XL** (the dataset is downloaded automatically on first use, 3.3 GB):

```bash
python evaluate.py ptbxl medformer_ggm_1bit
```

**SST-2** (the dev set is downloaded automatically on first use):

```bash
python evaluate.py sst2 bert_ggm_1bit_N1
```

**QNLI** (the dev set is downloaded automatically on first use):

```bash
python evaluate.py qnli bert_ggm_1bit_N1
```

The 1.58-bit checkpoints (`resnet18_ggm_1.58bit_N1`, `resnet34_ggm_1.58bit_N1`) are run the same
way. To list the names of all models, or to run every checkpoint of a dataset:

```bash
python evaluate.py list
python evaluate.py tinyimagenet all
```

With bash (Linux, macOS, WSL), `bash run_eval.sh` can be used in place of `python evaluate.py`,
with the same arguments, e.g. `bash run_eval.sh cifar10 vgg16_ggm_1bit_N1`.

Checkpoints are read from `checkpoints/<dataset>/<model>.pt` and test sets from
`data/<dataset>/`. In the checkpoints, the binary weights of the GGM layers are stored as packed bits
(1 bit per weight; 2 bits per weight for the 1.58-bit models), and each projection matrix G is stored
as the seed it is rebuilt from when the model is loaded.

## Options

```
python evaluate.py <tinyimagenet|imagenet|cifar10|adftd|ptbxl|sst2|qnli> <model|all|list> [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--device cuda` / `--device cpu` | `cuda` if a GPU is available, else `cpu` | where to run |
| `--batch_size N` | 64 (images), 128 (ADFTD, PTB-XL), 32 (SST-2, QNLI) | images or samples per batch; lower it if the GPU runs out of memory |
| `--limit_batches N` | 0 (all) | stop after N batches, as a quick check that everything runs; the result is not comparable to a full run |
| `--num_workers N` | min(8, number of CPUs) | processes that load images (image datasets only) |
| `--data_root DIR` | `data/<dataset>` | where the test set is (test sets that are downloaded automatically go there) |
| `--checkpoint_dir DIR` | `checkpoints/<dataset>` | where the `.pt` files are |
| `--no_download` | off | stop with an error instead of downloading the test set (all datasets except ImageNet) |
| `--json_out FILE` | none | also write the results to a JSON file |

Examples:

```bash
python evaluate.py tinyimagenet resnet18_ggm_1bit_N1 --limit_batches 2      # quick check, 2 batches
python evaluate.py cifar10 resnet18_ggm_1bit_N1 --device cpu                  # run on the CPU
python evaluate.py imagenet all --batch_size 32 --json_out results.json
python evaluate.py imagenet resnet34_ggm_1bit_N1 --data_root /datasets/imagenet
python evaluate.py adftd medformer_ggm_1bit --data_root /datasets/ADFTD
python evaluate.py --help                                                      # full option list
```

## Test sets

**TinyImageNet-200.** Its own `test/` split has no public labels, so the labelled `val/` split
(10,000 images) is used as the test set. It is downloaded automatically on first use: only that
split is fetched from the official archive (`https://cs231n.stanford.edu/tiny-imagenet-200.zip`)
using HTTP range requests, about 35 MB instead of 248 MB; if the server does not allow range
requests, the whole archive is downloaded, `val/` and `wnids.txt` are extracted, and the archive
is deleted.

**ImageNet-1k.** ImageNet cannot be downloaded automatically: its terms of access require each
user to register and accept them. Only the validation set (50,000 images, used as the test set)
and the development kit are needed, not the training set.

1. Create an account at https://image-net.org/signup.php and wait for it to be approved.
2. Log in and open the ILSVRC2012 download page,
   https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
   (also reachable from https://image-net.org/download-images.php).
3. Download these two files:

   | File | Section on the page | Size | MD5 |
   |---|---|---:|---|
   | `ILSVRC2012_img_val.tar` | Validation images (all tasks) | 6.3 GB | `29b22e2961454d5413ddabcf34fc5622` |
   | `ILSVRC2012_devkit_t12.tar.gz` | Development kit (Task 1 & 2) | 2.5 MB | `fa75699e90414af021442c21a62c3abf` |

4. Create the folder `data/imagenet/` in the repository and move both files into it.

5. Run `python evaluate.py imagenet <model>`. On the first run the archives are checked against the
   MD5 sums above and unpacked into `data/imagenet/val/<wnid>/<image>.JPEG` (1000 class folders).
   This needs about 13 GB of free disk space; afterwards `ILSVRC2012_img_val.tar` can be deleted.

An existing copy in the `val/<wnid>/<image>.JPEG` layout can be used directly with
`--data_root <dir>` (the directory containing `val/`). If an earlier unpacking was interrupted,
delete `data/imagenet/val` and run again.

**CIFAR-10.** The test split (10,000 images) is downloaded automatically on first use by
torchvision from `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` (163 MB) into
`data/cifar10/`. An existing copy can be used with `--data_root <dir>` (the directory containing
`cifar-10-batches-py/`).

**ADFTD and PTB-XL.** The processed datasets of Medformer are downloaded automatically on first use
from Google Drive (ADFTD 2.6 GB, PTB-XL 3.3 GB) and unpacked into `data/adftd/` and `data/ptbxl/`;
the archive is deleted afterwards. If Google Drive refuses the download (it limits how often a file
can be downloaded per day), download them by hand instead:

1. Download the processed datasets:

   | Dataset | Link |
   |---|---|
   | ADFTD | https://drive.google.com/file/d/1QcX_M58IQUBn3lDBlVVL0SDN7_QI1vWe/view |
   | PTB-XL | https://drive.google.com/file/d/1whskRvTZUNb1Qph2SeXEdpcU2rQY0T1E/view |

2. Extract them so that each dataset directory contains `Feature/` and `Label/`:

   ```
   data/adftd/Feature/   data/adftd/Label/label.npy
   data/ptbxl/Feature/   data/ptbxl/Label/label.npy
   ```

   An existing copy elsewhere can be used with `--data_root <dir>` (the directory that contains
   `Feature/` and `Label/`).

**SST-2 and QNLI.** As in the paper, the models are evaluated on the GLUE dev splits (the test
splits have no public labels): 872 sentences for SST-2 and 5,463 question/sentence pairs for QNLI.
On first use `dev.tsv` is taken from the official GLUE archive
(`https://dl.fbaipublicfiles.com/glue/data/SST-2.zip`, 7 MB, and `.../QNLIv2.zip`, 11 MB),
checked against its SHA-256 and saved as `data/sst2/dev.tsv` and `data/qnli/dev.tsv`; the archive is
deleted afterwards. An existing copy can be used with `--data_root <dir>` (the directory containing
`dev.tsv`).

## Files

```
evaluate.py               evaluation entry point (python evaluate.py --help)
run_eval.sh               the same, for bash: bash run_eval.sh <arguments>
requirements.txt
checkpoints/<dataset>/    <model>.pt
ggm_eval/
  projections.py          projection matrices G, rebuilt from their seeds
  layers.py               GGM layers (2-D and 1-D convolutions, linear) with packed weights
  resnet.py               ResNets for TinyImageNet, ImageNet (timm) and CIFAR-10
  vgg.py                  VGG-16 for CIFAR-10
  medformer.py            Medformer for ADFTD and PTB-XL
  bert.py                 BERT-base for SST-2 and QNLI
  tokenization.py         BERT WordPiece tokenizer
  models.py               list of released models
  checkpoint.py           checkpoint loading
  data.py                 test sets (download and loading)
  metrics.py              ADFTD / PTB-XL metrics
```

## License

Original contributions use Oregon State University's
[License Notice for Free Academic Research Use](LICENSE).
The notice specifies permitted research and internal-evaluation uses and the
contact for commercial-use requests. Applicable third-party notices will be
retained for any third-party material included in the release.
