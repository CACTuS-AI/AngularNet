# AngularNet

Research artifacts for the AngularNet NeurIPS paper. Release contents are being
prepared from the ggd-vision research project.

This branch contains the evaluation code for the trained GGM ResNet-18 / ResNet-34 models on the
TinyImageNet-200 and ImageNet-1k test sets. All commands below are run from the repository root.

## Setup

Linux or WSL2, Python 3.10 or newer, PyTorch 2.5 or newer (tested with 2.5 to 2.15).

1. Create an environment, with conda:

   ```bash
   conda create -n AngularNet python=3.11 -y
   conda activate AngularNet
   ```

   or with a plain virtual environment (on Ubuntu/Debian, `sudo apt install python3-venv` may
   be needed first):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install PyTorch for your system. Pick your OS, `pip` and your CUDA version (or CPU) at
   https://pytorch.org/get-started/locally/ and run the command shown there, for example:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # NVIDIA GPU (nvidia-smi shows CUDA 12.6 or higher)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu     # no NVIDIA GPU
   ```

3. Install the remaining packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Check the installation (on a GPU machine this should print `GPU: True`):

   ```bash
   python -c "import torch; print(torch.__version__, 'GPU:', torch.cuda.is_available())"
   ```

## Quick start

```bash
bash run_eval.sh tinyimagenet resnet18_ggm_1bit_N1
bash run_eval.sh imagenet resnet34_ggm_1bit_N1
bash run_eval.sh tinyimagenet all                # every checkpoint of one dataset
bash run_eval.sh list                            # all models
```

Checkpoints are read from `checkpoints/<dataset>/<model>.pt` and test sets from
`data/<dataset>/`.

## Options

```
bash run_eval.sh <tinyimagenet|imagenet> <model|all|list> [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--device cuda` / `--device cpu` | `cuda` if a GPU is available, else `cpu` | where to run |
| `--batch_size N` | 64 | images per batch; lower it if the GPU runs out of memory |
| `--limit_batches N` | 0 (all) | stop after N batches, as a quick check that everything runs; the result is not comparable to a full run |
| `--num_workers N` | min(8, number of CPUs) | processes that load images |
| `--data_root DIR` | `data/<dataset>` | where the test set is (TinyImageNet is downloaded there) |
| `--checkpoint_dir DIR` | `checkpoints/<dataset>` | where the `.pt` files are |
| `--no_download` | off | TinyImageNet: stop with an error instead of downloading the test set |
| `--json_out FILE` | none | also write the results to a JSON file |

Examples:

```bash
bash run_eval.sh tinyimagenet resnet18_ggm_1bit_N1 --limit_batches 2      # quick check, 2 batches
bash run_eval.sh tinyimagenet resnet18_ggm_1bit_N1 --device cpu             # run on the CPU
bash run_eval.sh imagenet all --batch_size 32 --json_out results.json
bash run_eval.sh imagenet resnet34_ggm_1bit_N1 --data_root /datasets/imagenet
python evaluate.py --help                                                    # full option list
```

## Models

| Dataset | Model | Architecture | Bits (act / weight) | N |
|---|---|---|---|---:|
| tinyimagenet | `resnet18_ggm_1bit_N1` | ResNet-18 | 1 / 1 | 1 |
| tinyimagenet | `resnet18_ggm_1.58bit_N1` | ResNet-18 | 1.58 / 1.58 | 1 |
| tinyimagenet | `resnet18_ggm_1bit_N2` | ResNet-18 | 1 / 1 | 2 |
| tinyimagenet | `resnet18_ggm_1bit_N3` | ResNet-18 | 1 / 1 | 3 |
| tinyimagenet | `resnet34_ggm_1bit_N1` | ResNet-34 | 1 / 1 | 1 |
| tinyimagenet | `resnet34_ggm_1.58bit_N1` | ResNet-34 | 1.58 / 1.58 | 1 |
| tinyimagenet | `resnet34_ggm_1bit_N2` | ResNet-34 | 1 / 1 | 2 |
| imagenet | `resnet18_ggm_1bit_N1` | ResNet-18 | 1 / 1 | 1 |
| imagenet | `resnet34_ggm_1bit_N1` | ResNet-34 | 1 / 1 | 1 |

1.58-bit means ternary {-1, 0, +1} quantization. N is the projection factor: each GGM layer uses
N x (input channels x kernel height x kernel width) random projection directions.

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

4. Put both files in `data/imagenet/`:

   ```bash
   mkdir -p data/imagenet
   mv ~/Downloads/ILSVRC2012_img_val.tar ~/Downloads/ILSVRC2012_devkit_t12.tar.gz data/imagenet/
   ```

5. Run `bash run_eval.sh imagenet <model>`. On the first run the archives are checked against the
   MD5 sums above and unpacked into `data/imagenet/val/<wnid>/<image>.JPEG` (1000 class folders).
   This needs about 13 GB of free disk space; afterwards `ILSVRC2012_img_val.tar` can be deleted.

An existing copy in the `val/<wnid>/<image>.JPEG` layout can be used directly with
`--data_root <dir>` (the directory containing `val/`). If an earlier unpacking was interrupted,
delete `data/imagenet/val` and run again.



## Files

```
run_eval.sh               one-command entry point
evaluate.py               evaluation CLI (python evaluate.py --help)
requirements.txt
checkpoints/<dataset>/    <model>.pt
ggm_eval/
  projections.py          G from seeds
  tinyimagenet/           GGM convolution, ResNet, checkpoint loading, test set download
  imagenet/               GGM convolution, ResNet (timm), checkpoint loading, test set loading
```

## License

Original contributions use Oregon State University's
[License Notice for Free Academic Research Use](LICENSE).
The notice specifies permitted research and internal-evaluation uses and the
contact for commercial-use requests. Applicable third-party notices will be
retained for any third-party material included in the release.
