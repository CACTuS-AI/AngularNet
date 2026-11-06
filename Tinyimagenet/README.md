# TinyImageNet evaluation for GGM ResNets

Evaluation code for the trained GGM ResNet-18 / ResNet-34 models on the TinyImageNet-200
test set.

## Quick start

```bash
pip install -r requirements.txt            # for a GPU, install the CUDA build of PyTorch first
bash run_eval.sh resnet18_ggm_1bit_N1      # evaluate one model
bash run_eval.sh all                       # evaluate every checkpoint in checkpoints/
bash run_eval.sh list                      # show the available models
```

On first use the test set is downloaded to `data/` (about 35 MB, see below). Checkpoints are
read from `checkpoints/<model>.pt`. Extra arguments go to `evaluate.py`, for example
`bash run_eval.sh resnet34_ggm_1bit_N2 --batch_size 32` or `--device cpu`.

## Models

| Model | Architecture | Bits (act / weight) | N |
|---|---|---|---:|
| `resnet18_ggm_1bit_N1` | ResNet-18 | 1 / 1 | 1 |
| `resnet18_ggm_1.58bit_N1` | ResNet-18 | 1.58 / 1.58 | 1 |
| `resnet18_ggm_1bit_N2` | ResNet-18 | 1 / 1 | 2 |
| `resnet18_ggm_1bit_N3` | ResNet-18 | 1 / 1 | 3 |
| `resnet34_ggm_1bit_N1` | ResNet-34 | 1 / 1 | 1 |
| `resnet34_ggm_1.58bit_N1` | ResNet-34 | 1.58 / 1.58 | 1 |
| `resnet34_ggm_1bit_N2` | ResNet-34 | 1 / 1 | 2 |

1.58-bit means ternary {-1, 0, +1} quantization. N is the projection factor: each GGM layer
uses N x (input channels x kernel height x kernel width) random projection directions.

## Test set

TinyImageNet's own `test/` split has no public labels, so the labelled `val/` split (10,000
images, 50 per class) is used as the test set. `run_eval.sh` downloads only that split from
the official archive (`https://cs231n.stanford.edu/tiny-imagenet-200.zip`) using HTTP range
requests, about 35 MB instead of 248 MB. If the server does not allow range requests, it
downloads the whole archive, extracts `val/` and `wnids.txt`, and deletes the archive. An
existing copy can be used with `--data_root <dir>`, where `<dir>` contains
`tiny-imagenet-200/` (with `wnids.txt` and `val/`, in the original layout or sorted into one
folder per class).

## Evaluation settings

On a GPU the defaults are bf16 autocast, TF32, channels-last memory format, deterministic
cuDNN and batch 64, with the test images in a fixed order (class, then file name). On a CPU
the evaluation runs in fp32.

Reported metrics: top-1 and top-5 accuracy, and cross-entropy with label smoothing 0.1.

## Files

```
run_eval.sh          one-command entry point
evaluate.py          evaluation CLI (python evaluate.py --help)
requirements.txt
checkpoints/         <model>.pt
ggm_eval/
  ggm_conv.py        GGM convolution, forward pass only
  resnet.py          ResNet-18/34 with a 3x3 stride-1 stem for 64x64 inputs
  projections.py     G from seeds
  checkpoint.py      checkpoint loading
  data.py            TinyImageNet test set download and loading
  models.py          model list
```
