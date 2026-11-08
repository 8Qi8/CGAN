# CGANet

Source code accompanying **CGANet: Camouflage Generation via Adversarial Feature Fusion for Anti-Detection in Remote Sensing Images**.

This repository contains the supplied model implementation, training scripts, and image-generation inference code. The original model classes and checkpoint names are retained for compatibility with the research implementation.

## Code and data availability

The source code is publicly available in this repository. The research data are available from the corresponding author upon reasonable request. Images, masks, trained weights, and generated experimental outputs are not bundled with this source-code release.

The paper uses a test set of **418 instances**. The repository does not automatically reconstruct this split: reproducing the paper's results requires the corresponding data, split and pairing information, checkpoints, and evaluation settings. The additional diffusion-baseline and forensic-model evaluation pipelines are not included in this repository.

## Environment

The environment recorded with the supplied implementation uses Python 3.8, PyTorch 1.11.0, and torchvision 0.12.0. Training and inference require an NVIDIA GPU and a compatible CUDA-enabled PyTorch installation; the original implementation selects `cuda:0`.

For the recorded CUDA 11.3 environment:

```bash
conda create -n cganet python=3.8
conda activate cganet
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -r requirements.txt
```

## Files

| File | Purpose |
| --- | --- |
| `net.py` | Encoder, generator modules, decoder, losses, and discriminator |
| `function.py` | Local feature normalization and appearance-transfer operations |
| `train.py` | Original training entry point with checkpoint saving |
| `test.py` | Original generation and masked-compositing entry point |
| `infer.py` | Launcher for user-supplied images; prepares output folders and full-image crop arguments |
| `sampler.py` | Training sampler |
| `params_position.py` | Original example placement presets |
| `netsmall.py`, `train_w.py` | Additional supplied research variants |

The model implementation files are preserved as supplied. `infer.py` calls the original `test.py` without changing the network or its generation procedure.

## Required checkpoints

Place the matching weights in the following locations, or pass their paths to the launcher:

```text
models/
  vgg_normalised.pth
experiments/
  decoder_iter_80000.pth
  PSF_iter_80000.pth
  NFFM_iter_80000.pth
  FEFM_iter_80000.pth
```

The normalized VGG weights must match the encoder defined in `net.py`. The module weights must come from the corresponding trained CGANet checkpoint; unrelated weights are not interchangeable.

## Generate an image

Prepare an RGB foreground image, a matching single-channel mask, and an RGB background image. The foreground and mask must have the same dimensions. Nonzero mask pixels indicate the foreground object. Generation preserves the original implementation's resizing and centered placement.

```bash
python infer.py --foreground /path/to/foreground.jpg --mask /path/to/mask.png --background /path/to/background.jpg --weights-dir experiments --vgg models/vgg_normalised.pth --output output
```

Run `python infer.py --help` to see the arguments. The launcher writes PNG files in:

- `output/images/`: generated composites;
- `output/original/`: direct-copy composites;
- `output/mask/`: target masks.

To use another visible GPU, configure `CUDA_VISIBLE_DEVICES` before launching the script. The process still addresses the selected device as `cuda:0`.

## Train

Prepare the foreground, mask, and background folders according to the original data organization. Keep data and checkpoint files outside version control.

```bash
python train.py --fore_dir /path/to/train/fore --mask_dir /path/to/train/mask --back_dir /path/to/train/back --vgg models/vgg_normalised.pth --save_dir experiments --log_dir logs --max_iter 80000 --batch_size 4 --n_threads 0
```

The command uses `train.py`, which saves module checkpoints every 10,000 iterations by default. The supplied implementation retains its original data-sampling and randomization behavior; no new train/validation/test partition is created by the training script.

## Evaluation scope

The generation scripts do not calculate the paper's image-quality, COD, or forensic metrics. Those evaluations require their respective fixed models and ground-truth masks. When evaluating generated images, ensure that target masks are aligned with the final output images. Multiple visual examples alone do not establish cross-scene generalization.
