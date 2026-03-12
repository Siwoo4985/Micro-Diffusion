# Micro Diffusion

**Minimal text diffusion in Python.**

[![Python 3](https://img.shields.io/badge/Python-3-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-required-013243.svg?logo=numpy)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-optional-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

Karpathy's [MicroGPT](https://karpathy.github.io/2026/02/12/microgpt/) showed how GPT works in a few hundred lines. This project aims for the same style of explanation, but for **discrete text diffusion**.

Built for readers who want to understand the core text-diffusion loop without a large framework, large dataset pipeline, or production training stack.

Quick links: [Quick Start](#quick-start) · [Files](#files) · [Dataset](#dataset) · [Reproducibility](#reproducibility) · [Limitations](#limitations)

## Why This Repo Exists

Most text-generation examples focus on autoregressive models. This repository shows the other path:

- Mask tokens instead of shifting them
- Denoise the full sequence instead of predicting strictly left to right
- Swap the denoiser architecture without changing the diffusion loop

It is best treated as a learning repo and maintenance-friendly reference implementation, not a benchmark or production system.

## Autoregressive vs Diffusion

| | Autoregressive (GPT, etc.) | Diffusion (This Project) |
|---|---|---|
| **Generation** | Left to right, one token at a time | All at once, refining from noise |
| **Attention** | Causal (can only look left) | Bidirectional (looks everywhere) |
| **Analogy** | Writing word by word | Solving a crossword puzzle |
| **Training** | Predict the next token | Predict the erased tokens |

## How It Works

Take the name `"emma"`:

```text
Forward (training - add noise by masking):
  t=0:   e m m a      <- clean
  t=25:  e _ m a      <- some letters masked
  t=50:  _ _ m _      <- more masked
  t=100: _ _ _ _      <- fully masked

Reverse (generation - remove noise by unmasking):
  t=100: _ _ _ _      <- start from all masked
  t=75:  _ m _ _      <- fill in confident guesses first
  t=50:  e m _ a      <- keep going
  t=0:   e m m a      <- done
```

Train: given masked text at noise level `t`, predict the original.  
Generate: start from all masks, then reveal the most confident predictions first.

## Files

| File | Denoiser | Dependencies | Notes |
|---|---|---|---|
| `train_minimal.py` | 2-layer MLP | NumPy | Smallest runnable version, now script-safe and CLI-driven. |
| `train_pure.py` | 3-layer MLP + skip | NumPy | More commentary, forward-process preview, temperature sweep. |
| `train.py` | 4-layer Transformer | PyTorch | Bidirectional transformer version with the same diffusion loop. |
| `names.txt` | - | - | 32,000 lowercase names from U.S. SSA-style data. |

All three share the same masking, denoising, and confidence-based unmasking idea.  
The NumPy scripts cap sequence length at 12, so they train on 31,979 names.  
The PyTorch script keeps a max length of 16 and uses all 32,000 names.

## Dataset

The training data is [`names.txt`](./names.txt), a list of 32,000 lowercase names using only `a-z`.

- Character vocabulary: 26 letters
- Special tokens: PAD and MASK
- NumPy variants: names longer than 12 characters are filtered out
- PyTorch variant: uses the full file with a max sequence length of 16

This keeps the tokenizer simple and makes it easy to inspect the full corruption and denoising process by eye.

## Quick Start

### Prerequisites

- Python 3
- `pip`
- For `train.py`, a PyTorch build compatible with your Python version and platform

### NumPy scripts

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 train_minimal.py
python3 train_pure.py
```

### PyTorch script

```bash
pip install -r requirements.txt
python3 train.py
```

If PyTorch is missing, `train.py` now exits with a direct install hint instead of a raw import traceback.

## Handy CLI Flags

Each training script supports a small CLI for quick experiments:

```bash
python3 train_minimal.py --steps 100 --samples 5 --seed 7
python3 train_pure.py --steps 100 --samples 5 --no-temperature-sweep
python3 train.py --steps 100 --samples 5 --no-forward-preview
```

Useful options:

- `--steps`: override training iterations
- `--batch-size`: override batch size
- `--samples`: number of generated names
- `--temperature`: primary sampling temperature
- `--seed`: make runs repeatable
- `--quiet`: reduce logs

## Expected Output

Each script trains for a short run, then prints generated names. The exact output changes from run to run, but the structure is always:

```text
Training...
...
Generated names:
  amira
  kayna
  noria
```

For quick maintenance checks, reduce steps and sample count:

```bash
python3 train_minimal.py --steps 10 --samples 3 --seed 7
python3 train_pure.py --steps 10 --samples 3 --seed 7 --no-temperature-sweep
python3 train.py --steps 10 --samples 3 --seed 7 --no-temperature-sweep
```

## Maintenance

This repository is intentionally small, so the main maintenance risks are environmental drift and script behavior drift.

Current baseline:

- `requirements.txt` installs the shared NumPy baseline plus PyTorch for the transformer path.
- Scripts no longer start training just because they were imported.
- `train_minimal.py` is compatible with older Python runtimes that do not support dictionary union syntax.
- `train.py` avoids the duplicate temperature-sweep sampling call that previously doubled part of the runtime.

## Reproducibility

This project is intentionally lightweight, so reproducibility is partial rather than strict.

- All scripts support `--seed`
- NumPy runs are easy to make repeatable on the same machine
- PyTorch runs may still vary across devices and torch builds
- There is currently no checkpoint saving, evaluation harness, or experiment tracking

If you plan to extend this repository, the next maintenance step is usually adding saved checkpoints and a small regression harness for generated outputs.

## Concepts

**Discrete diffusion.** Image diffusion adds Gaussian noise to pixels. Text is discrete, so this project uses masking instead: replace tokens with `[MASK]`. This is an absorbing-state diffusion process.

**Cosine schedule.** Tokens are masked slowly at first, then faster later. This tends to work better than a constant masking rate.

**Bidirectional attention.** GPT uses causal masking. Diffusion models can look at both left and right context because they refine the whole sequence together.

**Confidence-based unmasking.** Each reverse step reveals only the predictions the model is most confident about. Uncertain positions stay masked until later.

**Temperature.** Lower values stay conservative. Higher values explore more and break down sooner.

## Architecture

MLP versions (`train_minimal.py`, `train_pure.py`):

```text
one-hot(noisy tokens) + timestep -> Linear -> ReLU -> Linear -> logits
```

Transformer version (`train.py`):

```text
Token Embed + Pos Embed + Time Embed -> Transformer x 4 -> logits
```

The diffusion loop is swappable. The denoiser could be an MLP, transformer, CNN, or something else.

## Toy vs Production

| | Here | Production (MDLM, SEDD, etc.) |
|---|---|---|
| Data | 32K names | Billions of tokens |
| Vocab | 28 (a-z + pad + mask) | 32K-100K BPE |
| Model | Small MLP / 4-layer Transformer | 12-48 layer Transformer |
| Params | 170K-239K | 100M-10B |
| Training | Minutes, CPU | Days or weeks, GPU cluster |

Same core loop: mask -> denoise -> unmask.

## Why Care About Text Diffusion

Autoregressive models dominate, but diffusion can:

- Generate all tokens in parallel
- Edit any part of text, not just append
- Control what goes where more easily
- Generate in an order other than left to right

Quality still trails strong autoregressive models, but the gap has narrowed.

## Limitations

- Character-level names only, not wordpiece or BPE tokenization
- No validation split, held-out metrics, or benchmark reporting
- No checkpointing or resume support
- No packaged module layout yet; the project is still script-first
- README examples are educational, not guarantees of sample quality

This is deliberate. The repo optimizes for readability and hackability over completeness.

## References

- [D3PM](https://arxiv.org/abs/2107.03006) - Austin et al., 2021
- [Diffusion-LM](https://arxiv.org/abs/2205.14217) - Li et al., 2022
- [MDLM](https://arxiv.org/abs/2406.07524) - Sahoo et al., 2024
- [SimpleDM](https://arxiv.org/abs/2406.04329) - Shi et al., 2024
- [MicroGPT](https://karpathy.github.io/2026/02/12/microgpt/) - Karpathy, inspiration for this project

## License

MIT
