# Micro Diffusion

**Minimal text diffusion in Python.**

Karpathy's [MicroGPT](https://karpathy.github.io/2026/02/12/microgpt/) showed how GPT works in a few hundred lines. This project aims for the same style of explanation, but for **discrete text diffusion**.

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

## Quick Start

### NumPy scripts

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 train_minimal.py
python3 train_pure.py
```

### PyTorch script

Install a PyTorch build compatible with your Python version, then:

```bash
pip install -r requirements-torch.txt
python3 train.py
```

If PyTorch is missing, `train.py` now exits with a direct install hint instead of a raw import traceback.

## Handy CLI Flags

Each training script supports a small CLI for smoke tests and quick experiments:

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
- `--quiet`: reduce logs for smoke tests

## Maintenance

This repository is intentionally small, so the main maintenance risks are environmental drift and script behavior drift.

Current baseline:

- `requirements.txt` covers the NumPy-only paths.
- `requirements-torch.txt` layers PyTorch on top of the NumPy baseline.
- Scripts no longer start training just because they were imported.
- `train_minimal.py` is compatible with older Python runtimes that do not support dictionary union syntax.
- `train.py` avoids the duplicate temperature-sweep sampling call that previously doubled part of the runtime.

Run the smoke checks with:

```bash
python3 -m unittest discover -s tests
```

The test suite always checks Python compilation and the NumPy scripts.  
The PyTorch smoke test is skipped automatically when `torch` is not installed.

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

## References

- [D3PM](https://arxiv.org/abs/2107.03006) - Austin et al., 2021
- [Diffusion-LM](https://arxiv.org/abs/2205.14217) - Li et al., 2022
- [MDLM](https://arxiv.org/abs/2406.07524) - Sahoo et al., 2024
- [SimpleDM](https://arxiv.org/abs/2406.04329) - Shi et al., 2024
- [MicroGPT](https://karpathy.github.io/2026/02/12/microgpt/) - Karpathy, inspiration for this project

## License

MIT
