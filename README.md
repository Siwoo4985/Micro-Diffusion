# Micro Diffusion

**Minimal text diffusion in Python.**

Karpathy’s [MicroGPT](https://karpathy.github.io/2026/02/12/microgpt/) showed how GPT works in ~200 lines. This does the same for **text diffusion** — a different way to generate text.

## Autoregressive vs Diffusion

|              |Autoregressive (GPT, etc.)       |Diffusion (This Project)        |
|--------------|---------------------------------|--------------------------------|
|**Generation**|Left → Right, one token at a time|All at once, refining from noise|
|**Attention** |Causal (can only look left)      |Bidirectional (looks everywhere)|
|**Analogy**   |Writing word by word             |Solving a crossword puzzle      |
|**Training**  |“Predict next token”             |“Predict erased tokens”         |

## How It Works

Take the name `"emma"`:

```
Forward (training — add noise by masking):
  t=0:   e m m a      ← clean
  t=25:  e _ m a      ← some letters masked
  t=50:  _ _ m _      ← more masked
  t=100: _ _ _ _      ← fully masked

Reverse (generation — remove noise by unmasking):
  t=100: _ _ _ _      ← start from all masked
  t=75:  _ m _ _      ← fill in confident guesses first
  t=50:  e m _ a      ← keep going
  t=0:   e m m a      ← done
```

Train: “given masked text at noise level t, predict the original.”
Generate: start from all masks, unmask most confident predictions first.

## Files

|File              |Lines  |Denoiser             |Deps   |                             |
|------------------|-------|---------------------|-------|-----------------------------|
|`train_minimal.py`|**143**|2-layer MLP          |NumPy  |Bare minimum.                |
|`train_pure.py`   |292    |3-layer MLP + skip   |NumPy  |More comments, visualization.|
|`train.py`        |413    |Transformer (4-layer)|PyTorch|Bidirectional Transformer.   |
|`names.txt`       |—      |—                    |—      |32K names (U.S. SSA data).   |

Same diffusion algorithm in all three. Only the denoiser is different.

## Quick Start

```bash
# just numpy, no framework
python3 train_minimal.py

# more verbose, with visualization
python3 train_pure.py

# transformer version (needs pytorch)
pip install torch
python3 train.py
```

Trains in a few minutes on CPU. No GPU needed.

## Example Output

```
Forward Process: "raylynn"
  t=  0 (mask   0.0%): raylynn
  t= 10 (mask  15.3%): r_ylynn
  t= 20 (mask  50.6%): ____y_n
  t= 40 (mask 100.0%): _______

Reverse (generation):
  t=50: _______________    ← noise
  t=37: _or_a              ← unmasking
  t=25: noria              ← done
```

Generated names:

```
train_minimal.py : timea, zaniya, juno, amira, mana, harin, daren
train_pure.py    : kayana, marina, kalina, maren, maria, damira, casiana
train.py         : noria, ava, randi, erynn, zaynna, lalisa, branyl
```

Temperature:

```
0.5: mara, ralya, lenah, kal, mal           ← safe
0.8: noria, ava, randi, erynn, lalisa       ← balanced
1.0: rahany, korianth, maheon               ← adventurous
1.5: aridpye, nzllauae, diryta             ← falling apart
```

## Concepts

**Discrete diffusion.** Image diffusion adds Gaussian noise to pixels. Text is discrete, so we use masking instead — replace tokens with `[MASK]`. This is called “absorbing state” diffusion.

**Cosine schedule.** Masks tokens slowly at first, then faster. Works better than masking at a constant rate.

**Bidirectional attention.** GPT uses causal masking (can’t look right). Diffusion models look at all positions, since they refine everything at once.

**Confidence-based unmasking.** At each step, only reveal the predictions the model is most sure about. Less sure ones stay masked for later.

**Temperature.** Low = safe/common results. High = weird/creative results.

## Architecture

MLP version (`train_minimal.py`, `train_pure.py`):

```
one-hot(noisy tokens) + timestep → Linear → ReLU → Linear → logits
```

Transformer version (`train.py`):

```
Token Embed + Pos Embed + Time Embed → Transformer × 4 → logits
```

The diffusion loop is the same in both. The denoiser is swappable — MLP, Transformer, CNN, whatever.

## Toy vs Production

|        |Here                             |Production (MDLM, SEDD, etc.)|
|--------|---------------------------------|-----------------------------|
|Data    |32K names                        |Billions of tokens           |
|Vocab   |28 (a-z + pad + mask)            |32K-100K BPE                 |
|Model   |2-layer MLP / 4-layer Transformer|12-48 layer Transformer      |
|Params  |170K-206K                        |100M-10B                     |
|Training|Minutes, CPU                     |Days/weeks, GPU cluster      |

Same core loop: mask → denoise → unmask.

## Why Care About Text Diffusion

Autoregressive models dominate, but diffusion can:

- Generate all tokens in parallel
- Edit any part of text (not just append)
- Control what goes where more easily
- Generate in any order, not just left to right

Still behind autoregressive in quality, but getting closer.

## References

- [D3PM](https://arxiv.org/abs/2107.03006) — Austin et al., 2021
- [Diffusion-LM](https://arxiv.org/abs/2205.14217) — Li et al., 2022
- [MDLM](https://arxiv.org/abs/2406.07524) — Sahoo et al., 2024
- [SimpleDM](https://arxiv.org/abs/2406.04329) — Shi et al., 2024
- [MicroGPT](https://karpathy.github.io/2026/02/12/microgpt/) — Karpathy. Inspiration for this project.

## License

MIT