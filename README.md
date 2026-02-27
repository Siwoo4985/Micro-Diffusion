# Micro Diffusion

**A minimal discrete text diffusion model for learning.**

Like Karpathy's [MicroGPT](https://karpathy.github.io/2026/02/12/microgpt/) showed the algorithmic essence of GPT in ~200 lines, **Micro Diffusion** shows the essence of **text diffusion models** — a fundamentally different approach to text generation.

## Autoregressive vs Diffusion: The Core Idea

|  | Autoregressive (GPT, etc.) | Diffusion (This Project) |
|--|---------------------|-------------------------|
| **Generation** | Left → Right, one token at a time | All tokens at once, refining from noise |
| **Attention** | Causal (can only look left) | Bidirectional (can look everywhere) |
| **Analogy** | Writing a sentence word by word | Sculptor revealing a statue from marble |
| **Training** | "Given previous tokens, predict next" | "Given partially erased text, predict original" |

## How Text Diffusion Works

Imagine the name `"emma"` written on a chalkboard:

```
Forward Process (adding noise — used during training):
  t=0:   e m m a      ← clean (original)
  t=25:  e _ m a      ← some letters erased
  t=50:  _ _ m _      ← more erased
  t=75:  _ _ _ _      ← almost all erased
  t=100: _ _ _ _      ← fully erased (pure noise)

Reverse Process (removing noise — used during generation):
  t=100: _ _ _ _      ← start from blank chalkboard
  t=75:  _ m _ _      ← model guesses most confident letters
  t=50:  e m _ a      ← more letters revealed
  t=25:  e m m a      ← almost done
  t=0:   e m m a      ← clean result!
```

The model learns: **"Given partially erased text at noise level t, predict the original letters."**

Then at generation time, it starts from a fully blank slate and iteratively fills in letters — **most confident guesses first**, like solving a crossword puzzle.

## What's Included

| File | Lines | Denoiser | Dependencies | Description |
|------|-------|----------|-------------|-------------|
| `train_minimal.py` | **143** | 2-layer MLP | NumPy | The essence of text diffusion. Nothing more. |
| `train_pure.py` | 292 | 3-layer MLP + skip | NumPy | Commented educational version with visualization. |
| `train.py` | 413 | Transformer (4-layer) | PyTorch | Full bidirectional Transformer denoiser. |
| `names.txt` | — | — | — | 32,000 English names from U.S. SSA data. |

All three files implement the **same diffusion algorithm**. Only the denoiser architecture differs — because the diffusion mechanism is architecture-agnostic.

## Quick Start

### Minimal Version (143 lines, zero framework)

```bash
python3 train_minimal.py
```

### Educational Version (more comments, visualization)

```bash
python3 train_pure.py
```

### Transformer Version

```bash
pip install torch
python3 train.py
```

All will:
1. Show the forward diffusion process (progressive masking)
2. Train the denoiser (~1-5 minutes on CPU)
3. Generate new names via reverse diffusion

## Example Output

### Forward Process
```
Forward Process: "raylynn"
  t=  0 (mask   0.0%): raylynn
  t=  5 (mask   4.2%): raylynn
  t= 10 (mask  15.3%): r_ylynn
  t= 20 (mask  50.6%): ____y_n
  t= 30 (mask  85.6%): _a_l_n_
  t= 40 (mask 100.0%): _______
```

### Training
```
step     0 | loss 3.3375
step  1000 | loss 1.5489
step  3000 | loss 1.5611
step  4999 | loss 1.3324
```

### Reverse Diffusion (Generation)
```
  t=50 (  0.0%): _______________      ← pure noise
  t=37 ( 26.0%): _or_a                ← confident letters first
  t=25 ( 50.0%): noria                ← filling in
  t= 1 ( 98.0%): noria                ← done!
```

### Generated Names

```
train_minimal.py : timea, zaniya, juno, amira, mana, harin, daren
train_pure.py    : kayana, marina, kalina, maren, maria, damira, casiana
train.py         : noria, ava, randi, erynn, zaynna, lalisa, branyl
```

### Temperature Comparison (train.py)
```
Temperature 0.5: mara, ralya, lenah, kal, mal
Temperature 0.8: noria, ava, randi, erynn, lalisa
Temperature 1.0: rahany, korianth, maheon
Temperature 1.5: aridpye, nzllauae, diryta
```

## Key Concepts You'll Learn

### 1. Discrete Diffusion (Absorbing State)
Unlike image diffusion that adds Gaussian noise to continuous pixels, text diffusion works with **discrete tokens**. The noise process is **masking** — replacing tokens with a special `[MASK]` token. This is the "absorbing state" because once masked, a token stays masked (in the forward process).

### 2. Noise Schedule
Not all noise levels are equally useful. A **cosine schedule** adds noise gradually at first (preserving structure) and aggressively later. This produces better results than a naive linear schedule.

### 3. Bidirectional Attention
GPT must use **causal masking** because it generates left-to-right — position 5 can't peek at position 6. Diffusion models see **everything at once** because they refine all positions simultaneously. This is why the Transformer in `train.py` uses bidirectional (BERT-style) attention.

### 4. Confidence-Based Unmasking
During generation, the model predicts all positions but only **unmasks the most confident** ones at each step. Less confident positions stay masked for later refinement. This is like solving a crossword — fill in what you're sure about first.

### 5. Temperature
Controls the randomness of generation:
- **Low (0.5):** Conservative, picks high-probability tokens → common, safe names
- **Medium (0.8):** Balanced creativity and coherence
- **High (1.5):** Wild, explores unlikely tokens → unusual, sometimes broken names

## Architecture Comparison

### `train_minimal.py` / `train_pure.py` (NumPy — MLP Denoiser)
```
Input: one-hot(noisy tokens) + timestep → flat vector
  ↓ Linear + ReLU
  ↓ Linear
Output: logits (max_len × vocab_size)
```

### `train.py` (PyTorch — Transformer Denoiser)
```
Input: [MASK, e, MASK, MASK, a, PAD, ...] + timestep t
  ↓ Token Embedding + Position Embedding + Timestep Embedding
  ↓ Transformer Block × 4 (bidirectional self-attention + MLP)
  ↓ Output projection
Output: logits (max_len × vocab_size)
```

The **diffusion training/sampling loop is identical** across all three. Only the denoiser differs — because the denoiser is a pluggable component. You could swap in a CNN, RNN, or any architecture; the diffusion algorithm stays the same.

## Scaling Up (What Production Models Change)

This is a toy model. Production text diffusion models (like MDLM, SEDD, Plaid) change:

| Aspect | Micro Diffusion | Production |
|--------|----------------|-----------|
| Data | 32,000 names (SSA) | Billions of text tokens |
| Vocab | 28 (a-z + pad + mask) | 32K-100K (BPE subwords) |
| Model | 2-layer MLP / 4-layer Transformer | 12-48 layer Transformer |
| Parameters | 170K-206K | 100M-10B |
| Training | Minutes on CPU | Days/weeks on GPU clusters |
| Sequence length | 12-16 characters | 512-4096 tokens |

The **core algorithm** (mask → denoise → unmask) stays the same.

## Why Text Diffusion Matters

Autoregressive models (GPT) dominate text generation today, but diffusion offers unique advantages:

- **Parallel generation**: All tokens generated simultaneously (potentially faster)
- **Natural editing**: Can mask and regenerate any part of text (not just append)
- **Controllable generation**: Easier to guide what gets generated where
- **No left-to-right bias**: Generation order is by confidence, not position

The tradeoff: diffusion models for text are still catching up to GPT in raw quality, but the gap is closing rapidly.

## References

- [Structured Denoising Diffusion Models (D3PM)](https://arxiv.org/abs/2107.03006) — Austin et al., 2021
- [Diffusion-LM](https://arxiv.org/abs/2205.14217) — Li et al., 2022
- [Masked Diffusion Language Models (MDLM)](https://arxiv.org/abs/2406.07524) — Sahoo et al., 2024
- [Simple and Effective Masked Diffusion Models (SimpleDM)](https://arxiv.org/abs/2406.04329) — Shi et al., 2024
- [MicroGPT by Karpathy](https://karpathy.github.io/2026/02/12/microgpt/) — The inspiration for this project

## License

MIT
