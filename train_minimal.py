"""
Micro Diffusion — Discrete Text Diffusion in a Small NumPy Script
=================================================================
The algorithmic essence of text diffusion, nothing more.

GPT:       left-to-right, one token at a time.
Diffusion: all tokens at once, from noise to text.

  Forward:  e m m a  ->  e _ m _  ->  _ _ _ _   (mask letters)
  Reverse:  _ _ _ _  ->  _ m _ a  ->  e m m a   (unmask by confidence)

python train_minimal.py
"""

import argparse
import math
import os
import random

import numpy as np

# --- Config ---
max_len = 12
hidden = 256
T = 40
steps = 5000
lr = 5e-4
B = 64

# --- Data & Tokenizer ---
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "names.txt")) as f:
    names = [l.strip().lower() for l in f if l.strip() and len(l.strip()) <= max_len]

chars = sorted(set("".join(names)))
PAD = len(chars)
MASK = len(chars) + 1
V = len(chars) + 2  # vocab size
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for c, i in c2i.items()}
i2c[PAD] = "."
i2c[MASK] = "_"


def encode(name):
    return np.array([c2i[c] for c in name] + [PAD] * (max_len - len(name)), dtype=np.int32)


def decode(ids):
    return "".join(i2c[int(i)] for i in ids).replace(".", "").replace("_", "")


data = np.stack([encode(n) for n in names])

# --- Forward Process: progressively mask tokens ---
def mask_rate(t):
    return 1.0 - math.cos(((t / T) + 0.008) / 1.008 * math.pi / 2) ** 2


def add_noise(x, t):
    m = np.random.rand(*x.shape) < mask_rate(t)
    noisy = x.copy()
    noisy[m] = MASK
    return noisy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def reset_model():
    global D, W1, b1, W2, b2
    global mW1, vW1, mb1, vb1, mW2, vW2, mb2, vb2

    # --- Denoiser: 2-layer MLP (architecture doesn't matter, diffusion does) ---
    D = max_len * V + 1  # input: flattened one-hot + timestep
    W1 = np.random.randn(D, hidden).astype(np.float32) * math.sqrt(2.0 / D)
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = np.random.randn(hidden, max_len * V).astype(np.float32) * math.sqrt(2.0 / hidden)
    b2 = np.zeros(max_len * V, dtype=np.float32)

    # --- Training state ---
    mW1 = np.zeros_like(W1)
    vW1 = np.zeros_like(W1)
    mb1 = np.zeros_like(b1)
    vb1 = np.zeros_like(b1)
    mW2 = np.zeros_like(W2)
    vW2 = np.zeros_like(W2)
    mb2 = np.zeros_like(b2)
    vb2 = np.zeros_like(b2)


def parameter_count():
    return W1.size + b1.size + W2.size + b2.size


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / (e.sum(-1, keepdims=True) + 1e-10)


def forward(x_ids, t):
    """Predict original tokens from noisy input at timestep t."""
    bs = x_ids.shape[0]
    oh = np.zeros((bs, max_len * V), dtype=np.float32)
    for i in range(bs):
        for j in range(max_len):
            oh[i, j * V + x_ids[i, j]] = 1.0
    x = np.concatenate([oh, np.full((bs, 1), t / T, dtype=np.float32)], 1)
    z = x @ W1 + b1
    h = np.maximum(z, 0)
    logits = (h @ W2 + b2).reshape(bs, max_len, V)
    return logits, (x, z, h)


def adam(p, g, m, v, s, total_steps):
    m[:] = 0.9 * m + 0.1 * g
    v[:] = 0.999 * v + 0.001 * g ** 2
    mh = m / (1 - 0.9 ** (s + 1))
    vh = v / (1 - 0.999 ** (s + 1))
    lr_t = lr * min(1.0, (s + 1) / 200.0) * max(0.1, 1 - s / float(max(total_steps, 1)))
    p -= lr_t * mh / (np.sqrt(vh) + 1e-8)


def train(num_steps=None, batch_size=None, report_every=500, verbose=True):
    actual_steps = steps if num_steps is None else num_steps
    actual_batch_size = B if batch_size is None else batch_size

    if verbose:
        print("Dataset: {0} names, vocab {1}".format(len(names), V))
        print("Model: {0:,} parameters".format(parameter_count()))
        print("\nTraining...")

    for step in range(actual_steps):
        x0 = data[np.random.randint(0, len(data), actual_batch_size)]
        t = random.randint(1, T)
        xt = add_noise(x0, t)
        logits, (x_in, z, h) = forward(xt, t)
        probs = softmax(logits)

        loss = -sum(
            math.log(max(probs[i, j, x0[i, j]], 1e-10))
            for i in range(actual_batch_size)
            for j in range(max_len)
        ) / float(actual_batch_size * max_len)

        dl = probs.copy()
        for i in range(actual_batch_size):
            for j in range(max_len):
                dl[i, j, x0[i, j]] -= 1.0
        dl /= float(actual_batch_size * max_len)
        dl_flat = dl.reshape(actual_batch_size, max_len * V)

        dW2 = h.T @ dl_flat
        db2 = dl_flat.sum(0)
        dh = dl_flat @ W2.T
        dz = dh * (z > 0)
        dW1 = x_in.T @ dz
        db1 = dz.sum(0)

        np.clip(dW1, -1, 1, out=dW1)
        np.clip(dW2, -1, 1, out=dW2)
        adam(W1, dW1, mW1, vW1, step, actual_steps)
        adam(b1, db1, mb1, vb1, step, actual_steps)
        adam(W2, dW2, mW2, vW2, step, actual_steps)
        adam(b2, db2, mb2, vb2, step, actual_steps)

        if verbose and (step % report_every == 0 or step == actual_steps - 1):
            print("  step {0:5d} | loss {1:.4f}".format(step, loss))


def sample(n=20, temp=0.8):
    x = np.full((n, max_len), MASK, dtype=np.int32)
    for t in range(T, 0, -1):
        logits, _ = forward(x, t)
        probs = softmax(logits / temp)
        pred = np.array(
            [[np.random.choice(V, p=probs[i, j]) for j in range(max_len)] for i in range(n)],
            dtype=np.int32,
        )
        tgt = mask_rate(t - 1) if t > 1 else 0
        cur = mask_rate(t)
        masked = (x == MASK)
        if tgt > 0 and cur > 0:
            conf = probs.max(-1)
            conf[~masked] = float("inf")
            for i in range(n):
                masked_positions = np.where(masked[i])[0]
                if len(masked_positions) == 0:
                    continue
                keep = min(int(len(masked_positions) * tgt / max(cur, 1e-8)), len(masked_positions))
                unmask = masked_positions[np.argsort(conf[i][masked_positions])[keep:]]
                x[i, unmask] = pred[i, unmask]
        else:
            x[masked] = pred[masked]
    return [decode(x[i]) for i in range(n)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the minimal NumPy diffusion demo.")
    parser.add_argument("--steps", type=int, default=steps, help="Training iterations.")
    parser.add_argument("--batch-size", type=int, default=B, help="Batch size.")
    parser.add_argument("--samples", type=int, default=20, help="Number of names to sample.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--report-every", type=int, default=500, help="Training log interval.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for NumPy and Python.")
    parser.add_argument("--quiet", action="store_true", help="Suppress training progress logs.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.seed is not None:
        set_seed(args.seed)

    reset_model()
    train(
        num_steps=args.steps,
        batch_size=args.batch_size,
        report_every=args.report_every,
        verbose=not args.quiet,
    )

    generated = sample(n=args.samples, temp=args.temperature)
    if args.quiet:
        print("Generated names: {0}".format(", ".join(generated)))
    else:
        print("\nGenerated names:")
        for name in generated:
            print("  {0}".format(name))


reset_model()


if __name__ == "__main__":
    main()
