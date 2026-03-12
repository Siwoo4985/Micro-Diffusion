"""
Micro Diffusion (Pure NumPy) - Discrete Text Diffusion from Scratch
====================================================================

No PyTorch, no TensorFlow -- just NumPy and math.

This implements the full discrete diffusion pipeline:
  1. Forward process: gradually mask (erase) tokens
  2. Denoiser: MLP that predicts original tokens from masked input
  3. Training: teach the MLP to denoise at all noise levels
  4. Sampling: start from all-masked, iteratively unmask by confidence

The diffusion mechanism is identical to the PyTorch version.
"""

import argparse
import math
import os
import random

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
max_len = 12
hidden_dim = 256
T = 40
num_steps = 5000
lr = 5e-4
batch_size = 64

# ---------------------------------------------------------------------------
# Dataset & Tokenizer
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "names.txt"), "r") as f:
    all_names = [line.strip().lower() for line in f if line.strip()]
    all_names = [n for n in all_names if len(n) <= max_len]

chars = sorted(set("".join(all_names)))
PAD_TOKEN = len(chars)
MASK_TOKEN = len(chars) + 1
vocab_size = len(chars) + 2

char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for c, i in char_to_id.items()}
id_to_char[PAD_TOKEN] = "."
id_to_char[MASK_TOKEN] = "_"


def encode(name):
    ids = [char_to_id[c] for c in name[:max_len]]
    ids += [PAD_TOKEN] * (max_len - len(ids))
    return np.array(ids, dtype=np.int32)


def decode(ids):
    return "".join(id_to_char.get(int(i), "?") for i in ids).replace(".", "").replace("_", "")


data = np.stack([encode(n) for n in all_names])

# ---------------------------------------------------------------------------
# Noise Schedule
# ---------------------------------------------------------------------------
def cosine_mask_rate(t, T_max, s=0.008):
    return 1.0 - math.cos(((t / T_max) + s) / (1 + s) * math.pi / 2) ** 2


def add_noise(x_0, t):
    rate = cosine_mask_rate(t, T)
    noise = np.random.rand(*x_0.shape)
    mask = noise < rate
    x_t = x_0.copy()
    x_t[mask] = MASK_TOKEN
    return x_t, mask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def xavier(fan_in, fan_out):
    scale = math.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out).astype(np.float32) * scale


def reset_model():
    global input_dim, W1, b1, W2, b2, W3, b3, adam_m, adam_v

    # -----------------------------------------------------------------------
    # MLP Denoiser (3-layer with skip connection)
    # -----------------------------------------------------------------------
    input_dim = max_len * vocab_size + 1

    W1 = xavier(input_dim, hidden_dim)
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    W2 = xavier(hidden_dim, hidden_dim)
    b2 = np.zeros(hidden_dim, dtype=np.float32)
    W3 = xavier(hidden_dim, max_len * vocab_size)
    b3 = np.zeros(max_len * vocab_size, dtype=np.float32)

    adam_m = {
        "W1": np.zeros_like(W1),
        "b1": np.zeros_like(b1),
        "W2": np.zeros_like(W2),
        "b2": np.zeros_like(b2),
        "W3": np.zeros_like(W3),
        "b3": np.zeros_like(b3),
    }
    adam_v = {
        "W1": np.zeros_like(W1),
        "b1": np.zeros_like(b1),
        "W2": np.zeros_like(W2),
        "b2": np.zeros_like(b2),
        "W3": np.zeros_like(W3),
        "b3": np.zeros_like(b3),
    }


def parameter_count():
    return sum(p.size for p in [W1, b1, W2, b2, W3, b3])


def adam_step(name_p, param, grad, step_num, lr_t, beta1=0.9, beta2=0.999, eps=1e-8):
    adam_m[name_p] = beta1 * adam_m[name_p] + (1 - beta1) * grad
    adam_v[name_p] = beta2 * adam_v[name_p] + (1 - beta2) * grad ** 2
    m_hat = adam_m[name_p] / (1 - beta1 ** (step_num + 1))
    v_hat = adam_v[name_p] / (1 - beta2 ** (step_num + 1))
    param -= lr_t * m_hat / (np.sqrt(v_hat) + eps)


def softmax_2d(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-10)


def forward(x_ids, t):
    batch = x_ids.shape[0]
    x_oh = np.zeros((batch, max_len * vocab_size), dtype=np.float32)
    for i in range(batch):
        for j in range(max_len):
            x_oh[i, j * vocab_size + x_ids[i, j]] = 1.0
    t_feat = np.full((batch, 1), t / float(T), dtype=np.float32)
    x_in = np.concatenate([x_oh, t_feat], axis=1)

    z1 = x_in @ W1 + b1
    h1 = np.maximum(z1, 0)
    z2 = h1 @ W2 + b2
    h2 = np.maximum(z2, 0) + h1
    logits_flat = h2 @ W3 + b3
    logits = logits_flat.reshape(batch, max_len, vocab_size)

    return logits, (x_in, z1, h1, z2, h2)


def train_step(x_0, t, step_num, total_steps):
    x_t, _ = add_noise(x_0, t)
    logits, (x_in, z1, h1, z2, h2) = forward(x_t, t)

    probs = softmax_2d(logits)

    loss = 0.0
    total = x_0.shape[0] * max_len
    for i in range(x_0.shape[0]):
        for j in range(max_len):
            loss -= math.log(max(probs[i, j, x_0[i, j]], 1e-10))
    loss /= float(total)

    dlogits = probs.copy()
    for i in range(x_0.shape[0]):
        for j in range(max_len):
            dlogits[i, j, x_0[i, j]] -= 1.0
    dlogits /= float(total)

    dlogits_flat = dlogits.reshape(x_0.shape[0], max_len * vocab_size)

    clip_val = 1.0

    dW3 = h2.T @ dlogits_flat
    db3 = dlogits_flat.sum(axis=0)

    dh2 = dlogits_flat @ W3.T
    dz2 = dh2 * (z2 > 0).astype(np.float32)
    dW2 = h1.T @ dz2
    db2 = dz2.sum(axis=0)

    dh1 = dz2 @ W2.T + dh2
    dz1 = dh1 * (z1 > 0).astype(np.float32)
    dW1 = x_in.T @ dz1
    db1 = dz1.sum(axis=0)

    for grad in [dW1, db1, dW2, db2, dW3, db3]:
        np.clip(grad, -clip_val, clip_val, out=grad)

    warmup = min(1.0, (step_num + 1) / 200.0)
    decay = max(0.1, 1.0 - step_num / float(max(total_steps, 1)))
    lr_t = lr * warmup * decay

    adam_step("W1", W1, dW1, step_num, lr_t)
    adam_step("b1", b1, db1, step_num, lr_t)
    adam_step("W2", W2, dW2, step_num, lr_t)
    adam_step("b2", b2, db2, step_num, lr_t)
    adam_step("W3", W3, dW3, step_num, lr_t)
    adam_step("b3", b3, db3, step_num, lr_t)

    return loss


def train(total_steps=None, batch_size_override=None, report_every=500, verbose=True):
    actual_steps = num_steps if total_steps is None else total_steps
    actual_batch_size = batch_size if batch_size_override is None else batch_size_override

    if verbose:
        print("Dataset: {0} names, vocab: {1}, max_len: {2}".format(len(all_names), vocab_size, max_len))
        print("Model: {0:,} parameters".format(parameter_count()))
        print("\nTraining for {0} steps...".format(actual_steps))
        print("{0:>6s} | {1:>8s} | {2:>3s} | {3:>6s}".format("step", "loss", "t", "mask%"))
        print("-" * 35)

    for step in range(actual_steps):
        idx = np.random.randint(0, len(data), actual_batch_size)
        x_0 = data[idx]
        t = random.randint(1, T)
        loss = train_step(x_0, t, step, actual_steps)

        if verbose and (step % report_every == 0 or step == actual_steps - 1):
            rate = cosine_mask_rate(t, T)
            print("{0:6d} | {1:8.4f} | {2:3d} | {3:5.1f}%".format(step, loss, t, rate * 100))


def sample(num_samples=10, temperature=0.8, verbose=True):
    x = np.full((num_samples, max_len), MASK_TOKEN, dtype=np.int32)

    for t in range(T, 0, -1):
        logits, _ = forward(x, t)
        probs = softmax_2d(logits / temperature)

        x0_pred = np.zeros((num_samples, max_len), dtype=np.int32)
        for i in range(num_samples):
            for j in range(max_len):
                x0_pred[i, j] = np.random.choice(vocab_size, p=probs[i, j])

        target_rate = cosine_mask_rate(t - 1, T) if t > 1 else 0.0
        current_rate = cosine_mask_rate(t, T)
        is_masked = x == MASK_TOKEN

        if target_rate > 0 and current_rate > 0:
            max_probs = probs.max(axis=-1)
            max_probs[~is_masked] = float("inf")
            for i in range(num_samples):
                masked_pos = np.where(is_masked[i])[0]
                if len(masked_pos) == 0:
                    continue
                conf = max_probs[i][masked_pos]
                sorted_idx = np.argsort(conf)
                n_keep = int(len(masked_pos) * target_rate / max(current_rate, 1e-8))
                n_keep = min(n_keep, len(masked_pos))
                unmask_pos = masked_pos[sorted_idx[n_keep:]]
                x[i, unmask_pos] = x0_pred[i, unmask_pos]
        else:
            x[is_masked] = x0_pred[is_masked]

        if verbose and t in [T, T * 3 // 4, T // 2, T // 4, 1]:
            pct = 100 * (T - t) / float(T)
            previews = []
            for i in range(min(4, num_samples)):
                s = "".join(id_to_char.get(int(x[i][j]), "?") for j in range(max_len))
                previews.append(s.rstrip("."))
            print("  t={0:3d} ({1:5.1f}%): {2}".format(t, pct, " | ".join(previews)))

    return [decode(x[i]) for i in range(num_samples)]


def visualize_forward():
    name = random.choice(all_names)
    x_0 = encode(name).reshape(1, -1)
    print('\nForward Process: "{0}"'.format(name))
    for t_val in [0, T // 8, T // 4, T // 2, 3 * T // 4, T]:
        if t_val == 0:
            display = name
        else:
            x_t, _ = add_noise(x_0, t_val)
            display = "".join(id_to_char.get(int(x_t[0][j]), "?") for j in range(len(name)))
        rate = cosine_mask_rate(t_val, T) if t_val > 0 else 0.0
        print("  t={0:3d} (mask {1:5.1f}%): {2}".format(t_val, rate * 100, display))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the pure NumPy diffusion demo.")
    parser.add_argument("--steps", type=int, default=num_steps, help="Training iterations.")
    parser.add_argument("--batch-size", type=int, default=batch_size, help="Batch size.")
    parser.add_argument("--samples", type=int, default=15, help="Number of names to sample.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Primary sampling temperature.")
    parser.add_argument("--report-every", type=int, default=500, help="Training log interval.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for NumPy and Python.")
    parser.add_argument("--quiet", action="store_true", help="Suppress training progress logs.")
    parser.add_argument("--no-forward-preview", action="store_true", help="Skip the masking visualization.")
    parser.add_argument("--no-temperature-sweep", action="store_true", help="Skip the extra temperature comparison.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.seed is not None:
        set_seed(args.seed)

    reset_model()

    if not args.quiet:
        print("=" * 55)
        print("  Micro Diffusion (Pure NumPy)")
        print("=" * 55)

    if not args.quiet and not args.no_forward_preview:
        visualize_forward()

    train(
        total_steps=args.steps,
        batch_size_override=args.batch_size,
        report_every=args.report_every,
        verbose=not args.quiet,
    )

    generated = sample(num_samples=args.samples, temperature=args.temperature, verbose=not args.quiet)
    if args.quiet:
        print("Generated names: {0}".format(", ".join(generated)))
    else:
        print("\n" + "=" * 55)
        print("  Generating Names")
        print("=" * 55)
        print("  Results: {0}".format(", ".join(generated)))

    if not args.no_temperature_sweep:
        print("\n" + "=" * 55)
        print("  Temperature Comparison")
        print("=" * 55)
        for temp in [0.6, 0.8, 1.0]:
            gen = sample(num_samples=args.samples, temperature=temp, verbose=False)
            print("\n--- Temperature {0} ---".format(temp))
            print("  Results: {0}".format(", ".join(gen)))


reset_model()


if __name__ == "__main__":
    main()
