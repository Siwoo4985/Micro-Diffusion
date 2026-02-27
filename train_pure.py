"""
Micro Diffusion (Pure NumPy) - Discrete Text Diffusion from Scratch
====================================================================

No PyTorch, no TensorFlow -- just NumPy and math.

This implements the FULL discrete diffusion pipeline:
  1. Forward process: gradually mask (erase) tokens
  2. Denoiser: MLP that predicts original tokens from masked input
  3. Training: teach the MLP to denoise at all noise levels
  4. Sampling: start from all-masked, iteratively unmask by confidence

The diffusion mechanism is IDENTICAL to the PyTorch version.
"""

import numpy as np
import math
import os
import random

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
max_len    = 12       # max name length (shorter = easier to learn)
hidden_dim = 256      # MLP width
T          = 40       # diffusion timesteps
num_steps  = 5000     # training iterations
lr         = 5e-4
batch_size = 64

# ---------------------------------------------------------------------------
# Dataset & Tokenizer
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "names.txt"), "r") as f:
    all_names = [line.strip().lower() for line in f if line.strip()]
    all_names = [n for n in all_names if len(n) <= max_len]  # filter long names

chars = sorted(set("".join(all_names)))
PAD_TOKEN  = len(chars)
MASK_TOKEN = len(chars) + 1
vocab_size = len(chars) + 2

char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for c, i in char_to_id.items()}
id_to_char[PAD_TOKEN]  = "."
id_to_char[MASK_TOKEN] = "_"

def encode(name):
    ids = [char_to_id[c] for c in name[:max_len]]
    ids += [PAD_TOKEN] * (max_len - len(ids))
    return np.array(ids, dtype=np.int32)

def decode(ids):
    return "".join(id_to_char.get(int(i), "?") for i in ids).replace(".", "").replace("_", "")

data = np.stack([encode(n) for n in all_names])
print(f"Dataset: {len(all_names)} names, vocab: {vocab_size}, max_len: {max_len}")

# ---------------------------------------------------------------------------
# Noise Schedule
# ---------------------------------------------------------------------------
def cosine_mask_rate(t, T_max, s=0.008):
    return 1.0 - math.cos(((t / T_max) + s) / (1 + s) * math.pi / 2) ** 2

def add_noise(x_0, t):
    rate = cosine_mask_rate(t, T)
    noise = np.random.rand(*x_0.shape)
    # Mask ALL positions (including PAD) -- model must learn to predict PAD too
    mask = (noise < rate)
    x_t = x_0.copy()
    x_t[mask] = MASK_TOKEN
    return x_t, mask

# ---------------------------------------------------------------------------
# MLP Denoiser (3-layer with skip connection)
# ---------------------------------------------------------------------------
# The DIFFUSION part (masking/unmasking/sampling) is architecture-agnostic.
# You could replace this MLP with a Transformer, CNN, or anything else.
# The key educational content is the diffusion loop, not the denoiser.

input_dim = max_len * vocab_size + 1  # flattened one-hot + timestep

def xavier(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out).astype(np.float32) * math.sqrt(2.0 / (fan_in + fan_out))

# Parameters
W1 = xavier(input_dim, hidden_dim)
b1 = np.zeros(hidden_dim, dtype=np.float32)
W2 = xavier(hidden_dim, hidden_dim)
b2 = np.zeros(hidden_dim, dtype=np.float32)
W3 = xavier(hidden_dim, max_len * vocab_size)
b3 = np.zeros(max_len * vocab_size, dtype=np.float32)

n_params = sum(p.size for p in [W1,b1,W2,b2,W3,b3])
print(f"Model: {n_params:,} parameters")

# Adam state
adam_m = {k: np.zeros_like(v) for k, v in [("W1",W1),("b1",b1),("W2",W2),("b2",b2),("W3",W3),("b3",b3)]}
adam_v = {k: np.zeros_like(v) for k, v in [("W1",W1),("b1",b1),("W2",W2),("b2",b2),("W3",W3),("b3",b3)]}

def adam_step(name_p, param, grad, step_num, lr_t, beta1=0.9, beta2=0.999, eps=1e-8):
    adam_m[name_p] = beta1 * adam_m[name_p] + (1 - beta1) * grad
    adam_v[name_p] = beta2 * adam_v[name_p] + (1 - beta2) * grad ** 2
    m_hat = adam_m[name_p] / (1 - beta1 ** (step_num + 1))
    v_hat = adam_v[name_p] / (1 - beta2 ** (step_num + 1))
    param -= lr_t * m_hat / (np.sqrt(v_hat) + eps)

def softmax_2d(x):
    """Softmax over last axis, batch-safe."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-10)

def forward(x_ids, t):
    B = x_ids.shape[0]
    # One-hot encode: (B, max_len) -> (B, max_len * vocab_size)
    x_oh = np.zeros((B, max_len * vocab_size), dtype=np.float32)
    for i in range(B):
        for j in range(max_len):
            x_oh[i, j * vocab_size + x_ids[i, j]] = 1.0
    # Append timestep
    t_feat = np.full((B, 1), t / T, dtype=np.float32)
    x_in = np.concatenate([x_oh, t_feat], axis=1)

    # Layer 1
    z1 = x_in @ W1 + b1
    h1 = np.maximum(z1, 0)
    # Layer 2 with skip connection from input
    z2 = h1 @ W2 + b2
    h2 = np.maximum(z2, 0) + h1  # skip connection
    # Output
    logits_flat = h2 @ W3 + b3
    logits = logits_flat.reshape(B, max_len, vocab_size)

    return logits, (x_in, z1, h1, z2, h2)

def train_step(x_0, t, step_num):
    global W1, b1, W2, b2, W3, b3
    B = x_0.shape[0]

    x_t, mask = add_noise(x_0, t)
    logits, (x_in, z1, h1, z2, h2) = forward(x_t, t)

    # Softmax + cross-entropy loss on ALL positions (model must learn PAD too)
    probs = softmax_2d(logits)  # (B, max_len, vocab_size)

    loss = 0.0
    total = B * max_len
    for i in range(B):
        for j in range(max_len):
            loss -= math.log(max(probs[i, j, x_0[i, j]], 1e-10))
    loss /= total

    # Backward: dL/d(logits) = (probs - one_hot) / total
    dlogits = probs.copy()
    for i in range(B):
        for j in range(max_len):
            dlogits[i, j, x_0[i, j]] -= 1.0
    dlogits /= total

    dlogits_flat = dlogits.reshape(B, max_len * vocab_size)

    # Gradient clipping (by value)
    clip_val = 1.0

    # Layer 3 gradients
    dW3 = h2.T @ dlogits_flat
    db3 = dlogits_flat.sum(axis=0)

    # Layer 2 backprop (with skip)
    dh2 = dlogits_flat @ W3.T
    dh2_pre_skip = dh2.copy()
    dh1_skip = dh2.copy()  # from skip connection
    dz2 = dh2_pre_skip * (z2 > 0).astype(np.float32)
    dW2 = h1.T @ dz2
    db2 = dz2.sum(axis=0)

    # Layer 1 backprop
    dh1 = dz2 @ W2.T + dh1_skip
    dz1 = dh1 * (z1 > 0).astype(np.float32)
    dW1 = x_in.T @ dz1
    db1 = dz1.sum(axis=0)

    # Clip gradients
    for g in [dW1, db1, dW2, db2, dW3, db3]:
        np.clip(g, -clip_val, clip_val, out=g)

    # Learning rate schedule
    warmup = min(1.0, (step_num + 1) / 200)
    decay = max(0.1, 1.0 - step_num / num_steps)
    lr_t = lr * warmup * decay

    adam_step("W1", W1, dW1, step_num, lr_t)
    adam_step("b1", b1, db1, step_num, lr_t)
    adam_step("W2", W2, dW2, step_num, lr_t)
    adam_step("b2", b2, db2, step_num, lr_t)
    adam_step("W3", W3, dW3, step_num, lr_t)
    adam_step("b3", b3, db3, step_num, lr_t)

    return loss

def sample(num_samples=10, temperature=0.8, verbose=True):
    x = np.full((num_samples, max_len), MASK_TOKEN, dtype=np.int32)

    for t in range(T, 0, -1):
        logits, _ = forward(x, t)
        probs = softmax_2d(logits / temperature)

        # Sample predictions
        x0_pred = np.zeros((num_samples, max_len), dtype=np.int32)
        for i in range(num_samples):
            for j in range(max_len):
                x0_pred[i, j] = np.random.choice(vocab_size, p=probs[i, j])

        target_rate = cosine_mask_rate(t - 1, T) if t > 1 else 0.0
        current_rate = cosine_mask_rate(t, T)
        is_masked = (x == MASK_TOKEN)

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

        if verbose and t in [T, T*3//4, T//2, T//4, 1]:
            pct = 100 * (T - t) / T
            previews = []
            for i in range(min(4, num_samples)):
                s = "".join(id_to_char.get(int(x[i][j]), "?") for j in range(max_len))
                previews.append(s.rstrip("."))
            print(f"  t={t:3d} ({pct:5.1f}%): {' | '.join(previews)}")

    return [decode(x[i]) for i in range(num_samples)]

# ---------------------------------------------------------------------------
# Visualize Forward Process
# ---------------------------------------------------------------------------
def visualize_forward():
    name = random.choice(all_names)
    x_0 = encode(name).reshape(1, -1)
    print(f"\nForward Process: \"{name}\"")
    for t_val in [0, T//8, T//4, T//2, 3*T//4, T]:
        if t_val == 0:
            display = name
        else:
            x_t, _ = add_noise(x_0, t_val)
            display = "".join(id_to_char.get(int(x_t[0][j]), "?") for j in range(len(name)))
        rate = cosine_mask_rate(t_val, T) if t_val > 0 else 0.0
        print(f"  t={t_val:3d} (mask {rate*100:5.1f}%): {display}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  Micro Diffusion (Pure NumPy)")
    print("=" * 55)

    visualize_forward()

    print(f"\nTraining for {num_steps} steps...")
    print(f"{'step':>6s} | {'loss':>8s} | {'t':>3s} | {'mask%':>6s}")
    print("-" * 35)

    for step in range(num_steps):
        idx = np.random.randint(0, len(data), batch_size)
        x_0 = data[idx]
        t = random.randint(1, T)
        loss = train_step(x_0, t, step)

        if step % 500 == 0 or step == num_steps - 1:
            rate = cosine_mask_rate(t, T)
            print(f"{step:6d} | {loss:8.4f} | {t:3d} | {rate*100:5.1f}%")

    print("\n" + "=" * 55)
    print("  Generating Names")
    print("=" * 55)

    for temp in [0.6, 0.8, 1.0]:
        print(f"\n--- Temperature {temp} ---")
        gen = sample(num_samples=15, temperature=temp, verbose=(temp == 0.8))
        print(f"  Results: {', '.join(gen)}")
