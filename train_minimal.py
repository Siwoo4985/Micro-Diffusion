"""
Micro Diffusion — Discrete Text Diffusion in ~150 lines
========================================================
The algorithmic essence of text diffusion, nothing more.

GPT:       left-to-right, one token at a time.
Diffusion: all tokens at once, from noise to text.

  Forward:  e m m a  →  e _ m _  →  _ _ _ _   (mask letters)
  Reverse:  _ _ _ _  →  _ m _ a  →  e m m a   (unmask by confidence)

python train_minimal.py
"""

import numpy as np, math, os, random

# --- Config ---
max_len, hidden, T, steps, lr, B = 12, 256, 40, 5000, 5e-4, 64

# --- Data & Tokenizer ---
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "names.txt")) as f:
    names = [l.strip().lower() for l in f if l.strip() and len(l.strip()) <= max_len]
chars = sorted(set("".join(names)))
PAD, MASK = len(chars), len(chars) + 1
V = len(chars) + 2  # vocab size
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for c, i in c2i.items()} | {PAD: ".", MASK: "_"}

def encode(name):
    return np.array([c2i[c] for c in name] + [PAD] * (max_len - len(name)), dtype=np.int32)
def decode(ids):
    return "".join(i2c[int(i)] for i in ids).replace(".","").replace("_","")

data = np.stack([encode(n) for n in names])
print(f"{len(names)} names, vocab {V}")

# --- Forward Process: progressively mask tokens ---
def mask_rate(t):
    return 1.0 - math.cos(((t/T) + 0.008) / 1.008 * math.pi / 2) ** 2

def add_noise(x, t):
    m = np.random.rand(*x.shape) < mask_rate(t)
    noisy = x.copy(); noisy[m] = MASK
    return noisy

# --- Denoiser: 2-layer MLP (architecture doesn't matter, diffusion does) ---
D = max_len * V + 1  # input: flattened one-hot + timestep
W1 = np.random.randn(D, hidden).astype(np.float32) * math.sqrt(2/D)
b1 = np.zeros(hidden, dtype=np.float32)
W2 = np.random.randn(hidden, max_len * V).astype(np.float32) * math.sqrt(2/hidden)
b2 = np.zeros(max_len * V, dtype=np.float32)
print(f"{W1.size+b1.size+W2.size+b2.size:,} parameters")

def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / (e.sum(-1, keepdims=True) + 1e-10)

def forward(x_ids, t):
    """Predict original tokens from noisy input at timestep t."""
    bs = x_ids.shape[0]
    # One-hot encode + timestep
    oh = np.zeros((bs, max_len * V), dtype=np.float32)
    for i in range(bs):
        for j in range(max_len):
            oh[i, j * V + x_ids[i,j]] = 1.0
    x = np.concatenate([oh, np.full((bs,1), t/T, dtype=np.float32)], 1)
    # Forward: Linear → ReLU → Linear
    z = x @ W1 + b1
    h = np.maximum(z, 0)
    logits = (h @ W2 + b2).reshape(bs, max_len, V)
    return logits, (x, z, h)

# --- Training: teach denoiser to predict clean tokens from noisy input ---
# Adam state
mW1,vW1 = np.zeros_like(W1),np.zeros_like(W1)
mb1,vb1 = np.zeros_like(b1),np.zeros_like(b1)
mW2,vW2 = np.zeros_like(W2),np.zeros_like(W2)
mb2,vb2 = np.zeros_like(b2),np.zeros_like(b2)

def adam(p, g, m, v, s):
    m[:] = 0.9*m + 0.1*g; v[:] = 0.999*v + 0.001*g**2
    mh = m/(1-0.9**(s+1)); vh = v/(1-0.999**(s+1))
    lr_t = lr * min(1, (s+1)/200) * max(0.1, 1-s/steps)
    p -= lr_t * mh/(np.sqrt(vh)+1e-8)

print(f"\nTraining...")
for step in range(steps):
    x0 = data[np.random.randint(0, len(data), B)]
    t = random.randint(1, T)
    xt = add_noise(x0, t)
    logits, (x_in, z, h) = forward(xt, t)
    probs = softmax(logits)

    # Cross-entropy loss + backprop
    loss = -sum(math.log(max(probs[i,j,x0[i,j]], 1e-10))
                for i in range(B) for j in range(max_len)) / (B * max_len)

    # Gradient: d(softmax CE)/d(logits) = probs - one_hot
    dl = probs.copy()
    for i in range(B):
        for j in range(max_len):
            dl[i,j,x0[i,j]] -= 1.0
    dl /= B * max_len
    dl_flat = dl.reshape(B, max_len * V)

    dW2 = h.T @ dl_flat; db2 = dl_flat.sum(0)
    dh = dl_flat @ W2.T; dz = dh * (z > 0)
    dW1 = x_in.T @ dz; db1 = dz.sum(0)

    np.clip(dW1,-1,1,out=dW1); np.clip(dW2,-1,1,out=dW2)
    adam(W1,dW1,mW1,vW1,step); adam(b1,db1,mb1,vb1,step)
    adam(W2,dW2,mW2,vW2,step); adam(b2,db2,mb2,vb2,step)

    if step % 500 == 0 or step == steps-1:
        print(f"  step {step:5d} | loss {loss:.4f}")

# --- Sampling: reverse diffusion, unmask by confidence ---
def sample(n=20, temp=0.8):
    x = np.full((n, max_len), MASK, dtype=np.int32)
    for t in range(T, 0, -1):
        logits, _ = forward(x, t)
        probs = softmax(logits / temp)
        # Sample predictions
        pred = np.array([[np.random.choice(V, p=probs[i,j]) for j in range(max_len)] for i in range(n)])
        # Unmask most confident predictions, keep least confident masked
        tgt = mask_rate(t-1) if t > 1 else 0
        cur = mask_rate(t)
        masked = (x == MASK)
        if tgt > 0 and cur > 0:
            conf = probs.max(-1); conf[~masked] = float("inf")
            for i in range(n):
                mp = np.where(masked[i])[0]
                if len(mp) == 0: continue
                keep = min(int(len(mp) * tgt / max(cur,1e-8)), len(mp))
                unmask = mp[np.argsort(conf[i][mp])[keep:]]
                x[i, unmask] = pred[i, unmask]
        else:
            x[masked] = pred[masked]
    return [decode(x[i]) for i in range(n)]

print(f"\nGenerated names:")
for name in sample():
    print(f"  {name}")
