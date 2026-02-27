"""
Micro Diffusion - A Minimal Discrete Text Diffusion Model
==========================================================

Like Karpathy's MicroGPT showed the essence of GPT in ~200 lines,
Micro Diffusion shows the essence of text diffusion models.

THE KEY DIFFERENCE:
  GPT (autoregressive):  Generates text LEFT -> RIGHT, one token at a time.
  Diffusion (this code): Generates ALL tokens AT ONCE, refining from noise.

HOW TEXT DIFFUSION WORKS:
  Imagine you have the name "emma" written on a chalkboard.

  Forward Process (adding noise - used during training):
    Step 0:   e m m a      <- clean (original)
    Step 25:  e _ m a      <- some letters erased (masked)
    Step 50:  _ _ m _      <- more erased
    Step 75:  _ _ _ _      <- almost all erased
    Step 100: _ _ _ _      <- fully erased (pure noise)

  Reverse Process (removing noise - used during generation):
    Step 100: _ _ _ _      <- start from blank
    Step 75:  _ m _ _      <- model guesses some letters
    Step 50:  e m _ a      <- more letters revealed
    Step 25:  e m m a      <- almost done
    Step 0:   e m m a      <- clean result!

  The model learns: "Given partially erased text at noise level t,
  predict what the original letters were."

ANOTHER KEY DIFFERENCE:
  GPT uses CAUSAL attention (can only look LEFT, like reading a book)
  Diffusion uses BIDIRECTIONAL attention (can look EVERYWHERE, like a puzzle)
  Because diffusion doesn't generate left-to-right, every position
  can attend to every other position -- just like BERT.

Dependencies: PyTorch
Dataset: ~900 English names (names.txt)
Run: python train.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import random

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Intentionally small for fast CPU training. Increase for better results.

max_len    = 16       # maximum sequence length (names padded to this)
n_embd     = 64       # embedding dimension
n_head     = 4        # number of attention heads
n_layer    = 4        # number of transformer layers
T          = 50       # number of diffusion timesteps
num_steps  = 3000     # training iterations
lr         = 3e-4     # learning rate
batch_size = 64       # names per training step

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Dataset & Tokenizer
# ---------------------------------------------------------------------------
# Character-level tokenizer, just like MicroGPT.
# Vocabulary: a-z (26) + PAD (padding) + MASK (noise token)
#
# PAD fills unused positions in short names ("emma" -> "emma" + 12 PADs)
# MASK is the "erased" token used in diffusion.

script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "names.txt"), "r") as f:
    names = [line.strip().lower() for line in f if line.strip()]

chars = sorted(set("".join(names)))
PAD_TOKEN  = len(chars)
MASK_TOKEN = len(chars) + 1
vocab_size = len(chars) + 2

char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for c, i in char_to_id.items()}
id_to_char[PAD_TOKEN]  = "."
id_to_char[MASK_TOKEN] = "_"

def encode(name):
    """Convert a name string to a fixed-length tensor of token IDs."""
    ids = [char_to_id[c] for c in name[:max_len]]
    ids += [PAD_TOKEN] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

def decode(ids):
    """Convert token IDs back to a string, stripping pad/mask."""
    return "".join(id_to_char[i.item()] for i in ids).replace(".", "").replace("_", "")

data = torch.stack([encode(name) for name in names])
print(f"Dataset: {len(names)} names, vocab size: {vocab_size}, max length: {max_len}")
print(f"Examples: {names[:5]}")

# ---------------------------------------------------------------------------
# Noise Schedule -- The Forward Process
# ---------------------------------------------------------------------------
# Defines HOW noise is added during training.
#
# At timestep t (from 0 to T):
#   mask_rate = t / T  (linear), or a cosine schedule for better results.
#   Each token is independently replaced with MASK with this probability.
#
# We use a COSINE schedule because it:
#   - Adds noise more gradually at the start (preserving structure longer)
#   - Produces better results (from "Improved DDPM" paper)

def cosine_mask_rate(t, T_max, s=0.008):
    """Cosine noise schedule. Returns masking probability at timestep t."""
    return 1.0 - math.cos(((t / T_max) + s) / (1 + s) * math.pi / 2) ** 2

def add_noise(x_0, t):
    """
    Forward process: corrupt clean tokens x_0 at noise level t.
    ALL positions (including PAD) are masked with probability mask_rate(t).

    The model must learn to predict PAD tokens too — this is how it learns
    that names have different lengths. Without this, generation would fill
    all positions with letters.
    """
    rate = cosine_mask_rate(t, T)
    noise = torch.rand_like(x_0.float())
    mask = (noise < rate)
    x_t = x_0.clone()
    x_t[mask] = MASK_TOKEN
    return x_t, mask

# ---------------------------------------------------------------------------
# Transformer Denoiser
# ---------------------------------------------------------------------------
# The neural network that learns to predict original tokens from noisy input.
#
# KEY DIFFERENCE FROM GPT:
#   GPT's attention has a CAUSAL mask (triangle) so position i can only
#   see positions 0..i-1.
#
#   Our model has NO mask -- every position sees every other position
#   (bidirectional). This is because diffusion refines all positions at
#   once and needs full context to make good predictions.

class RMSNorm(nn.Module):
    """Root Mean Square Normalization (simpler than LayerNorm)."""
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-8) * self.scale

class SelfAttention(nn.Module):
    """
    Multi-head self-attention -- BIDIRECTIONAL.
    Every position attends to every other position.
    No causal mask, unlike GPT.
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        # Scaled dot-product attention -- NO causal mask!
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out)

class MLP(nn.Module):
    """Feed-forward network: expand, activate, project back."""
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerBlock(nn.Module):
    """One layer: attention (communicate) + MLP (think) with residuals."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn  = SelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.mlp   = MLP(n_embd)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class DiffusionTransformer(nn.Module):
    """
    The complete denoiser.

    Input:  noisy tokens (some MASK) + timestep t
    Output: logits over vocab for each position

    The timestep embedding tells the model HOW MUCH noise was applied.
    A sequence with 10% masking needs different predictions than 90%.
    """
    def __init__(self):
        super().__init__()
        self.tok_emb  = nn.Embedding(vocab_size, n_embd)
        self.pos_emb  = nn.Embedding(max_len, n_embd)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, n_embd), nn.GELU(), nn.Linear(n_embd, n_embd),
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(n_embd, n_head) for _ in range(n_layer)]
        )
        self.norm_f  = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x_t, t):
        B, L = x_t.shape
        tok = self.tok_emb(x_t)
        pos = self.pos_emb(torch.arange(L, device=x_t.device))
        t_norm = torch.tensor([[t / T]], dtype=torch.float, device=x_t.device)
        t_emb = self.time_mlp(t_norm)
        h = tok + pos + t_emb.unsqueeze(1)
        for block in self.blocks:
            h = block(h)
        h = self.norm_f(h)
        return self.lm_head(h)

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
# For each step:
#   1. Sample a batch of clean names              x_0
#   2. Sample a random noise level                t ~ Uniform(1, T)
#   3. Add noise (mask tokens)                    x_t = add_noise(x_0, t)
#   4. Model predicts original tokens             logits = model(x_t, t)
#   5. Compute cross-entropy loss on non-PAD positions
#   6. Backprop and update
#
# GPT training:       "Given tokens 0..i-1, predict token i"
# Diffusion training: "Given all tokens with some masked, predict all original tokens"

def train():
    model = DiffusionTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")
    print(f"Training for {num_steps} steps on {device}...\n")

    data_d = data.to(device)

    for step in range(num_steps):
        model.train()
        idx = torch.randint(0, len(data_d), (batch_size,))
        x_0 = data_d[idx]
        t = random.randint(1, T)
        x_t, mask = add_noise(x_0, t)
        logits = model(x_t, t)

        # Loss on ALL positions (model must learn PAD predictions too)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size), x_0.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == num_steps - 1:
            print(f"step {step:5d} | loss {loss.item():.4f} | t={t:3d} | "
                  f"mask_rate={cosine_mask_rate(t, T):.2f}")

    return model

# ---------------------------------------------------------------------------
# Sampling -- The Reverse Process (Generation)
# ---------------------------------------------------------------------------
# This is where text is generated from pure noise.
#
# Algorithm:
#   1. Start with ALL MASK tokens: "_ _ _ _ _ _ _ _"
#   2. For t = T, T-1, ..., 1:
#      a. Model predicts what each position should be
#      b. Among masked positions, UNMASK the most confident ones
#      c. The rest stay masked for the next step
#   3. Return the final clean sequence
#
# "Confidence-based unmasking" is like solving a crossword puzzle:
#   - First fill in the words you're most sure about
#   - Those give clues for harder words
#   - Gradually the whole puzzle is filled in

@torch.no_grad()
def sample(model, num_samples=10, temperature=0.8, verbose=True):
    """Generate new names using the reverse diffusion process."""
    model.eval()
    x = torch.full((num_samples, max_len), MASK_TOKEN, dtype=torch.long, device=device)

    if verbose:
        print(f"\nSampling {num_samples} names (temperature={temperature})")
        print("-" * 50)

    for t in range(T, 0, -1):
        logits = model(x, t)
        probs = F.softmax(logits / temperature, dim=-1)

        # Sample predicted tokens for all positions
        flat_probs = probs.view(-1, vocab_size)
        x0_pred = torch.multinomial(flat_probs, 1).view(num_samples, max_len)

        # Target mask rate at t-1
        target_rate = cosine_mask_rate(t - 1, T) if t > 1 else 0.0
        current_rate = cosine_mask_rate(t, T)

        is_masked = (x == MASK_TOKEN)

        if target_rate > 0 and current_rate > 0:
            # Confidence-based unmasking
            max_probs, _ = probs.max(dim=-1)
            max_probs[~is_masked] = float("inf")

            for i in range(num_samples):
                masked_pos = is_masked[i].nonzero(as_tuple=True)[0]
                if len(masked_pos) == 0:
                    continue
                conf = max_probs[i][masked_pos]
                sorted_idx = conf.argsort()  # ascending: least confident first
                # How many to keep masked
                n_keep = int(len(masked_pos) * target_rate / max(current_rate, 1e-8))
                n_keep = min(n_keep, len(masked_pos))
                # Unmask the most confident ones
                unmask_idx = masked_pos[sorted_idx[n_keep:]]
                x[i, unmask_idx] = x0_pred[i, unmask_idx]
        else:
            # Final step: unmask everything
            x[is_masked] = x0_pred[is_masked]

        # Show progress at key moments
        if verbose and t in [T, T*3//4, T//2, T//4, 1]:
            pct = 100 * (T - t) / T
            previews = []
            for i in range(min(3, num_samples)):
                s = "".join(id_to_char[x[i][j].item()] for j in range(max_len))
                previews.append(s.rstrip("."))
            print(f"  t={t:3d} ({pct:5.1f}%): {' | '.join(previews)}")

    # Decode results
    generated = [decode(x[i]) for i in range(num_samples)]
    if verbose:
        print(f"\nGenerated names:")
        for name in generated:
            print(f"  {name}")
    return generated

# ---------------------------------------------------------------------------
# Visualize Forward Process
# ---------------------------------------------------------------------------

def visualize_forward():
    """Show how a clean name gets progressively corrupted."""
    name = random.choice(names)
    x_0 = encode(name).unsqueeze(0).to(device)

    print(f"\nForward Process: \"{name}\"")
    print(f"  (Showing progressive masking)\n")

    for t in [0, T//8, T//4, T//2, 3*T//4, T]:
        if t == 0:
            display = name
        else:
            x_t, _ = add_noise(x_0, t)
            display = "".join(id_to_char[x_t[0][j].item()] for j in range(len(name)))
        rate = cosine_mask_rate(t, T) if t > 0 else 0.0
        print(f"  t={t:3d} (mask {rate*100:5.1f}%): {display}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  Micro Diffusion -- Discrete Text Diffusion Model")
    print("=" * 55)

    visualize_forward()
    model = train()

    print("\n" + "=" * 55)
    print("  Generation")
    print("=" * 55)
    sample(model, num_samples=20, temperature=0.8)

    print("\n" + "=" * 55)
    print("  Temperature Comparison")
    print("=" * 55)
    for temp in [0.5, 0.8, 1.0, 1.5]:
        print(f"\n--- Temperature {temp} ---")
        sample(model, num_samples=5, temperature=temp, verbose=False)
        names_gen = sample(model, num_samples=5, temperature=temp, verbose=False)
        print(f"  {', '.join(names_gen)}")
