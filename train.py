"""
Micro Diffusion - A Minimal Discrete Text Diffusion Model
==========================================================

Like Karpathy's MicroGPT showed the essence of GPT in ~200 lines,
Micro Diffusion shows the essence of text diffusion models.

The key difference:
  GPT (autoregressive):  generates text left -> right, one token at a time.
  Diffusion (this code): generates all tokens at once, refining from noise.

How text diffusion works:
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
    Step 0:   e m m a      <- clean result

The model learns: "Given partially erased text at noise level t,
predict what the original letters were."

Dependencies: PyTorch
Dataset: 32K English names (names.txt)
Run: python train.py
"""

import argparse
import math
import os
import random

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    torch = None
    nn = None
    F = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
max_len = 16
n_embd = 64
n_head = 4
n_layer = 4
T = 50
num_steps = 3000
lr = 3e-4
batch_size = 64

device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Dataset & Tokenizer
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "names.txt"), "r") as f:
    names = [line.strip().lower() for line in f if line.strip()]

chars = sorted(set("".join(names)))
PAD_TOKEN = len(chars)
MASK_TOKEN = len(chars) + 1
vocab_size = len(chars) + 2

char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for c, i in char_to_id.items()}
id_to_char[PAD_TOKEN] = "."
id_to_char[MASK_TOKEN] = "_"


def require_torch():
    if torch is None:
        raise SystemExit(
            "PyTorch is required for train.py. Install a torch build compatible "
            "with your Python version, then rerun."
        )


def set_seed(seed):
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


if torch is not None:

    def encode(name):
        """Convert a name string to a fixed-length tensor of token IDs."""
        ids = [char_to_id[c] for c in name[:max_len]]
        ids += [PAD_TOKEN] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


    def decode(ids):
        """Convert token IDs back to a string, stripping pad/mask."""
        return "".join(id_to_char[i.item()] for i in ids).replace(".", "").replace("_", "")


    data = torch.stack([encode(name) for name in names])


    def cosine_mask_rate(t, T_max, s=0.008):
        return 1.0 - math.cos(((t / T_max) + s) / (1 + s) * math.pi / 2) ** 2


    def add_noise(x_0, t):
        rate = cosine_mask_rate(t, T)
        noise = torch.rand_like(x_0.float())
        mask = noise < rate
        x_t = x_0.clone()
        x_t[mask] = MASK_TOKEN
        return x_t, mask


    class RMSNorm(nn.Module):
        def __init__(self, dim):
            super(RMSNorm, self).__init__()
            self.scale = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-8) * self.scale


    class SelfAttention(nn.Module):
        def __init__(self, n_embd, n_head):
            super(SelfAttention, self).__init__()
            self.n_head = n_head
            self.head_dim = n_embd // n_head
            self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
            self.proj = nn.Linear(n_embd, n_embd, bias=False)

        def forward(self, x):
            batch, length, dim = x.shape
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            q = q.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            att = F.softmax(att, dim=-1)
            out = att @ v
            out = out.transpose(1, 2).contiguous().view(batch, length, dim)
            return self.proj(out)


    class MLP(nn.Module):
        def __init__(self, n_embd):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
            self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

        def forward(self, x):
            return self.fc2(F.gelu(self.fc1(x)))


    class TransformerBlock(nn.Module):
        def __init__(self, n_embd, n_head):
            super(TransformerBlock, self).__init__()
            self.norm1 = RMSNorm(n_embd)
            self.attn = SelfAttention(n_embd, n_head)
            self.norm2 = RMSNorm(n_embd)
            self.mlp = MLP(n_embd)

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x


    class DiffusionTransformer(nn.Module):
        def __init__(self):
            super(DiffusionTransformer, self).__init__()
            self.tok_emb = nn.Embedding(vocab_size, n_embd)
            self.pos_emb = nn.Embedding(max_len, n_embd)
            self.time_mlp = nn.Sequential(
                nn.Linear(1, n_embd),
                nn.GELU(),
                nn.Linear(n_embd, n_embd),
            )
            self.blocks = nn.ModuleList(
                [TransformerBlock(n_embd, n_head) for _ in range(n_layer)]
            )
            self.norm_f = RMSNorm(n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        def forward(self, x_t, t):
            _, length = x_t.shape
            tok = self.tok_emb(x_t)
            pos = self.pos_emb(torch.arange(length, device=x_t.device))
            t_norm = torch.tensor([[t / float(T)]], dtype=torch.float, device=x_t.device)
            t_emb = self.time_mlp(t_norm)
            h = tok + pos + t_emb.unsqueeze(1)
            for block in self.blocks:
                h = block(h)
            h = self.norm_f(h)
            return self.lm_head(h)


    def train(total_steps=None, batch_size_override=None, report_every=200, verbose=True):
        actual_steps = num_steps if total_steps is None else total_steps
        actual_batch_size = batch_size if batch_size_override is None else batch_size_override

        model = DiffusionTransformer().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        data_d = data.to(device)

        if verbose:
            n_params = sum(p.numel() for p in model.parameters())
            print("Dataset: {0} names, vocab size: {1}, max length: {2}".format(len(names), vocab_size, max_len))
            print("Model: {0:,} parameters".format(n_params))
            print("Training for {0} steps on {1}...\n".format(actual_steps, device))

        for step in range(actual_steps):
            model.train()
            idx = torch.randint(0, len(data_d), (actual_batch_size,))
            x_0 = data_d[idx]
            t = random.randint(1, T)
            x_t, _ = add_noise(x_0, t)
            logits = model(x_t, t)

            loss = F.cross_entropy(logits.view(-1, vocab_size), x_0.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and (step % report_every == 0 or step == actual_steps - 1):
                print(
                    "step {0:5d} | loss {1:.4f} | t={2:3d} | mask_rate={3:.2f}".format(
                        step, loss.item(), t, cosine_mask_rate(t, T)
                    )
                )

        return model


    @torch.no_grad()
    def sample(model, num_samples=10, temperature=0.8, verbose=True):
        model.eval()
        x = torch.full((num_samples, max_len), MASK_TOKEN, dtype=torch.long, device=device)

        if verbose:
            print("\nSampling {0} names (temperature={1})".format(num_samples, temperature))
            print("-" * 50)

        for t in range(T, 0, -1):
            logits = model(x, t)
            probs = F.softmax(logits / temperature, dim=-1)

            flat_probs = probs.view(-1, vocab_size)
            x0_pred = torch.multinomial(flat_probs, 1).view(num_samples, max_len)

            target_rate = cosine_mask_rate(t - 1, T) if t > 1 else 0.0
            current_rate = cosine_mask_rate(t, T)
            is_masked = x == MASK_TOKEN

            if target_rate > 0 and current_rate > 0:
                max_probs, _ = probs.max(dim=-1)
                max_probs[~is_masked] = float("inf")

                for i in range(num_samples):
                    masked_pos = is_masked[i].nonzero(as_tuple=True)[0]
                    if len(masked_pos) == 0:
                        continue
                    conf = max_probs[i][masked_pos]
                    sorted_idx = conf.argsort()
                    n_keep = int(len(masked_pos) * target_rate / max(current_rate, 1e-8))
                    n_keep = min(n_keep, len(masked_pos))
                    unmask_idx = masked_pos[sorted_idx[n_keep:]]
                    x[i, unmask_idx] = x0_pred[i, unmask_idx]
            else:
                x[is_masked] = x0_pred[is_masked]

            if verbose and t in [T, T * 3 // 4, T // 2, T // 4, 1]:
                pct = 100 * (T - t) / float(T)
                previews = []
                for i in range(min(3, num_samples)):
                    s = "".join(id_to_char[x[i][j].item()] for j in range(max_len))
                    previews.append(s.rstrip("."))
                print("  t={0:3d} ({1:5.1f}%): {2}".format(t, pct, " | ".join(previews)))

        return [decode(x[i]) for i in range(num_samples)]


    def visualize_forward():
        name = random.choice(names)
        x_0 = encode(name).unsqueeze(0).to(device)

        print('\nForward Process: "{0}"'.format(name))
        print("  (Showing progressive masking)\n")

        for t in [0, T // 8, T // 4, T // 2, 3 * T // 4, T]:
            if t == 0:
                display = name
            else:
                x_t, _ = add_noise(x_0, t)
                display = "".join(id_to_char[x_t[0][j].item()] for j in range(len(name)))
            rate = cosine_mask_rate(t, T) if t > 0 else 0.0
            print("  t={0:3d} (mask {1:5.1f}%): {2}".format(t, rate * 100, display))

else:
    data = None

    def encode(name):  # pragma: no cover - exercised only when torch is present
        require_torch()

    def decode(ids):  # pragma: no cover - exercised only when torch is present
        require_torch()

    def cosine_mask_rate(t, T_max, s=0.008):
        return 1.0 - math.cos(((t / T_max) + s) / (1 + s) * math.pi / 2) ** 2

    def add_noise(x_0, t):  # pragma: no cover - exercised only when torch is present
        require_torch()

    def train(total_steps=None, batch_size_override=None, report_every=200, verbose=True):
        require_torch()

    def sample(model, num_samples=10, temperature=0.8, verbose=True):
        require_torch()

    def visualize_forward():
        require_torch()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the PyTorch diffusion demo.")
    parser.add_argument("--steps", type=int, default=num_steps, help="Training iterations.")
    parser.add_argument("--batch-size", type=int, default=batch_size, help="Batch size.")
    parser.add_argument("--samples", type=int, default=20, help="Number of names to sample.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Primary sampling temperature.")
    parser.add_argument("--report-every", type=int, default=200, help="Training log interval.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for Python and PyTorch.")
    parser.add_argument("--quiet", action="store_true", help="Suppress training progress logs.")
    parser.add_argument("--no-forward-preview", action="store_true", help="Skip the masking visualization.")
    parser.add_argument("--no-temperature-sweep", action="store_true", help="Skip the extra temperature comparison.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.seed is not None:
        set_seed(args.seed)

    require_torch()

    if not args.quiet:
        print("=" * 55)
        print("  Micro Diffusion -- Discrete Text Diffusion Model")
        print("=" * 55)

    if not args.quiet and not args.no_forward_preview:
        visualize_forward()

    model = train(
        total_steps=args.steps,
        batch_size_override=args.batch_size,
        report_every=args.report_every,
        verbose=not args.quiet,
    )

    generated = sample(model, num_samples=args.samples, temperature=args.temperature, verbose=not args.quiet)
    if args.quiet:
        print("Generated names: {0}".format(", ".join(generated)))
    else:
        print("\n" + "=" * 55)
        print("  Generation")
        print("=" * 55)
        print("  {0}".format(", ".join(generated)))

    if not args.no_temperature_sweep:
        print("\n" + "=" * 55)
        print("  Temperature Comparison")
        print("=" * 55)
        for temp in [0.5, 0.8, 1.0, 1.5]:
            names_gen = sample(model, num_samples=args.samples, temperature=temp, verbose=False)
            print("\n--- Temperature {0} ---".format(temp))
            print("  {0}".format(", ".join(names_gen)))


if __name__ == "__main__":
    main()
