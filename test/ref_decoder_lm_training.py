"""
PyTorch reference for decoder_lm training step (teacher forcing).

Matches dnn.c's decoder_lm_train_step exactly:
  1. Forward: embed -> N x transformer_block -> norm -> lm_head
  2. Shift logits[:, :-1, :] -> predict input_ids[:, 1:]
  3. Cross-entropy loss over vocab dim
  4. Backward + optimizer step

Usage:
    uv run --with torch python test/ref_decoder_lm_training.py
    uv run --with torch python test/ref_decoder_lm_training.py --small
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

SMALL = '--small' in sys.argv
SEED = 42

if SMALL:
    B, N, vocab_size, d_model, n_layers, n_heads, intermediate = \
        2, 4, 8, 4, 2, 2, 8
else:
    B, N, vocab_size, d_model, n_layers, n_heads, intermediate = \
        2, 8, 16, 8, 3, 2, 16

d_k = d_model // n_heads
assert d_model == n_heads * d_k

torch.manual_seed(SEED)

# ── Model (matches dnn.c decoder_lm exactly) ──

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate, bias=True)
        self.up_proj   = nn.Linear(d_model, intermediate, bias=True)
        self.down_proj = nn.Linear(intermediate, d_model, bias=True)

    def forward(self, x):
        gate = self.gate_proj(x)
        up   = self.up_proj(x)
        hidden = F.silu(gate) * up
        return self.down_proj(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_k, intermediate):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k

        self.attn_norm = nn.LayerNorm(d_model, elementwise_affine=True)
        self.ffn_norm  = nn.LayerNorm(d_model, elementwise_affine=True)

        self.q_proj   = nn.Linear(d_model, n_heads * d_k, bias=True)
        self.k_proj   = nn.Linear(d_model, n_heads * d_k, bias=True)
        self.v_proj   = nn.Linear(d_model, n_heads * d_k, bias=True)
        self.out_proj = nn.Linear(n_heads * d_k, d_model, bias=True)

        self.ffn = SwiGLUFFN(d_model, intermediate)

    def forward(self, x):
        residual = x
        h = self.attn_norm(x)

        Q = self.q_proj(h)
        K = self.k_proj(h)
        V = self.v_proj(h)

        B, N, _ = Q.shape
        Q = Q.view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        mask = torch.triu(torch.full((N, N), float('-inf'), device=x.device), diagonal=1)
        scores = scores + mask

        P = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(P, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, -1)
        attn_out = self.out_proj(attn_out)
        x = residual + attn_out

        residual = x
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = residual + h
        return x


class DecoderLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_k, intermediate):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_k, intermediate)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model, elementwise_affine=True)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, input_ids):
        h = self.embedding(input_ids)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits


def train_step(model, opt, input_ids):
    """Single training step matching dnn.c decoder_lm_train_step."""
    logits = model(input_ids)                       # [B, N, vocab]

    # Shift: logits[:, :-1, :] -> predict input_ids[:, 1:]
    logits_shifted = logits[:, :-1, :]              # [B, N-1, vocab]
    targets = input_ids[:, 1:]                      # [B, N-1]

    # Cross-entropy over vocab dim (logits: [B, N-1, vocab], targets: [B, N-1])
    loss = F.cross_entropy(
        logits_shifted.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction='mean'
    )

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss


# ── Run ──
model = DecoderLM(vocab_size, d_model, n_layers, n_heads, d_k, intermediate)
model.train()

opt = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                        eps=1e-8, weight_decay=0.01)

# Fixed input sequence
input_ids = torch.randint(0, vocab_size, (B, N))

# Capture initial state
init_emb_weight = model.embedding.weight.detach().clone()

n_steps = 3
losses = []
for step in range(n_steps):
    loss = train_step(model, opt, input_ids)
    losses.append(loss.item())

# Capture post-training state
final_emb_weight = model.embedding.weight.detach()
emb_weight_change = (final_emb_weight - init_emb_weight).abs().max().item()

def count_params(m):
    return sum(p.numel() for p in m.parameters())

print(f"/* Autogenerated by test/ref_decoder_lm_training.py", file=sys.stderr)
print(f"   seed={SEED}, config: B={B}, N={N}, vocab={vocab_size}, d_model={d_model},", file=sys.stderr)
print(f"   n_layers={n_layers}, n_heads={n_heads}, d_k={d_k}, intermediate={intermediate}", file=sys.stderr)
print(f"   n_steps={n_steps}, total_params={count_params(model)} */", file=sys.stderr)

# Print everything as C literals
def fmt_ints(name, t):
    flat = t.detach().cpu().numpy().flatten().astype(int)
    vals = ', '.join(str(v) for v in flat)
    return f'    int {name}[] = {{{vals}}};\n'

def fmt_floats(name, t):
    flat = t.detach().cpu().numpy().flatten()
    vals = ', '.join(f'{v:.10f}f' for v in flat)
    return f'    float {name}[] = {{{vals}}};\n'

print(f"/* Config: B={B}, N={N}, vocab={vocab_size}, d_model={d_model},")
print(f"   n_layers={n_layers}, n_heads={n_heads}, d_k={d_k}, intermediate={intermediate} */")
print()

# Input IDs
print(fmt_ints('input_ids', input_ids))

# Loss values
print(f"    float step_losses[] = {{{', '.join(f'{v:.10f}f' for v in losses)}}};")
print()

# Initial param weights for reference (block 0)
g = model.blocks[0].q_proj.weight.grad
if g is not None:
    print(fmt_floats('step3_grad_block0_q_proj_weight', g))

g = model.embedding.weight.grad
if g is not None:
    print(fmt_floats('step3_grad_embedding_weight', g))

# Parameter change magnitude
print(f"    float embedding_weight_change = {emb_weight_change:.10f}f;")
print(f"    float final_loss = {losses[-1]:.10f}f;")

print(file=sys.stderr)
print(f"// Losses: {losses}", file=sys.stderr)
print(f"// Embedding weight max change: {emb_weight_change:.6e}", file=sys.stderr)
print(f"// Done.", file=sys.stderr)
