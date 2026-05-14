"""
PyTorch reference for gradient clipping (L2 norm + value clipping).

Usage:
    uv run --with torch python test/ref_grad_clip.py

Generates reference values used by test/test_grad_clip.c.
"""
import torch
import math

SEED = 42
torch.manual_seed(SEED)

# ── Configs ──
N_PARAMS = 3
PARAM_SIZES = [100, 50, 25]

# ── Test 1: clip_grad_norm_ basic ──
print("=== clip_grad_norm_ basic ===")
torch.manual_seed(SEED)
params = [torch.randn(s, requires_grad=True) for s in PARAM_SIZES]
loss = sum((p * 2.0).sum() for p in params)
loss.backward()

# Compute total norm before clipping
total_norm_before = 0.0
for p in params:
    if p.grad is not None:
        total_norm_before += p.grad.norm(2).item() ** 2
total_norm_before = math.sqrt(total_norm_before)

# Clip
clip_norm_value = 1.0
total_norm_after = torch.nn.utils.clip_grad_norm_(params, clip_norm_value)

print(f"  total_norm_before: {total_norm_before:.15e}")
print(f"  total_norm_after:  {total_norm_after:.15e}")
print(f"  max_norm:          {clip_norm_value}")

# Verify clipping happened
total_norm_check = 0.0
for p in params:
    if p.grad is not None:
        total_norm_check += p.grad.norm(2).item() ** 2
total_norm_check = math.sqrt(total_norm_check)
print(f"  total_norm_after_check: {total_norm_check:.15e}")

is_clipped = total_norm_check <= clip_norm_value * 1.0001
print(f"  clipped: {is_clipped}")

# Print norm for C test
print(f"\n  /* ref_clip_norm_before = {total_norm_before:.10e}f */")
print(f"  /* ref_clip_norm_after  = {total_norm_before:.10e}f (return value is BEFORE) */")
print(f"  /* ref_clip_norm_expected_norm = {total_norm_check:.10e}f */")

# Print first few grad values for C check
print(f"\n  /* First 5 grad values after clip (first param): */")
g = params[0].grad.flatten()
for i in range(min(5, len(g))):
    print(f"  /*   grad[{i}] = {g[i]:.10e}f */")

# ── Test 2: no clipping when norm < max_norm ──
print(f"\n=== clip_grad_norm_ no-op ===")
torch.manual_seed(SEED)
params2 = [torch.randn(s, requires_grad=True) for s in PARAM_SIZES]
loss2 = sum((p * 0.001).sum() for p in params2)  # small grads
loss2.backward()

# Compute norm before
norm_before = 0.0
for p in params2:
    if p.grad is not None:
        norm_before += p.grad.norm(2).item() ** 2
norm_before = math.sqrt(norm_before)

# Clip with large max_norm (won't trigger)
large_max = 1000.0
norm_returned = torch.nn.utils.clip_grad_norm_(params2, large_max)

norm_after = 0.0
for p in params2:
    if p.grad is not None:
        norm_after += p.grad.norm(2).item() ** 2
norm_after = math.sqrt(norm_after)

print(f"  norm_before:  {norm_before:.10e}")
print(f"  norm_returned: {norm_returned:.10e}")
print(f"  norm_after:   {norm_after:.10e}")
print(f"  unchanged: {abs(norm_before - norm_after) < 1e-6}")

# ── Test 3: clip_grad_value_ ──
print(f"\n=== clip_grad_value_ ===")
torch.manual_seed(SEED)
params3 = [torch.randn(s, requires_grad=True) for s in PARAM_SIZES]
loss3 = sum((p * 2.0).sum() for p in params3)
loss3.backward()

# Get pre-clip stats
max_abs_before = max(p.grad.abs().max().item() for p in params3)
print(f"  max_abs_grad_before: {max_abs_before:.10e}")

# Value clip
clip_val = 0.1
for p in params3:
    p.grad.data.clamp_(-clip_val, clip_val)

max_abs_after = max(p.grad.abs().max().item() for p in params3)
print(f"  max_abs_grad_after:  {max_abs_after:.10e}")
print(f"  clipped: {max_abs_after <= clip_val * 1.0001}")

# Print a few values
g0 = params3[0].grad.flatten()
print(f"\n  /* First 5 grad values after value clip (first param): */")
for i in range(min(5, len(g0))):
    print(f"  /*   grad[{i}] = {g0[i]:.10e}f */")

# ── Test 4: gradient clipping in training step ──
print(f"\n=== clip in training (decoder_lm style) ===")
torch.manual_seed(SEED)
B, N, vocab, d_model, n_layers, n_heads, d_k, intermediate = 2, 4, 8, 4, 2, 2, 2, 8

class SwiGLUFFN(torch.nn.Module):
    def __init__(self, d_model, intermediate):
        super().__init__()
        self.gate_proj = torch.nn.Linear(d_model, intermediate, bias=True)
        self.up_proj = torch.nn.Linear(d_model, intermediate, bias=True)
        self.down_proj = torch.nn.Linear(intermediate, d_model, bias=True)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_k, intermediate):
        super().__init__()
        self.q_proj = torch.nn.Linear(d_model, n_heads * d_k, bias=True)
        self.k_proj = torch.nn.Linear(d_model, n_heads * d_k, bias=True)
        self.v_proj = torch.nn.Linear(d_model, n_heads * d_k, bias=True)
        self.out_proj = torch.nn.Linear(n_heads * d_k, d_model, bias=True)
        self.attn_norm = torch.nn.LayerNorm(d_model, eps=1e-5)
        self.ffn_norm = torch.nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = SwiGLUFFN(d_model, intermediate)
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x):
        residual = x
        h = self.attn_norm(x)
        B, N, _ = h.shape
        Q = self.q_proj(h).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(h).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(h).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        # Causal attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        causal_mask = torch.triu(torch.full((N, N), float('-inf')), diagonal=1).to(x.device)
        scores = scores + causal_mask
        attn = torch.nn.functional.softmax(scores, dim=-1) @ V
        attn = attn.transpose(1, 2).contiguous().view(B, N, self.n_heads * self.d_k)
        attn = self.out_proj(attn)
        x = residual + attn
        residual = x
        h = self.ffn_norm(x)
        x = residual + self.ffn(h)
        return x

class DecoderLM(torch.nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_k, intermediate):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_k, intermediate)
            for _ in range(n_layers)
        ])
        self.norm = torch.nn.LayerNorm(d_model, eps=1e-5)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, input_ids):
        h = self.embedding(input_ids)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits

torch.manual_seed(SEED)
model = DecoderLM(vocab, d_model, n_layers, n_heads, d_k, intermediate)
opt = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

input_ids = torch.tensor([[3, 6, 7, 0], [4, 3, 4, 7]])

# Gather all params for clip
all_params = list(model.parameters())

# Training steps with gradient clipping
losses_with_clip = []
norms_before = []
norms_after = []
clip_max_norm = 1.0

for step in range(3):
    opt.zero_grad()
    logits = model(input_ids)
    logits_shifted = logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(logits_shifted.transpose(1, 2), targets)
    loss.backward()

    # Measure norm before clipping
    norm_b = 0.0
    for p in all_params:
        if p.grad is not None:
            norm_b += p.grad.norm(2).item() ** 2
    norm_b = math.sqrt(norm_b)
    norms_before.append(norm_b)

    # Clip
    torch.nn.utils.clip_grad_norm_(all_params, clip_max_norm)

    # Measure norm after clipping
    norm_a = 0.0
    for p in all_params:
        if p.grad is not None:
            norm_a += p.grad.norm(2).item() ** 2
    norm_a = math.sqrt(norm_a)
    norms_after.append(norm_a)

    opt.step()
    losses_with_clip.append(loss.item())

print(f"  clip_max_norm = {clip_max_norm}")
for s in range(3):
    print(f"  step {s}: loss={losses_with_clip[s]:.10e}  norm_before={norms_before[s]:.10e}  norm_after={norms_after[s]:.10e}")

# Reference: training WITHOUT clipping for comparison
print(f"\n=== training WITHOUT clip (for comparison) ===")
torch.manual_seed(SEED)
model2 = DecoderLM(vocab, d_model, n_layers, n_heads, d_k, intermediate)
opt2 = torch.optim.AdamW(model2.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

losses_no_clip = []
for step in range(3):
    opt2.zero_grad()
    logits = model2(input_ids)
    logits_shifted = logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(logits_shifted.transpose(1, 2), targets)
    loss.backward()
    opt2.step()
    losses_no_clip.append(loss.item())

for s in range(3):
    print(f"  step {s}: loss={losses_no_clip[s]:.10e}")

# ── Test 5: extreme gradient norm clipping ──
print(f"\n=== extreme clip (max_norm=0.01) ===")
torch.manual_seed(SEED)
params5 = [torch.randn(20, requires_grad=True)]
loss5 = (params5[0] * 100.0).sum()  # make large grads
loss5.backward()

norm5_before = params5[0].grad.norm(2).item()
total_norm5 = torch.nn.utils.clip_grad_norm_(params5, 0.01)
norm5_after = params5[0].grad.norm(2).item()
print(f"  norm_before: {norm5_before:.10e}")
print(f"  norm_after:  {norm5_after:.10e}")
print(f"  clipped_norm <= 0.01: {norm5_after <= 0.01001}")
print(f"  return value matches before: {abs(total_norm5 - norm5_before) < 1e-6}")

# ── Test 6: max_norm=0 (should be no-op) ──
print(f"\n=== max_norm=0 (no-op) ===")
torch.manual_seed(SEED)
params6 = [torch.randn(20, requires_grad=True)]
loss6 = (params6[0] * 2.0).sum()
loss6.backward()
g_copy = params6[0].grad.clone()
torch.nn.utils.clip_grad_norm_(params6, 0.0)
diff = (params6[0].grad - g_copy).abs().max().item()
print(f"  max_grad_diff_after_no_op_clip: {diff:.10e} (should be 0.0)")

print(f"\n/* Reference values for C test: */")
print(f"/* clip_max_norm        = {clip_max_norm} */")
print(f"/* loss_with_clip[0]    = {losses_with_clip[0]:.10e}f */")
print(f"/* loss_with_clip[1]    = {losses_with_clip[1]:.10e}f */")
print(f"/* loss_with_clip[2]    = {losses_with_clip[2]:.10e}f */")
print(f"/* norm_before_clip[0]  = {norms_before[0]:.10e}f */")
print(f"/* norm_after_clip[0]   = {norms_after[0]:.10e}f */")
