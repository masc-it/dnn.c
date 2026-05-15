"""
PyTorch reference: generation with prefix + RoPE.
Verifies cached == non-cached equivalence with and without RoPE.

Usage:
    uv run --with torch python test/ref_generation_prefix.py
    uv run --with torch python test/ref_generation_prefix.py --seed 42

Output: C-style literals for use in test_generation_prefix.c
  - Generation with/without cache, argmax
  - Generation with/without cache + RoPE
  - Longer prefix (8 tokens)
  - Short prefix (1 token)
  - Prompt prefix preserved
  - Cached == non-cached verified (asserted)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import random

SEED = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 42

# Config — matches test_generation_prefix.c
vocab_size, d_model, n_layers, n_heads, intermediate = \
    8, 4, 2, 2, 8
d_k = d_model // n_heads
assert d_model == n_heads * d_k

torch.manual_seed(SEED)
random.seed(SEED)

# ── Model (exact match with dnn.c) ──

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

    def _apply_rope(self, x, freqs_cos, freqs_sin, pos_offset=0):
        """Apply RoPE to x [B, H, N, d_k].  freq tables [N_max, d_k//2]."""
        B, H, N, D = x.shape
        assert D % 2 == 0
        half = D // 2
        # Reshape to pairs: [B, H, N, half, 2]
        x_pairs = x.view(B, H, N, half, 2)
        x_even = x_pairs[..., 0]  # [B, H, N, half]
        x_odd  = x_pairs[..., 1]  # [B, H, N, half]
        # Slice freq tables at correct positions
        cos_slice = freqs_cos[pos_offset:pos_offset + N, :].unsqueeze(0).unsqueeze(0)  # [1,1,N,half]
        sin_slice = freqs_sin[pos_offset:pos_offset + N, :].unsqueeze(0).unsqueeze(0)
        # Rotate
        new_even = x_even * cos_slice - x_odd * sin_slice
        new_odd  = x_even * sin_slice + x_odd * cos_slice
        # Interleave back
        out = torch.stack([new_even, new_odd], dim=-1).view(B, H, N, D)
        return out

    def forward(self, x, rope=None, layer_cache=None, position_offset=0, freqs_cos=None, freqs_sin=None):
        """
        rope: tuple (freqs_cos, freqs_sin) or None
        layer_cache: (k_cache, v_cache) for cached generation
        """
        residual = x
        h = self.attn_norm(x)

        Q = self.q_proj(h)
        K = self.k_proj(h)
        V = self.v_proj(h)

        B, N, _ = Q.shape
        Q = Q.view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        K = K.view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # Apply RoPE
        if rope is not None:
            fc, fs = rope
            Q = self._apply_rope(Q, fc, fs, pos_offset=position_offset)
            K = self._apply_rope(K, fc, fs, pos_offset=position_offset)

        if layer_cache is not None:
            k_cache, v_cache = layer_cache
            seq_len = position_offset
            k_cache[:, :, seq_len:seq_len + N, :] = K
            v_cache[:, :, seq_len:seq_len + N, :] = V
            K_full = k_cache[:, :, :seq_len + N, :]
            V_full = v_cache[:, :, :seq_len + N, :]
        else:
            K_full = K
            V_full = V

        scale = 1.0 / math.sqrt(self.d_k)
        scores = torch.matmul(Q, K_full.transpose(-2, -1)) * scale
        if layer_cache is None:
            mask = torch.triu(torch.full((N, K_full.size(-2)), float('-inf'),
                                          device=x.device), diagonal=1)
            scores = scores + mask
        P = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(P, V_full)
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
        self._rope = None  # set by enable_rope
        self._seq_len = None  # per-layer position tracking

    def enable_rope(self, max_seq_len, base=10000.0):
        d_k = self.blocks[0].d_k
        assert d_k % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)  # [max_seq_len, d_k/2]
        self._rope = (freqs.cos(), freqs.sin())

    def forward(self, input_ids, kvcaches=None):
        h = self.embedding(input_ids)
        if kvcaches is not None:
            for i, block in enumerate(self.blocks):
                rope_slice = (self._rope[0], self._rope[1]) if self._rope else None
                h = block(h, rope=rope_slice, layer_cache=kvcaches[i],
                          position_offset=self._seq_len[i])
                self._seq_len[i] += input_ids.shape[1]
        else:
            for block in self.blocks:
                rope_slice = (self._rope[0], self._rope[1]) if self._rope else None
                h = block(h, rope=rope_slice)
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits

    def setup_kvcaches(self, B, H, max_seq, d_k, n_layers, device):
        self._seq_len = [0] * n_layers
        caches = []
        for _ in range(n_layers):
            k_cache = torch.zeros(B, H, max_seq, d_k, device=device)
            v_cache = torch.zeros(B, H, max_seq, d_k, device=device)
            caches.append((k_cache, v_cache))
        return caches


def generate(model, prompt_ids, max_new_tokens, temperature=0.0, use_cache=False):
    """
    Autoregressive generation matching dnn.c decoder_lm_generate.
    """
    model.eval()
    B, N = prompt_ids.shape
    assert B == 1
    device = prompt_ids.device
    vocab = model.vocab_size
    max_len = N + max_new_tokens
    output = prompt_ids[0].tolist()

    if use_cache:
        d_k = model.blocks[0].d_k
        H   = model.blocks[0].n_heads
        n_layers = len(model.blocks)
        max_seq = max_len
        kvcaches = model.setup_kvcaches(B, H, max_seq, d_k, n_layers, device)

        with torch.no_grad():
            for pos in range(N):
                token_tensor = torch.tensor([[output[pos]]], device=device)
                logits = model(token_tensor, kvcaches=kvcaches)
                if pos == N - 1:
                    last_logits = logits[0, 0, :]
                    if temperature == 0.0:
                        next_id = last_logits.argmax().item()
                    else:
                        probs = F.softmax(last_logits / temperature, dim=-1)
                        next_id = torch.multinomial(probs, 1).item()
                    output.append(next_id)
                    if next_id == 258:
                        return output

            while len(output) < max_len:
                token_tensor = torch.tensor([[output[-1]]], device=device)
                logits = model(token_tensor, kvcaches=kvcaches)
                last_logits = logits[0, 0, :]
                if temperature == 0.0:
                    next_id = last_logits.argmax().item()
                else:
                    probs = F.softmax(last_logits / temperature, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                output.append(next_id)
                if next_id == 258:
                    break
    else:
        with torch.no_grad():
            while len(output) < max_len:
                token_tensor = torch.tensor([output], device=device)
                logits = model(token_tensor)
                last_logits = logits[0, -1, :]
                if temperature == 0.0:
                    next_id = last_logits.argmax().item()
                else:
                    probs = F.softmax(last_logits / temperature, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                output.append(next_id)
                if next_id == 258:
                    break
    return output


def train_model(model, opt, input_ids, n_steps=5):
    model.train()
    for step in range(n_steps):
        logits = model(input_ids)
        logits_shifted = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        loss = F.cross_entropy(
            logits_shifted.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction='mean'
        )
        opt.zero_grad()
        loss.backward()
        opt.step()


# ── Main ──

def fmt_ints(name, arr):
    vals = ', '.join(str(v) for v in arr)
    print(f"    int {name}[] = {{{vals}}};")

def fmt_int(name, val):
    print(f"    int {name} = {val};")


def run_case(model, opt, prompt, max_new, label, use_rope=False):
    """Run generation with/without cache, verify equality, print values."""
    if use_rope:
        model.enable_rope(max_seq_len=64, base=10000.0)

    # Argmax generation
    gen_nocache = generate(model, prompt, max_new, temperature=0.0, use_cache=False)
    gen_cache   = generate(model, prompt, max_new, temperature=0.0, use_cache=True)

    assert gen_nocache == gen_cache, \
        f"[{label}] Cached/non-cached mismatch: {gen_nocache} vs {gen_cache}"

    # Prompt prefix preserved
    for i in range(prompt.shape[1]):
        assert gen_nocache[i] == prompt[0, i].item(), \
            f"[{label}] Prompt prefix not preserved at position {i}"

    fmt_ints(f'{label}_result', gen_nocache)
    fmt_int(f'{label}_len', len(gen_nocache))

    return gen_nocache


def main():
    B, N = 1, 3  # default prompt length

    # Print config header
    print(f"/* Autogenerated by test/ref_generation_prefix.py  seed={SEED} */")
    print(f"/* config: vocab={vocab_size}, d_model={d_model}, n_layers={n_layers},")
    print(f"   n_heads={n_heads}, d_k={d_k}, intermediate={intermediate} */")
    print()

    # ─── Case 1: No RoPE, short prompt (N=3) ───
    model = DecoderLM(vocab_size, d_model, n_layers, n_heads, d_k, intermediate)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01,
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    train_ids = torch.randint(0, vocab_size, (B, 4))
    train_model(model, opt, train_ids, n_steps=5)
    prompt = torch.tensor([[2, 5, 2]], dtype=torch.long)
    r1 = run_case(model, opt, prompt, 5, "case1_norope_short")
    print()

    # ─── Case 2: RoPE, short prompt (N=3) ───
    model2 = DecoderLM(vocab_size, d_model, n_layers, n_heads, d_k, intermediate)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=0.01,
                             betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    train_model(model2, opt2, train_ids, n_steps=5)
    prompt2 = torch.tensor([[2, 5, 2]], dtype=torch.long)
    r2 = run_case(model2, opt2, prompt2, 5, "case2_rope_short", use_rope=True)
    print()

    # ─── Case 3: RoPE, longer prefix (N=8) ───
    model3 = DecoderLM(vocab_size, d_model, n_layers, n_heads, d_k, intermediate)
    opt3 = torch.optim.AdamW(model3.parameters(), lr=0.01,
                             betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    train_model(model3, opt3, train_ids, n_steps=5)
    long_prompt = torch.tensor([[0, 3, 7, 1, 4, 2, 6, 5]], dtype=torch.long)
    r3 = run_case(model3, opt3, long_prompt, 5, "case3_rope_long", use_rope=True)
    print()

    # ─── Case 4: RoPE, single-token prompt (N=1) ───
    model4 = DecoderLM(vocab_size, d_model, n_layers, n_heads, d_k, intermediate)
    opt4 = torch.optim.AdamW(model4.parameters(), lr=0.01,
                             betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    train_model(model4, opt4, train_ids, n_steps=5)
    single_token = torch.tensor([[7]], dtype=torch.long)
    r4 = run_case(model4, opt4, single_token, 4, "case4_rope_singletoken", use_rope=True)
    print()

    # ─── Case 5: No RoPE, longer prefix (N=8) ───
    model5 = DecoderLM(vocab_size, d_model, n_layers, n_heads, d_k, intermediate)
    opt5 = torch.optim.AdamW(model5.parameters(), lr=0.01,
                             betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    train_model(model5, opt5, train_ids, n_steps=5)
    r5 = run_case(model5, opt5, long_prompt, 5, "case5_norope_long")
    print()

    # Summary for test runner
    print(f"/* All {5} cases passed.  Seeds: {SEED} */", file=sys.stderr)
    print(f"/* Case 1: {r1}  len={len(r1)} */", file=sys.stderr)
    print(f"/* Case 2: {r2}  len={len(r2)} */", file=sys.stderr)
    print(f"/* Case 3: {r3}  len={len(r3)} */", file=sys.stderr)
    print(f"/* Case 4: {r4}  len={len(r4)} */", file=sys.stderr)
    print(f"/* Case 5: {r5}  len={len(r5)} */", file=sys.stderr)


if __name__ == '__main__':
    main()
