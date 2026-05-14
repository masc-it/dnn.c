"""
PyTorch reference for autoregressive generation (argmax + temperature).

Matches dnn.c's decoder_lm_generate:
  1. Create a small decoder LM (same architecture as ref_decoder_lm.py)
  2. Train a few steps on fixed data
  3. Generate with argmax (temp=0) and temperature sampling
  4. Print generated token sequences for use in C tests

Usage:
    uv run --with torch python test/ref_generation.py
    uv run --with torch python test/ref_generation.py --small
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import random

SMALL = '--small' in sys.argv
SEED = 42

if SMALL:
    vocab_size, d_model, n_layers, n_heads, intermediate = \
        8, 4, 2, 2, 8
else:
    vocab_size, d_model, n_layers, n_heads, intermediate = \
        16, 8, 3, 2, 16

d_k = d_model // n_heads
assert d_model == n_heads * d_k

torch.manual_seed(SEED)
random.seed(SEED)

# ── Model (exact match with dnn.c decoder_lm) ──

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

    def forward(self, x, layer_cache=None, position_offset=0):
        """
        If layer_cache is provided (tuple (k_cache, v_cache) each [B, H, max_seq, d_k]),
        use cached K/V.  Only the last token is processed for Q.
        Returns output and updates cache in-place.
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

        if layer_cache is not None:
            k_cache, v_cache = layer_cache  # each [B, H, max_seq, d_k]
            # Append new K/V to cache
            seq_len = position_offset
            k_cache[:, :, seq_len:seq_len + N, :] = K
            v_cache[:, :, seq_len:seq_len + N, :] = V
            # Use full cached K/V for attention
            K_full = k_cache[:, :, :seq_len + N, :]
            V_full = v_cache[:, :, :seq_len + N, :]
        else:
            K_full = K
            V_full = V

        scale = 1.0 / math.sqrt(self.d_k)
        scores = torch.matmul(Q, K_full.transpose(-2, -1)) * scale

        if layer_cache is None:
            # Causal mask during full-sequence forward
            mask = torch.triu(torch.full((N, K_full.size(-2)), float('-inf'),
                                          device=x.device), diagonal=1)
            scores = scores + mask
        # else: single new token, all past visible, no mask needed

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

    def forward(self, input_ids, kvcaches=None):
        """
        input_ids: [B, N] int64 tokens
        kvcaches: optional list of (k_cache, v_cache) tuples, one per block.
                  Each cache is [B, H, max_seq, d_k] pre-allocated zero tensor.
        """
        h = self.embedding(input_ids)
        if kvcaches is not None:
            for i, block in enumerate(self.blocks):
                h = block(h, layer_cache=kvcaches[i], position_offset=self._seq_len[i])
                self._seq_len[i] = self._seq_len[i] + input_ids.shape[1]
        else:
            for block in self.blocks:
                h = block(h)
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits

    def setup_kvcaches(self, B, H, max_seq, d_k, n_layers, device):
        """Create pre-allocated K/V caches for all layers."""
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
    Returns list of generated token IDs (including prompt).
    """
    model.eval()
    B, N = prompt_ids.shape
    assert B == 1
    device = prompt_ids.device
    vocab = model.vocab_size
    max_len = N + max_new_tokens
    output = prompt_ids[0].tolist()  # list of ints

    if use_cache:
        d_k = model.blocks[0].d_k
        H   = model.blocks[0].n_heads
        n_layers = len(model.blocks)
        max_seq = max_len
        kvcaches = model.setup_kvcaches(B, H, max_seq, d_k, n_layers, device)

        with torch.no_grad():
            # Process prompt tokens one-by-one to populate cache
            for pos in range(N):
                token_tensor = torch.tensor([[output[pos]]], device=device)
                logits = model(token_tensor, kvcaches=kvcaches)
                if pos == N - 1:
                    # Sample from last token's logits
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
                token_tensor = torch.tensor([output], device=device)  # [1, cur_len]
                logits = model(token_tensor)  # [1, cur_len, vocab]
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


def train_model(model, opt, input_ids, n_steps=3):
    """Train for a few steps so model is in a known state."""
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

def main():
    B, N = 1, 3  # batch=1, short prompt for generation tests

    # Initialize model
    model = DecoderLM(vocab_size, d_model, n_layers, n_heads, d_k, intermediate)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0.01)

    # Train on a fixed prompt so model has non-random behavior
    train_ids = torch.randint(0, vocab_size, (B, 4))  # [1, 4]

    print(f"/* Autogenerated by test/ref_generation.py", file=sys.stderr)
    print(f"   seed={SEED}, config: vocab={vocab_size}, d_model={d_model},", file=sys.stderr)
    print(f"   n_layers={n_layers}, n_heads={n_heads}, d_k={d_k}, intermediate={intermediate}", file=sys.stderr)
    print(f"   training on fixed input */", file=sys.stderr)

    # Train
    train_model(model, opt, train_ids, n_steps=5)

    # Test prompt
    prompt = torch.tensor([[2, 5, 2]], dtype=torch.long)  # [1, 3]

    # --- Argmax generation (no cache) ---
    gen_no_cache = generate(model, prompt, max_new_tokens=5, temperature=0.0, use_cache=False)
    # --- Argmax generation (with cache) ---
    gen_with_cache = generate(model, prompt, max_new_tokens=5, temperature=0.0, use_cache=True)
    # --- Temperature sampling (seed set) ---
    gen_temp = generate(model, prompt, max_new_tokens=5, temperature=0.7, use_cache=False)
    # --- Temperature sampling with cache ---
    gen_temp_cache = generate(model, prompt, max_new_tokens=5, temperature=0.7, use_cache=True)

    # --- Short prompt (N=1) ---
    prompt_short = torch.tensor([[7]], dtype=torch.long)
    gen_short = generate(model, prompt_short, max_new_tokens=3, temperature=0.0, use_cache=False)
    gen_short_cache = generate(model, prompt_short, max_new_tokens=3, temperature=0.0, use_cache=True)

    # Print results as C literals
    def fmt_ints(name, arr):
        vals = ', '.join(str(v) for v in arr)
        print(f"    int {name}[] = {{{vals}}};")

    def fmt_int(name, val):
        print(f"    int {name} = {val};")

    # Config
    print(f"    int gen_config[] = {{")
    print(f"        {vocab_size}, /* vocab_size */")
    print(f"        {d_model}, /* d_model */")
    print(f"        {n_layers}, /* n_layers */")
    print(f"        {n_heads}, /* n_heads */")
    print(f"        {d_k}, /* d_k */")
    print(f"        {intermediate}, /* intermediate_size */")
    print(f"    }};")
    print()

    # Prompt
    print(f"    int gen_prompt[] = {{2, 5, 2}};")
    print(f"    int gen_prompt_len = 3;")
    print()

    # Argmax generation results
    fmt_ints('gen_no_cache', gen_no_cache)
    fmt_int('gen_no_cache_len', len(gen_no_cache))
    fmt_ints('gen_with_cache', gen_with_cache)
    fmt_int('gen_with_cache_len', len(gen_with_cache))
    print()

    # Short prompt
    fmt_ints('gen_short_nocache', gen_short)
    fmt_int('gen_short_nocache_len', len(gen_short))
    fmt_ints('gen_short_cache', gen_short_cache)
    fmt_int('gen_short_cache_len', len(gen_short_cache))
    print()

    # Verify cached == non-cached for argmax
    assert gen_no_cache == gen_with_cache, \
        f"Cache mismatch: {gen_no_cache} vs {gen_with_cache}"
    assert gen_short == gen_short_cache, \
        f"Short cache mismatch: {gen_short} vs {gen_short_cache}"

    # Print seed info for reproducibility
    print(f"    /* Generated with seed={SEED} */")
    print(f"    /* Argmax: {gen_no_cache} (len={len(gen_no_cache)}) */")
    print(f"    /* Short:  {gen_short} (len={len(gen_short)}) */")
    print(f"    /* Sampled (temp=0.7): {gen_temp} (len={len(gen_temp)}) */")

    print(file=sys.stderr)
    print(f"// Prompt: {prompt.tolist()}", file=sys.stderr)
    print(f"// Argmax (no cache): {gen_no_cache}", file=sys.stderr)
    print(f"// Argmax (cache):    {gen_with_cache}", file=sys.stderr)
    print(f"// Short (no cache):  {gen_short}", file=sys.stderr)
    print(f"// Short (cache):     {gen_short_cache}", file=sys.stderr)
    print(f"// Sampled:           {gen_temp}", file=sys.stderr)
    print(f"// Done.", file=sys.stderr)


if __name__ == '__main__':
    main()
