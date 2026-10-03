import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./model_dir"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_LAYERS, N_HEADS, N_KV, HEAD_DIM, MAX_LEN = 22, 32, 4, 64, 2048
GEN_TOKENS = 64
PROMPTS = [
    "The capital of China is",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea,",
    "The three primary colors are",
    "In machine learning, a gradient is",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
weights = {k: v.to(dtype=torch.bfloat16, device=DEVICE) for k, v in load_file(f"{MODEL_PATH}/model.safetensors").items()}

def rms_norm(x, weight, eps=1e-5):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)

def precompute_rope_cache(dim=HEAD_DIM, max_len=MAX_LEN):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
    freqs = torch.outer(torch.arange(max_len, dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]
    return cos.to(torch.bfloat16).to(DEVICE), sin.to(torch.bfloat16).to(DEVICE)

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

def new_cache():
    shape = (N_LAYERS, 1, N_KV, MAX_LEN, HEAD_DIM)
    return (torch.zeros(shape, dtype=torch.bfloat16, device=DEVICE), torch.zeros(shape, dtype=torch.bfloat16, device=DEVICE))

def run_layer(x, w, i, start, cos, sin, cache):
    T = x.shape[1]
    end = start + T
    p = f"model.layers.{i}."
    kc, vc = cache

    h = rms_norm(x, w[p + "input_layernorm.weight"])
    q = (h @ w[p + "self_attn.q_proj.weight"].T).view(1, T, N_HEADS, HEAD_DIM).transpose(1, 2)
    k = (h @ w[p + "self_attn.k_proj.weight"].T).view(1, T, N_KV, HEAD_DIM).transpose(1, 2)
    v = (h @ w[p + "self_attn.v_proj.weight"].T).view(1, T, N_KV, HEAD_DIM).transpose(1, 2)

    q = apply_rope(q, cos[:, :, start:end], sin[:, :, start:end])
    k = apply_rope(k, cos[:, :, start:end], sin[:, :, start:end])

    kc[i, :, :, start:end] = k
    vc[i, :, :, start:end] = v
    k_all = kc[i, :, :, :end].repeat_interleave(N_HEADS // N_KV, dim=1)
    v_all = vc[i, :, :, :end].repeat_interleave(N_HEADS // N_KV, dim=1)

    scores = (q @ k_all.transpose(-2, -1)) / 8.0
    mask = torch.full((T, end), float("-inf"), device=q.device, dtype=q.dtype).triu(start + 1)
    probs = torch.softmax((scores + mask).float(), dim=-1).to(v_all.dtype)
    attn = (probs @ v_all).transpose(1, 2).contiguous().view(1, T, N_HEADS * HEAD_DIM)
    x = x + attn @ w[p + "self_attn.o_proj.weight"].T

    h = rms_norm(x, w[p + "post_attention_layernorm.weight"])
    gate = h @ w[p + "mlp.gate_proj.weight"].T
    up = h @ w[p + "mlp.up_proj.weight"].T
    return x + (F.silu(gate) * up) @ w[p + "mlp.down_proj.weight"].T

def forward(tokens, start, cache, rope, hiddens=None, last_only=False):
    cos, sin = rope
    x = weights["model.embed_tokens.weight"][tokens]
    if hiddens is not None:
        hiddens.append(x)
    for i in range(N_LAYERS):
        x = run_layer(x, weights, i, start, cos, sin, cache)
        if hiddens is not None:
            hiddens.append(x)
    x = rms_norm(x, weights["model.norm.weight"])
    if hiddens is not None:
        hiddens[-1] = x
    if last_only:
        x = x[:, -1:]
    return x @ weights["lm_head.weight"].T

@torch.no_grad()
def generate(ids, n, rope):
    cache = new_cache()
    logits = forward(ids, 0, cache, rope, last_only=True)
    pos, out = ids.shape[1], []
    while True:
        tok = logits[0, -1].argmax().item()
        out.append(tok)
        if tok == tokenizer.eos_token_id or len(out) == n:
            return out
        logits = forward(torch.tensor([[tok]], device=DEVICE), pos, cache, rope, last_only=True)
        pos += 1

def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()

@torch.no_grad()
def main():
    hf = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to(DEVICE).eval()
    rope = precompute_rope_cache()
    layer1_errs, layer22_errs, logit_errs = [], [], []
    tf_hits = tf_total = free_hits = 0

    for n, prompt in enumerate(PROMPTS):
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)
        print(f"\n[{n}] {prompt!r} ({ids.shape[1]} tokens)")

        hiddens = []
        mine = forward(ids, 0, new_cache(), rope, hiddens)
        ref = hf(ids, output_hidden_states=True)
        errs = [rel_err(a, b) for a, b in zip(hiddens, ref.hidden_states)]
        layer1_errs.append(errs[1])
        layer22_errs.append(errs[22])
        logit_errs.append((mine.float() - ref.logits.float()).abs().mean().item())
        if n == 0:
            for i, e in enumerate(errs):
                print(f"  hidden_states[{i:2d}] rel err {e:.2e}")
        print(f"  layer 1 rel err {errs[1]:.2e}, layer 22 rel err {errs[22]:.2e}, mean abs logit err {logit_errs[-1]:.4f}")

        ref_tokens = hf.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=GEN_TOKENS)[0, ids.shape[1] :].tolist()

        full = torch.cat([ids, torch.tensor([ref_tokens], device=DEVICE)], dim=1)
        mine_full = forward(full, 0, new_cache(), rope)
        hits = (mine_full.argmax(-1) == hf(full).logits.argmax(-1)).sum().item()
        tf_hits += hits
        tf_total += full.shape[1]
        print(f"  teacher-forced argmax match {hits}/{full.shape[1]}")

        out = generate(ids, GEN_TOKENS, rope)
        ok = out == ref_tokens
        free_hits += ok
        if ok:
            print(f"  free-running greedy: exact match over {len(out)} tokens")
        else:
            d = next((j for j, (a, b) in enumerate(zip(out, ref_tokens)) if a != b), min(len(out), len(ref_tokens)))
            print(f"  free-running greedy: diverges at token {d}")
            print(f"    mine: {tokenizer.decode(out)!r}")
            print(f"    hf:   {tokenizer.decode(ref_tokens)!r}")

    n = len(PROMPTS)
    checks = [
        (f"layer 1 rel err < 1e-2 (max {max(layer1_errs):.2e})", max(layer1_errs) < 1e-2),
        (f"layer 22 rel err < 5e-2 (max {max(layer22_errs):.2e})", max(layer22_errs) < 5e-2),
        (f"mean abs logit err < 0.05 ({sum(logit_errs) / n:.4f})", sum(logit_errs) / n < 0.05),
        (f"teacher-forced argmax match >= 99% ({tf_hits}/{tf_total})", tf_hits / tf_total >= 0.99),
        (f"free-running greedy exact on >= 90% of prompts ({free_hits}/{n})", free_hits >= 0.9 * n),
    ]
    print("\n" + "-" * 60)
    for label, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {label}")

if __name__ == "__main__":
    main()