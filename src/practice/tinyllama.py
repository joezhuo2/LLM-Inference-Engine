import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from safetensors.torch import load_file

MODEL_PATH = "./model_dir"
PROMPT = "The capital of China is"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {DEVICE}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
weights = load_file(f"{MODEL_PATH}/model.safetensors")

for k, v in weights.items():
    weights[k] = v.to(dtype=torch.bfloat16, device=DEVICE)

def rms_norm(x, weight, eps=1e-5):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)

def precompute_rope_cache(dim: int = 64, max_seq_len: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=torch.bfloat16, device=DEVICE), emb.sin().to(dtype=torch.bfloat16, device=DEVICE)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)

def run_single_layer_test(x: torch.Tensor, weights: dict, layer_idx: int, seq_len: int, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    norm_x = rms_norm(x, weights[f"model.layers.{layer_idx}.input_layernorm.weight"])

    q = ((norm_x @ weights[f"model.layers.{layer_idx}.self_attn.q_proj.weight"].T).view(1, seq_len, 32, 64).transpose(1, 2))
    k = ((norm_x @ weights[f"model.layers.{layer_idx}.self_attn.k_proj.weight"].T).view(1, seq_len, 4, 64).transpose(1, 2))
    v = ((norm_x @ weights[f"model.layers.{layer_idx}.self_attn.v_proj.weight"].T).view(1, seq_len, 4, 64).transpose(1, 2))

    q = apply_rope(q, cos[:seq_len].unsqueeze(0).unsqueeze(0), sin[:seq_len].unsqueeze(0).unsqueeze(0))
    k = apply_rope(k, cos[:seq_len].unsqueeze(0).unsqueeze(0), sin[:seq_len].unsqueeze(0).unsqueeze(0))

    k_rep = k.repeat_interleave(8, dim=1)
    v_rep = v.repeat_interleave(8, dim=1)

    scores = (q @ k_rep.transpose(-2, -1)) / 8.0

    mask = torch.full((seq_len, seq_len), float("-inf"), device=q.device, dtype=q.dtype).triu(1)

    attn_weights = torch.softmax((scores + mask).float(), dim=-1).to(v_rep.dtype)
    attn_out = ((attn_weights @ v_rep).transpose(1, 2).contiguous().view(1, seq_len, 2048))
    attn_out = (attn_out @ weights[f"model.layers.{layer_idx}.self_attn.o_proj.weight"].T)

    x = x + attn_out
    norm_ffn_x = rms_norm(x, weights[f"model.layers.{layer_idx}.post_attention_layernorm.weight"])

    gate = (norm_ffn_x @ weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"].T)
    up = (norm_ffn_x @ weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"].T)
    ffn_out = ((F.silu(gate) * up) @ weights[f"model.layers.{layer_idx}.mlp.down_proj.weight"].T)

    return x + ffn_out

def forward(tokens: torch.Tensor, weights: dict) -> torch.Tensor:
    seq_len = tokens.shape[1]
    cos, sin = precompute_rope_cache(max_seq_len=max(seq_len, 128))
    
    x = weights["model.embed_tokens.weight"][tokens]
    
    for layer_idx in range(22):
        x = run_single_layer_test(x, weights, layer_idx, seq_len, cos, sin)
        
    x = rms_norm(x, weights["model.norm.weight"])
    
    logits = x @ weights["lm_head.weight"].T
    return logits

if __name__ == "__main__":
    tokens = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(DEVICE)
    
    with torch.no_grad():
        logits = forward(tokens, weights)
        
    next_token = torch.argmax(logits[0, -1, :]).item()
    decoded_word = tokenizer.decode([next_token])
    
    print("-" * 50)
    print(f"Prompt: '{PROMPT}'")
    print(f"Predicted Token ID: {next_token}")
    print(f"Predicted Text: '{decoded_word}'")
    print("-" * 50)
