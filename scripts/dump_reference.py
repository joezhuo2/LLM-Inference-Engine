import argparse
import json
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

TENSORS = {
    "input_ids": "I32 [T]: the prompt (chat template, no BOS) followed by HF's greedy continuation",
    "embed": "BF16 [T, hidden]: embedding lookup",
    "layers.{i}.out": "BF16 [T, hidden]: residual stream after decoder layer i",
    "final_norm": "BF16 [T, hidden]: final RMSNorm of the last layer's output",
    "logits": "BF16 [T, vocab]: lm_head of final_norm, row t predicts token t + 1; teacher-forced checks compare "
    "against these rows, not generated_ids, since BF16 near-ties make the two disagree at a few positions",
}

TRACE = {
    "ln1": "input_layernorm output [T, hidden]",
    "q": "q_proj output [T, heads * head_dim], before RoPE",
    "k": "k_proj output [T, kv_heads * head_dim], before RoPE",
    "v": "v_proj output [T, kv_heads * head_dim]",
    "attn": "o_proj input [T, heads * head_dim]: attention output, heads concatenated",
    "o": "o_proj output [T, hidden]",
    "ln2": "post_attention_layernorm output [T, hidden]",
    "gate": "gate_proj output [T, intermediate]",
    "up": "up_proj output [T, intermediate]",
    "act": "down_proj input [T, intermediate]: silu(gate) * up",
    "down": "down_proj output [T, hidden]",
}


def capture(model, trace: bool):
    store = {}
    hooks = []

    def keep(name, use_input=False):
        def hook(module, inputs, output):
            x = inputs[0] if use_input else output
            x = x[0] if isinstance(x, tuple) else x
            store[name] = x[0].detach().to("cpu").contiguous()

        return hook

    hooks.append(model.model.embed_tokens.register_forward_hook(keep("embed")))
    hooks.append(model.model.norm.register_forward_hook(keep("final_norm")))
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(keep(f"layers.{i}.out")))
        if not trace:
            continue
        p = f"layers.{i}."
        attn, mlp = layer.self_attn, layer.mlp
        hooks += [
            layer.input_layernorm.register_forward_hook(keep(p + "ln1")),
            attn.q_proj.register_forward_hook(keep(p + "q")),
            attn.k_proj.register_forward_hook(keep(p + "k")),
            attn.v_proj.register_forward_hook(keep(p + "v")),
            attn.o_proj.register_forward_hook(keep(p + "attn", use_input=True)),
            attn.o_proj.register_forward_hook(keep(p + "o")),
            layer.post_attention_layernorm.register_forward_hook(keep(p + "ln2")),
            mlp.gate_proj.register_forward_hook(keep(p + "gate")),
            mlp.up_proj.register_forward_hook(keep(p + "up")),
            mlp.down_proj.register_forward_hook(keep(p + "act", use_input=True)),
            mlp.down_proj.register_forward_hook(keep(p + "down")),
        ]
    return store, hooks


def chat_ids(tok, messages) -> list[int]:
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


@torch.no_grad()
def dump_prompt(model, tok, prompt: dict, max_new_tokens: int, trace: bool, out_dir: Path) -> dict:
    device = next(model.parameters()).device
    prompt_ids = chat_ids(tok, prompt["messages"])
    ids = torch.tensor([prompt_ids], device=device)
    generated = model.generate(
        ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens, do_sample=False
    )[0, len(prompt_ids) :].tolist()
    full = torch.tensor([prompt_ids + generated], device=device)

    store, hooks = capture(model, trace)
    try:
        logits = model(full).logits
    finally:
        for h in hooks:
            h.remove()
    store["logits"] = logits[0].to("cpu").contiguous()
    store["input_ids"] = full[0].to(torch.int32).to("cpu").contiguous()

    file = f"{prompt['name']}.safetensors"
    save_file(store, out_dir / file, metadata={"name": prompt["name"], "prompt_len": str(len(prompt_ids))})
    return {
        "name": prompt["name"],
        "file": file,
        "messages": prompt["messages"],
        "prompt_len": len(prompt_ids),
        "total_len": len(prompt_ids) + len(generated),
        "generated_ids": generated,
        "generated_text": tok.decode(generated, skip_special_tokens=True),
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HuggingFace TinyLlama on the fixed prompts and write the reference the engine is "
        "tested against: per-prompt safetensors files with the token ids, every layer's output and the "
        "logits, plus manifest.json."
    )
    parser.add_argument("--model", type=Path, required=True, help="model directory")
    parser.add_argument("--prompts", type=Path, default=Path(__file__).with_name("prompts.json"))
    parser.add_argument("--out", type=Path, required=True, help="output directory, e.g. tests/golden/reference")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--trace", default=None, help="prompt name whose per-layer intermediates are also saved (default: the first)"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager")
    model = model.to(args.device).eval()
    prompts = json.loads(args.prompts.read_text())["prompts"]
    trace = args.trace or prompts[0]["name"]
    if trace not in {p["name"] for p in prompts}:
        parser.error(f"--trace {trace}: no such prompt")

    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    for p in prompts:
        entries.append(dump_prompt(model, tok, p, args.max_new_tokens, p["name"] == trace, args.out))
        e = entries[-1]
        print(f"{e['name']:<16} prompt {e['prompt_len']:>3}  total {e['total_len']:>3}  {e['generated_text'][:60]!r}")

    manifest = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if args.device.startswith("cuda") else args.device,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "max_new_tokens": args.max_new_tokens,
        "tensors": TENSORS,
        "trace_tensors": {f"layers.{{i}}.{k}": v for k, v in TRACE.items()},
        "prompts": entries,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(entries)} prompts written to {args.out}")


if __name__ == "__main__":
    main()
