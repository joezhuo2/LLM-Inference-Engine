import argparse
import datetime
import json
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent

CONTEXTS = [
    {"name": "capital", "prompt": "capital", "prefix": "The capital of Canada is", "note": "one clear answer"},
    {"name": "haiku", "prompt": "haiku", "prefix": "", "note": "an open start"},
    {"name": "story", "prompt": "story", "prefix": "Once upon a time, a robot named", "note": "many names fit"},
]


def main():
    parser = argparse.ArgumentParser(description="Dump the top next-token logits of a few contexts on the CPU for the sampling sandbox on the site")
    parser.add_argument("--model", default=str(ROOT / "models/tinyllama"))
    parser.add_argument("--prompts", default=str(ROOT / "scripts/prompts.json"))
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--bin", type=float, default=0.25)
    parser.add_argument("--out", default=str(ROOT / "results/sampling_logits.json"))
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cpu").eval()
    prompts = {p["name"]: p["messages"] for p in json.loads(Path(args.prompts).read_text())["prompts"]}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()

    contexts = []
    for c in CONTEXTS:
        messages = prompts[c["prompt"]]
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) + c["prefix"]
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()
        values, index = logits.topk(args.top)
        rest = torch.ones_like(logits, dtype=torch.bool)
        rest[index] = False
        bins = torch.round(logits[rest] / args.bin) * args.bin
        centers, counts = bins.unique(return_counts=True)
        contexts.append(
            {
                "name": c["name"],
                "note": c["note"],
                "user": messages[-1]["content"],
                "answer_so_far": c["prefix"],
                "vocab": logits.numel(),
                "tokens": tok.convert_ids_to_tokens(index.tolist()),
                "logits": [round(v, 4) for v in values.tolist()],
                "tail": [[round(c, 4), n] for c, n in zip(centers.tolist(), counts.tolist(), strict=True)][::-1],
            }
        )
        print(f"{c['name']}: {ids.shape[1]} tokens, top {tok.convert_ids_to_tokens(index[:3].tolist())}, {len(centers)} tail bins")

    out = {
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": "CPU, bfloat16 Hugging Face reference",
        "commit": commit,
        "top": args.top,
        "bin": args.bin,
        "contexts": contexts,
    }
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
