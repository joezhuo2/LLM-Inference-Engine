import argparse
import datetime
import json
import re
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Answer a few chat prompts with the engine and record their tokens for the KV visualizer on the site")
    parser.add_argument("--engine", default=str(ROOT / "build/release/bin/engine"))
    parser.add_argument("--model", default=str(ROOT / "models/tinyllama"))
    parser.add_argument("--prompts", default=str(ROOT / "scripts/prompts.json"))
    parser.add_argument("--names", default="capital,haiku,kv_cache")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out", default=str(ROOT / "results/kv_demo.json"))
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    prompts = {p["name"]: p["messages"] for p in json.loads(Path(args.prompts).read_text())["prompts"]}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()

    requests = []
    for name in args.names.split(","):
        messages = prompts[name]
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise SystemExit(f"{name}: only single user messages are supported")
        text = messages[0]["content"]
        run = subprocess.run(
            [args.engine, "generate", "--model", args.model, "--prompt", text, "--max-new-tokens", str(args.max_new_tokens)],
            capture_output=True,
            text=True,
            check=True,
        )
        counts = re.search(r"(\d+) prompt tokens, (\d+) generated", run.stderr)
        if not counts:
            raise SystemExit(f"{name}: no token counts in the engine's output:\n{run.stderr}")
        prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
        if hasattr(prompt_ids, "keys"):
            prompt_ids = prompt_ids["input_ids"]
        with torch.no_grad():
            full = model.generate(prompt_ids.cuda(), max_new_tokens=args.max_new_tokens, max_length=None, do_sample=False)[0].tolist()
        prompt_ids = prompt_ids[0].tolist()
        output_ids = full[len(prompt_ids) :]
        answer = run.stdout.rstrip("\n")
        if len(prompt_ids) != int(counts.group(1)) or len(output_ids) != int(counts.group(2)):
            raise SystemExit(f"{name}: engine counted {counts.group(1)} + {counts.group(2)} tokens, HF greedy {len(prompt_ids)} + {len(output_ids)}")
        if tok.decode(output_ids, skip_special_tokens=True).strip() != answer.strip():
            raise SystemExit(f"{name}: HF greedy wrote a different answer from the engine's")
        requests.append(
            {
                "name": name,
                "prompt": text,
                "answer": answer,
                "prompt_tokens": tok.convert_ids_to_tokens(prompt_ids),
                "output_tokens": tok.convert_ids_to_tokens(output_ids),
            }
        )
        print(f"{name}: {len(prompt_ids)} prompt + {len(output_ids)} output tokens")

    out = {
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gpu": gpu,
        "commit": commit,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0,
        "requests": requests,
    }
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
