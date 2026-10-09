import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

HAND_WRITTEN = [
    "",
    " ",
    "  ",
    "Hello world",
    " Hello",
    "  leading spaces",
    "trailing ",
    "multi   space",
    "\nnewline",
    "a\n\nb",
    "tab\tsep",
    "x\r\ny",
    "</s> literal",
    "</s>abc",
    "a</s>b",
    "a </s> b",
    "<s>x",
    "x<unk>y",
    "</s></s>",
    "<</s>>",
    "<|user|>\nHi</s>\n<|assistant|>\n",
    "code: if (a<b) { return a>>2; }",
    "日本語 text",
    "emoji 🙂 ok",
    "Ünïcödé ñ",
    "a b",
    "'quoted' \"double\"",
    "1234567890",
]

ALPHABET = list("abcdefgh XYZ.,;:!?<>|/\\\n\t'\"0123456789") + ["</s>", "<s>", "<unk>", "é", "日", "🙂", "  "]


def fuzz_strings(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 30))) for _ in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write HuggingFace encode and decode results for hand-written and random strings, "
        "the golden for the C++ tokenizer."
    )
    parser.add_argument("--model", type=Path, required=True, help="model directory")
    parser.add_argument("--out", type=Path, required=True, help="output JSON file")
    parser.add_argument("--count", type=int, default=5000, help="number of random strings")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    cases = []
    for text in HAND_WRITTEN + fuzz_strings(args.count, args.seed):
        ids = tok(text)["input_ids"]
        cases.append({"text": text, "ids": ids, "decoded": tok.decode(ids, skip_special_tokens=True)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(cases)} cases written to {args.out}")


if __name__ == "__main__":
    main()
