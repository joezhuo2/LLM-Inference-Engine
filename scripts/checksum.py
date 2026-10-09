import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def checksums(model_dir: Path) -> dict:
    out = {}
    with safe_open(model_dir / "model.safetensors", framework="pt") as f:
        for name in sorted(f.keys()):
            x = f.get_tensor(name).to(torch.float64)
            out[name] = {
                "dtype": f.get_slice(name).get_dtype(),
                "shape": list(x.shape),
                "sum": x.sum().item(),
                "abs_sum": x.abs().sum().item(),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print an FP64 checksum of every tensor in model.safetensors, computed by PyTorch, "
        "in the same format as `engine checksum`."
    )
    parser.add_argument("--model", type=Path, required=True, help="model directory")
    parser.add_argument("--out", type=Path, help="also write the checksums as golden JSON to this file")
    args = parser.parse_args()

    tensors = checksums(args.model)
    for name, t in tensors.items():
        shape = "[" + ", ".join(map(str, t["shape"])) + "]"
        print(f"{name:<52} {t['dtype']:<5} {shape:<14} {t['sum']:+.10e} {t['abs_sum']:.10e}")
    print(f"{len(tensors)} tensors, checksummed by PyTorch")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"tensors": tensors, "source": "pytorch"}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
