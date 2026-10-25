#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot  # noqa: E402

PEAK_GB_S = 323.6
TARGET_BANDWIDTH_PCT = 60.0
TARGET_SPEEDUP = 5.0
TARGET_ARGMAX_PCT = 99.0
TARGET_LOGIT_ERROR = 0.05
TARGET_TIE_ULPS = 2

TEACHER = re.compile(r"teacher-forced argmax (\d+)/(\d+) \(([\d.]+)%\), mean \|dlogit\| ([\d.]+)")
FREE = re.compile(r"free-running greedy identical to HF on (\d+) of (\d+) prompts")
TIE = re.compile(r"differs at step (\d+): .* (\d+) BF16 ulps apart")


def parse_correctness(text):
    out = {}
    m = TEACHER.search(text)
    if m:
        out.update(argmax_matches=int(m[1]), argmax_total=int(m[2]), argmax_pct=float(m[3]),
                   mean_abs_logit=float(m[4]))
    m = FREE.search(text)
    if m:
        out.update(exact_prompts=int(m[1]), prompts=int(m[2]))
    gaps = [int(g) for _, g in TIE.findall(text)]
    out["near_ties"] = len(gaps)
    out["max_tie_ulps"] = max(gaps) if gaps else 0
    return out


def parse_environment(text):
    env = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            env[key.strip()] = value.strip()
    return env


ENGINE = ("engine", "engine + graphs")


def median_of(rows, system, batch, key):
    values = sorted(r[key] for r in rows if r["system"] == system and r["max_batch"] == batch
                    and r[key] == r[key])
    return values[len(values) // 2] if values else None


def best(rows, systems, batch, key):
    if isinstance(systems, str):
        systems = (systems,)
    medians = [m for m in (median_of(rows, s, batch, key) for s in systems) if m is not None]
    return max(medians) if medians else None


def verdict(ok):
    return "met" if ok else "not met"


def scoreboard(rows, correctness, peak=PEAK_GB_S):
    lines = ["| # | Criterion | Target | Measured | |", "|---|---|---|---|---|"]
    c = correctness
    if "argmax_pct" in c:
        ok = c["argmax_pct"] >= TARGET_ARGMAX_PCT and c["mean_abs_logit"] < TARGET_LOGIT_ERROR
        measured = (f"{c['argmax_pct']:.2f}% argmax over {c['argmax_total']} positions, "
                    f"mean abs logit error {c['mean_abs_logit']:.4f}")
        lines.append(f"| 1a | Teacher-forced vs HuggingFace | at least {TARGET_ARGMAX_PCT:g}% argmax, "
                     f"error under {TARGET_LOGIT_ERROR} | {measured} | {verdict(ok)} |")
    else:
        lines.append("| 1a | Teacher-forced vs HuggingFace | | not recorded | |")
    if "exact_prompts" in c:
        unmatched = c["prompts"] - c["exact_prompts"]
        ok = c["near_ties"] == unmatched and c["max_tie_ulps"] <= TARGET_TIE_ULPS
        note = ""
        if unmatched:
            ties = "the other" if c["near_ties"] == unmatched else f"{c['near_ties']} of the other"
            note = (f"; {ties} {unmatched} diverge at a near tie "
                    f"(at most {c['max_tie_ulps']} BF16 ulps apart in HF's logits)")
        lines.append(f"| 1b | Free-running greedy, first 64 tokens | exact, or diverging where HF's two "
                     f"logits are at most {TARGET_TIE_ULPS} BF16 ulps apart, on all {c['prompts']} | "
                     f"exact on {c['exact_prompts']} of {c['prompts']}{note} | {verdict(ok)} |")
    else:
        lines.append("| 1b | Free-running greedy | | not recorded | |")

    bw = best(rows, ENGINE, 1, "bandwidth_gb_s")
    if bw is not None:
        ok = 100.0 * bw / peak >= TARGET_BANDWIDTH_PCT
        lines.append(f"| 2 | Batch-1 decode bandwidth | at least {TARGET_BANDWIDTH_PCT:g}% of {peak:g} GB/s | "
                     f"{bw:.0f} GB/s = {100.0 * bw / peak:.0f}% | {verdict(ok)} |")
    else:
        lines.append("| 2 | Batch-1 decode bandwidth | | not recorded | |")

    t1 = best(rows, ENGINE, 1, "output_throughput")
    t16 = best(rows, ENGINE, 16, "output_throughput")
    if t1 and t16:
        ok = t16 / t1 >= TARGET_SPEEDUP
        lines.append(f"| 3 | Throughput at 16 concurrent requests | at least {TARGET_SPEEDUP:g}x batch 1 | "
                     f"{t16:.0f} vs {t1:.0f} tokens/s = {t16 / t1:.1f}x | {verdict(ok)} |")
    else:
        lines.append("| 3 | Throughput at 16 concurrent requests | | not recorded | |")
    return "\n".join(lines) + "\n"


def environment_block(env):
    order = [("date", "Date"), ("commit", "Commit"), ("gpu", "GPU"), ("driver", "Driver"),
             ("cuda", "CUDA"), ("build", "Build"), ("peak_gb_s", "Measured DRAM bandwidth (GB/s)"),
             ("peak_tflops", "Measured BF16 GEMM peak (TFLOP/s)"),
             ("workload", "Workload")]
    return "".join(f"- {label}: {env[key]}\n" for key, label in order if key in env)


def replace_block(text, name, content):
    begin = f"<!-- auto:{name} -->"
    end = f"<!-- /auto:{name} -->"
    i = text.find(begin)
    j = text.find(end)
    if i < 0 or j < 0 or j < i:
        raise ValueError(f"missing markers for block '{name}'")
    return text[:i + len(begin)] + "\n" + content.rstrip("\n") + "\n" + text[j:]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Refresh the measured blocks of docs/results.md")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--doc", type=Path, default=Path("docs/results.md"))
    ap.add_argument("--peak-gb-s", type=float, default=PEAK_GB_S)
    args = ap.parse_args(argv)

    summaries = sorted(args.results_dir.glob("*_summary.csv"))
    rows = plot.load_summaries(summaries)
    correctness_path = args.results_dir / "correctness.txt"
    correctness = parse_correctness(correctness_path.read_text()) if correctness_path.exists() else {}
    env_path = args.results_dir / "environment.txt"
    env = parse_environment(env_path.read_text()) if env_path.exists() else {}

    profile_path = args.results_dir / "decode_step.md"
    profile = ("No profile recorded.\n" if not profile_path.exists()
               else profile_path.read_text().replace("\n## ", "\n### ").replace("## ", "### ", 1))

    graphs_path = args.results_dir / "decode_step_graphs.md"
    graphs_profile = ("No profile recorded.\n" if not graphs_path.exists()
                      else graphs_path.read_text().replace("\n## ", "\n### ").replace("## ", "### ", 1))

    text = args.doc.read_text()
    text = replace_block(text, "environment", environment_block(env))
    text = replace_block(text, "profile", profile)
    if "<!-- auto:profile_graphs -->" in text:
        text = replace_block(text, "profile_graphs", graphs_profile)
    roofline_path = args.results_dir / "plots" / "roofline.md"
    if "<!-- auto:roofline -->" in text:
        text = replace_block(text, "roofline", roofline_path.read_text() if roofline_path.exists()
                             else "No roofline recorded.\n")
    text = replace_block(text, "scoreboard", scoreboard(rows, correctness, args.peak_gb_s))
    text = replace_block(text, "table", plot.table_markdown(rows) if rows else "No summaries found.\n")
    args.doc.write_text(text)
    print(f"updated {args.doc} from {len(summaries)} summaries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
