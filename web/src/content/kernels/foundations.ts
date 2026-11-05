import { type KernelStory, measured } from "./types";

const roof = "roofline";

const wholeStep = measured(roof, { "whole decode step": "engine + graphs", rows: "1" }, "% of roof", 1, "results");

export const foundations: KernelStory = {
  slug: "foundations",
  index: 0,
  title: "foundations",
  short: "How a GPU runs a kernel, where its data lives, and why decoding one token waits on memory and not on arithmetic.",
  formula: "time ≈ max(bytes ÷ bandwidth, FLOPs ÷ compute)",
  headline: {
    value: `${wholeStep}%`,
    label: "of the bandwidth roof for a whole batch-1 decode step",
  },
  sources: [
    { path: "docs/hardware.md", label: "SMs, shared memory, L2, registers, VRAM" },
    { path: "docs/results.md", label: "roofline and ceilings" },
    { path: "scripts/roofline.py", label: "roofline" },
    { path: "scripts/gemm_peak.cu", label: "compute ceiling" },
    { path: "src/practice/bandwith_test.cu", label: "bandwidth ceiling" },
    { path: "scripts/onchip_bench.cu", label: "L2, shared memory, register and latency numbers" },
  ],
  tests: [],
  testing: "",
  excerpts: [],
  measurements: [
    {
      doc: "results",
      section: roof,
      table: 0,
      caption: "each GEMM and attention at 1 and 16 rows per decode step: FLOPs and bytes counted from config.json, time from the Nsight Systems profiles",
      command: "python3 scripts/roofline.py",
    },
    {
      doc: "results",
      section: roof,
      table: 1,
      caption: "whole decode steps of the engine and the baselines against the same two ceilings",
      command: "python3 scripts/roofline.py",
    },
    {
      doc: "results",
      section: "on-chip-memory",
      table: 0,
      caption: "L2, shared memory, register and DRAM latency microbenchmarks: median and spread of the runs, SM clock measured inside each kernel",
      command: "nvcc -std=c++20 -O3 -arch=sm_120 scripts/onchip_bench.cu -o build/onchip_bench && python3 scripts/onchip_report.py",
    },
  ],
  timeline: [],
};
