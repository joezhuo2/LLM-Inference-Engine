"use client";

import { ceilings, hardware } from "@/lib/data";
import { StickyStory, type StoryStep } from "../sticky-story";
import { HierarchyFigure, MemoryFigure, type Tier } from "./figures";

const hw = hardware();
const peak = ceilings();

const blocks = 22;
const threadsPerBlock = 256;

const mb = (kb: number) => `${(kb / 1024).toLocaleString("en-US", { maximumFractionDigits: 1 })} MB`;

const tiers: Tier[] = [
  { name: "global memory (DRAM)", size: `${hw.vram_gb} GB`, note: `${peak.gbs} GB/s measured`, detail: "off the chip, shared by every SM", box: { x: 20, y: 40, w: 300, h: 50 } },
  { name: "L2 cache", size: mb(hw.l2_kb), note: "on the chip, automatic", detail: "shared by every SM", box: { x: 50, y: 130, w: 240, h: 44 } },
  { name: "shared memory", size: `${hw.shared_kb_per_sm} KB per SM`, note: `${hw.sms} SMs, one block's scratchpad`, detail: "the kernel decides what goes in", box: { x: 80, y: 214, w: 180, h: 44 } },
  { name: "registers", size: "per thread", note: "private to one thread", detail: "the fastest, gone when it exits", box: { x: 110, y: 298, w: 120, h: 44 } },
];

const steps: StoryStep[] = [
  {
    id: "hierarchy",
    title: "grid, block, warp, thread",
    body: (
      <>
        <p>
          A CUDA kernel is one function that thousands of threads run at once, each on its own piece of the data. One launch
          is a grid. The grid is cut into blocks, every block has the same number of threads, and a block runs on one
          streaming multiprocessor (SM) from start to finish. This GPU has {hw.sms} SMs.
        </p>
        <p>
          Inside a block the hardware groups threads in warps of 32. A warp executes one instruction at a time for all 32
          threads, so a thread that wants a different branch than its neighbors makes the warp run both. SwiGLU&apos;s first
          draft launched {blocks} blocks of {threadsPerBlock} threads per row: 8 warps in each block.
        </p>
      </>
    ),
  },
  {
    id: "memory",
    title: "four places to keep a number",
    body: (
      <>
        <p>
          A thread can work on data in four places. Global memory is the {hw.vram_gb} GB of DRAM that holds the weights and
          the KV cache; its measured bandwidth on this GPU is{" "}
          <span className="font-mono text-accent">{peak.gbs} GB/s</span>. On a laptop GPU it is GDDR, not HBM, but the role
          is the same. The L2 cache ({mb(hw.l2_kb)}) sits in front of it and every SM shares it.
        </p>
        <p>
          Each SM also has {hw.shared_kb_per_sm} KB of shared memory that the threads of a block use as a scratchpad, and
          every thread has registers of its own. Each step closer to the thread is smaller and faster. All {hw.sms} SMs
          together hold {mb(hw.sms * hw.shared_kb_per_sm)} of shared memory, a tiny fraction of the DRAM.
        </p>
      </>
    ),
  },
];

const captions = [
  `all ${blocks} blocks drawn, one block and one warp zoomed; SM count from docs/hardware.md`,
  `${hw.gpu}: sizes from docs/hardware.md, bandwidth from the bandwidth test in docs/results.md; shared and register speeds are not measured here`,
];

export function MemoryWallStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        switch (i) {
          case 0:
            return <HierarchyFigure sms={hw.sms} blocks={blocks} threadsPerBlock={threadsPerBlock} />;
          default:
            return <MemoryFigure tiers={tiers} />;
        }
      }}
    />
  );
}
