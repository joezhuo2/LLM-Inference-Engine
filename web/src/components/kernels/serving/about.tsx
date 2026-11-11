"use client";

import { motion } from "motion/react";
import { blockTokens, fmt, maxBatch, poolBlocks } from "@/content/kernels/serving";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";
import { StickyStory, type StoryStep } from "../sticky-story";

export const servingAbout: StoryStep = {
  id: "about",
  title: "what a scheduler decides",
  body: (
    <>
      <p>
        A GPU that answered one request at a time would read all of its weights for a single row of work. So the engine runs many requests in the same forward pass: one step, one row for each request that is decoding. Something has to choose, before every step, whose rows those are. That is the scheduler.
      </p>
      <p>
        A request is in one of three states: waiting in a queue, running with KV blocks of its own, or finished. Before each step the scheduler looks at the queue and at the running list and builds the step. The model runs it, one new token comes back for each row, and the scheduler looks again.
      </p>
      <p>
        It makes three decisions. Admission: may the request at the front of the queue start now? Growth: each running request needs room for one more token, so is there a block for it? Preemption: if there is not, who gives memory back?
      </p>
      <p>
        Two things can run out. Memory, because every cached token takes bytes from a pool that is fixed at startup. And time, because every row makes the step longer and every running request waits for the whole step. This chapter is about which of the two runs out first, and what the scheduler does when it happens.
      </p>
    </>
  ),
};

export const servingAboutCaption = `illustrative: four rows and a handful of requests; the engine's defaults are max_batch ${maxBatch} and a pool of ${fmt(poolBlocks)} blocks of ${blockTokens} tokens; the states are RequestState in include/engine/sched/request.h`;

const chip = { w: 34, h: 18 };
const rows = 4;
const rowY = (k: number) => 84 + k * 30;
const waitX = 28;
const runX = 214;
const doneX = 456;

function Chip({ x, y, name, cls = "fill-accent" }: { x: number; y: number; name: string; cls?: string }) {
  return (
    <g>
      <rect x={x} y={y} width={chip.w} height={chip.h} rx={3} className={cls} />
      <text x={x + chip.w / 2} y={y + 13} textAnchor="middle" className="fill-bg text-[10px]">
        {name}
      </text>
    </g>
  );
}

export function ServingAboutFigure() {
  const { on, move, grow } = useLoop(14);
  return (
    <Frame label={`A scheduler between a waiting queue, a running list of at most max_batch rows and the finished requests. Before every step it admits the front of the queue if a row and blocks are free, gives each running request one more token, and when no block is free sends the newest running request back to the front of the queue. Illustrative`}>
      <Label x={20} y={22} className="fill-text">before every step: who runs, who waits, who gives memory back</Label>
      <rect x={20} y={40} width={150} height={170} rx={5} className="fill-surface stroke-border" />
      <Label x={28} y={58}>waiting queue</Label>
      <rect x={200} y={40} width={216} height={170} rx={5} className="fill-surface stroke-border" />
      <Label x={208} y={58}>running list, at most max_batch rows</Label>
      <rect x={446} y={40} width={94} height={170} rx={5} className="fill-surface stroke-border" />
      <Label x={454} y={58}>finished</Label>
      {["A", "B", "C"].map((name, k) => (
        <g key={name}>
          <Chip x={runX} y={rowY(k)} name={name} />
          <rect x={runX + chip.w + 6} y={rowY(k) + 3} width={150} height={12} rx={2} className="fill-bg stroke-border" strokeWidth={0.5} />
          <motion.rect x={runX + chip.w + 6} y={rowY(k) + 3} width={50 + k * 30} height={12} rx={2} className="fill-accent/55" {...grow(0.04, 0.9)} />
        </g>
      ))}
      <rect x={runX - 4} y={rowY(3) - 4} width={chip.w + 8} height={chip.h + 8} rx={4} className="fill-none stroke-border" strokeDasharray="3 2" />
      <Chip x={waitX + 44} y={rowY(1)} name="F" cls="fill-baseline" />
      <Chip x={waitX + 88} y={rowY(1)} name="G" cls="fill-baseline" />
      <motion.g {...on(0, 0.1, false)}>
        <Chip x={waitX} y={rowY(1)} name="E" cls="fill-baseline" />
      </motion.g>
      <motion.g {...move(0.1, 0.24, runX - waitX, rowY(3) - rowY(1), true, 0.52)}>
        <Chip x={waitX} y={rowY(1)} name="E" />
      </motion.g>
      <motion.g {...move(0.56, 0.7, waitX - runX, rowY(1) - rowY(3), false, 0.9)}>
        <Chip x={runX} y={rowY(3)} name="E" cls="fill-warn" />
      </motion.g>
      <motion.g {...move(0.3, 0.44, doneX - runX, 0, true, 0.94)}>
        <Chip x={runX} y={rowY(0)} name="A" />
      </motion.g>
      <motion.g {...on(0.1, 0.52)}>
        <Label x={20} y={234} className="fill-accent">1 admission: a row is free and so are blocks for E&apos;s prompt, so E starts</Label>
      </motion.g>
      <motion.g {...on(0.3, 0.52)}>
        <Label x={20} y={252} className="fill-text">2 growth: every row gets one token a step; A reaches its end and leaves</Label>
      </motion.g>
      <motion.g {...on(0.56, 0.9)}>
        <Label x={20} y={234} className="fill-warn">3 preemption: a request needs a block and none is free,</Label>
        <Label x={20} y={252} className="fill-warn">so E, the newest, gives its blocks back and returns to the front of the queue</Label>
      </motion.g>
      <Label x={20} y={292} className="fill-text">two things run out</Label>
      <Label x={20} y={314}>memory: blocks held, of the pool</Label>
      <rect x={220} y={303} width={320} height={14} rx={2} className="fill-bg stroke-border" />
      <motion.rect x={220} y={303} width={320} height={14} rx={2} className="fill-accent" {...grow(0.04, 0.56)} />
      <Label x={20} y={342}>time: rows in the step, of max_batch</Label>
      <rect x={220} y={331} width={320} height={14} rx={2} className="fill-bg stroke-border" />
      {Array.from({ length: rows }, (_, k) => (
        <motion.rect key={k} x={220 + k * 80} y={331} width={77} height={14} rx={2} className="fill-accent" {...on(k < 3 ? 0 : 0.24, k === 0 ? 0.44 : k === 3 ? 0.56 : 0.94)} />
      ))}
      <Label x={20} y={378}>{`illustrative; the engine's defaults are max_batch ${maxBatch} and ${fmt(poolBlocks)} blocks of ${blockTokens} tokens`}</Label>
    </Frame>
  );
}

export function ServingAboutStory() {
  return <StickyStory kicker="the idea" steps={[servingAbout]} caption={() => servingAboutCaption} figure={() => <ServingAboutFigure />} />;
}
