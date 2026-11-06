"use client";

import { motion } from "motion/react";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";
import { StickyStory, type StoryStep } from "../sticky-story";
import { words } from "./prologue-figures";

export const aboutTokens = words.slice(0, 8);
export const aboutScores = [0.3, 2.4, 0.9, 0.1, 0.2, 0.6, 0.4, 1.2];
export const aboutWeights = (() => {
  const e = aboutScores.map((s) => Math.exp(s));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
})();
export const aboutBlock = 4;
export const aboutTable = [3, 1];

export const pagedAttentionAbout: StoryStep = {
  id: "about",
  title: "what paged attention is",
  body: (
    <>
      <p>
        Attention is the one place in a transformer where tokens look at each other. Each of TinyLlama&apos;s 22 layers has one, between a normalization and the MLP, and everything else in a layer works on every token alone.
      </p>
      <p>
        Three learned matrices turn each token into a query, a key and a value: what the token is looking for, what it offers, and what it hands over. The newest token&apos;s query is scored against every key in the cache, its own included, as a dot product divided by √d, where d is the 64 dimensions of a head. A softmax turns the scores into weights that add up to 1, and the output is the values summed with those weights.
      </p>
      <p>
        Paged is borrowed from operating systems, where virtual memory cuts a program&apos;s memory into fixed-size pages that can sit anywhere and are found through a page table. Paged attention, introduced with vLLM in 2023, does the same with the KV cache: keys and values live in blocks, a block table says where each of a sequence&apos;s blocks sits, and the kernel looks every key up there before scoring it. The arithmetic is unchanged; only where each key is read from changes.
      </p>
      <p>The next pages follow where those keys and values come from and why they end up in blocks. Then the kernel itself.</p>
    </>
  ),
};

export const pagedAttentionAboutCaption = "illustrative: 8 tokens in blocks of 4 with made-up scores; the engine uses 16-token blocks and heads of 64 dimensions";

const col = (i: number) => 20 + i * 66;
const base = 160;
const tall = 70;
const scan = (i: number) => 0.04 + i * 0.05;
const blockX = (p: number) => 20 + p * 132;
const slotX = (p: number, s: number) => blockX(p) + 8 + s * 28;
const poolY = 262;
const maxScore = Math.max(...aboutScores);
const maxWeight = Math.max(...aboutWeights);
const newest = aboutTokens.length - 1;

export function PagedAttentionAboutFigure() {
  const { on, grow } = useLoop(16);
  return (
    <Frame label="The newest token's query is scored against every key, its own included; a softmax turns the scores into weights; each key is found through the block table in a scattered block of the pool">
      <Label x={20} y={24}>{`the newest token, "${aboutTokens[newest]}", scores every key in the cache, its own included`}</Label>
      {aboutTokens.map((w, i) => {
        const t = scan(i);
        const p = aboutTable[Math.floor(i / aboutBlock)];
        const s = i % aboutBlock;
        return (
          <g key={i}>
            <rect x={col(i)} y={36} width={60} height={26} rx={3} className={i === newest ? "fill-accent/15 stroke-accent" : "fill-surface stroke-border"} />
            <Label x={col(i) + 30} y={53} anchor="middle" className="fill-text">
              {w}
            </Label>
            <motion.rect x={col(i) - 2} y={34} width={64} height={30} rx={4} className="fill-none stroke-accent" strokeWidth={2} {...on(t, t + 0.05, false)} />
            <motion.rect x={col(i) + 6} y={base - (aboutScores[i] / maxScore) * tall} width={20} height={(aboutScores[i] / maxScore) * tall} rx={2} className="fill-baseline" {...grow(t, t + 0.05, "y")} />
            <motion.rect x={col(i) + 34} y={base - (aboutWeights[i] / maxWeight) * tall} width={20} height={(aboutWeights[i] / maxWeight) * tall} rx={2} className="fill-accent" {...grow(0.46, 0.56, "y")} />
            <motion.rect x={slotX(p, s)} y={poolY + 6} width={24} height={22} rx={2} className="fill-accent" {...on(t, t + 0.05, false)} />
          </g>
        );
      })}
      <line x1={20} y1={base + 0.5} x2={540} y2={base + 0.5} className="stroke-border" />
      <rect x={20} y={170} width={10} height={10} rx={2} className="fill-baseline" />
      <Label x={36} y={179}>score q · k / √d</Label>
      <motion.g {...on(0.46)}>
        <rect x={160} y={170} width={10} height={10} rx={2} className="fill-accent" />
        <Label x={176} y={179}>weight after softmax: all eight add up to 1</Label>
      </motion.g>
      {aboutTable.map((p, k) => (
        <g key={k}>
          <rect x={col(k * aboutBlock)} y={196} width={4 * 66 - 6} height={24} rx={3} className="fill-bg stroke-border" />
          <Label x={col(k * aboutBlock) + 10} y={212} className="fill-text">{`block table: logical ${k} → physical ${p}`}</Label>
          <line x1={col(k * aboutBlock) + 129} y1={220} x2={blockX(p) + 60} y2={poolY} className="stroke-accent" strokeWidth={1.5} />
        </g>
      ))}
      {Array.from({ length: 4 }, (_, p) => {
        const k = aboutTable.indexOf(p);
        return (
          <g key={p}>
            <rect x={blockX(p)} y={poolY} width={120} height={34} rx={3} className={k >= 0 ? "fill-accent/10 stroke-accent" : p === 0 ? "fill-baseline/25 stroke-border" : "fill-surface stroke-border"} />
            {Array.from({ length: aboutBlock }, (_, s) => (
              <rect key={s} x={slotX(p, s)} y={poolY + 6} width={24} height={22} rx={2} className={p === 2 ? "fill-none stroke-border" : "fill-surface stroke-border"} strokeWidth={0.5} />
            ))}
            <Label x={blockX(p)} y={poolY + 50}>{`block ${p}${p === 0 ? ", another request" : p === 2 ? ", free" : ""}`}</Label>
          </g>
        );
      })}
      <Label x={20} y={340} className="fill-text">{`key of token t: physical block table[t / ${aboutBlock}], slot t % ${aboutBlock}`}</Label>
      <motion.g {...on(0.6)}>
        <Label x={20} y={364} className="fill-accent">out = Σ weight × value: the values mixed by how well each key matched</Label>
      </motion.g>
      <Label x={20} y={388}>the arithmetic is the same as without pages; only where each key is read from changes</Label>
    </Frame>
  );
}

export function PagedAttentionAboutStory() {
  return <StickyStory kicker="the idea" steps={[pagedAttentionAbout]} caption={() => pagedAttentionAboutCaption} figure={() => <PagedAttentionAboutFigure />} />;
}
