"use client";

import { motion } from "motion/react";
import { chunkBudget, chunkedSteps, fmt } from "@/content/kernels/chunked-prefill";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";
import { StickyStory, type StoryStep } from "../sticky-story";

export const chunkedPrefillAbout: StoryStep = {
  id: "about",
  title: "what prefill, decode and a chunk are",
  body: (
    <>
      <p>
        A request is answered in two phases. Prefill runs every token of the prompt through the model in one forward pass: all 22 layers see all the positions together, each layer writes their keys and values into the KV cache, and the logits of the last position give the first output token. The wait for it is the time to first token, TTFT.
      </p>
      <p>
        Decode follows. Each step feeds the newest token back in as a single row, reads the keys and values already in the cache instead of computing them again, and produces the next token. A decoding request moves exactly one token per step, so the time between its tokens is the length of a step.
      </p>
      <p>
        The engine serves many requests at once, and a step is one forward pass over all their rows together: one row for each decoding request, and one for each prompt token in prefill. A step costs more the more rows it has, and nobody gets a token until it ends, so a step that holds a long prompt makes every decoding request wait for that whole prefill.
      </p>
      <p>
        Chunked prefill cuts the prompt into pieces. Each step gets a token budget, a cap on its rows: the decode rows go in first and a chunk of the prompt fills what is left. The prompt fills its cache over several steps, and no step grows past the budget, so the decoders keep their pace. The rest of this chapter is how the engine&apos;s scheduler picks those rows.
      </p>
    </>
  ),
};

export const chunkedPrefillAboutCaption = "illustrative: one request with a 6-token prompt above; below, 3 decoding requests and a new prompt, each step drawn in proportion to its rows; prefill times are measured alone, docs/results.md";

const lengths = ["128", "512", "1,024", "2,048"];
const promptTokens = 6;
const decodeSteps = 3;
const decoders = 3;
const exampleTokens = 1024;
const budget = Number(chunkBudget);

export const chunkedPrefillAboutNumbers = {
  lengths,
  promptTokens,
  decodeSteps,
  decoders,
  budget,
};

const cell = 22;
const pitch = 26;
const promptX = 20;
const topY = 74;
const modelX = 196;
const cacheX = 300;
const cacheY = 62;
const outY = 112;
const decodeAt = (k: number) => 0.32 + k * 0.17;

const laneNames = ["A", "B", "C", "P"];
const laneH = 13;
const lanePitch = 16;
const stripX = 40;
const stripEnd = 540;
const minStep = 14;
const stepGap = 3;
const perToken = 300 / exampleTokens;

type Column = { x: number; w: number; decode: boolean[]; chunk: number; wait: boolean };

function columns(chunked: boolean): Column[] {
  const out: Column[] = [];
  let x = stripX;
  const push = (w: number, decode: boolean[], chunk: number, wait: boolean) => {
    out.push({ x, w, decode, chunk, wait });
    x += w + stepGap;
  };
  push(minStep, [true, true, true, false], 0, false);
  if (chunked) {
    for (const s of chunkedSteps(budget, decoders, exampleTokens)) push(Math.max(minStep, (s.decode + s.chunk) * perToken), [true, true, true, false], s.chunk, false);
  } else {
    push(exampleTokens * perToken, [false, false, false, false], exampleTokens, true);
  }
  while (x + minStep <= stripEnd) push(minStep, [true, true, true, true], 0, false);
  return out;
}

function Strip({ y, chunked, on }: { y: number; chunked: boolean; on: ReturnType<typeof useLoop>["on"] }) {
  const lane = (i: number) => y + i * lanePitch;
  const at = (x: number) => 0.06 + ((x - stripX) / (stripEnd - stripX)) * 0.8;
  return (
    <g>
      {laneNames.map((n, i) => (
        <g key={n}>
          <text x={22} y={lane(i) + 10} className={i === 3 ? "fill-series-2 text-[11px]" : "fill-accent text-[11px]"}>
            {n}
          </text>
          <line x1={stripX - 2} y1={lane(i) + laneH + 1.5} x2={stripEnd} y2={lane(i) + laneH + 1.5} className="stroke-border" />
        </g>
      ))}
      {columns(chunked).map((c) => (
        <motion.g key={c.x} {...on(at(c.x))}>
          {c.decode.map((d, i) => (d ? <rect key={i} x={c.x} y={lane(i)} width={Math.min(c.w, minStep)} height={laneH} rx={2} className="fill-accent" /> : null))}
          {c.wait &&
            [0, 1, 2].map((i) => (
              <rect key={i} x={c.x} y={lane(i)} width={c.w} height={laneH} rx={2} className="fill-none stroke-warn" strokeDasharray="4 3" />
            ))}
          {c.wait && (
            <text x={c.x + c.w / 2} y={lane(1) + 10} textAnchor="middle" className="fill-warn text-[10px]">
              A, B and C wait
            </text>
          )}
          {c.chunk > 0 && <rect x={c.decode[0] ? c.x + minStep + 1 : c.x} y={lane(3)} width={Math.max(2, c.w - (c.decode[0] ? minStep + 1 : 0))} height={laneH} rx={2} className="fill-series-2" />}
        </motion.g>
      ))}
    </g>
  );
}

export function ChunkedPrefillAboutFigure() {
  const { on, move } = useLoop(16);
  const chunks = chunkedSteps(budget, decoders, exampleTokens).length;
  return (
    <Frame
      label={`Top: one request. Prefill sends all ${promptTokens} prompt tokens through the model at once, the KV cache fills with their keys and values and the first token comes out; each decode step then sends one token in and gets one token out. Bottom: three decoding requests A, B and C and a new ${fmt(exampleTokens)}-token prompt P. As one step, the prompt holds a long step and A, B and C wait; in chunks under a budget of ${budget}, it takes ${chunks} steps and A, B and C get a token in each. Sizes are illustrative`}
    >
      <motion.g {...on(0.02, 0.3, false)}>
        <Label x={20} y={24} className="fill-text">{`prefill: all ${promptTokens} prompt tokens in one pass, the cache fills, token 1 comes out`}</Label>
      </motion.g>
      <motion.g {...on(0.31, 0.97)}>
        <Label x={20} y={24} className="fill-text">decode: each step, one token in, one row of cache, one token out</Label>
      </motion.g>
      <Label x={promptX} y={topY - 10} className="fill-series-2">prompt</Label>
      {Array.from({ length: promptTokens }, (_, i) => (
        <g key={`p${i}`}>
          <rect x={promptX + i * pitch} y={topY} width={cell} height={cell} rx={3} className="fill-series-2/30" />
          <motion.g {...move(0.04, 0.16, 36, 0, false, 0.18)}>
            <rect x={promptX + i * pitch} y={topY} width={cell} height={cell} rx={3} className="fill-series-2" />
          </motion.g>
        </g>
      ))}
      {Array.from({ length: decodeSteps }, (_, k) => (
        <motion.g key={`in${k}`} {...move(decodeAt(k), decodeAt(k) + 0.06, 20, 0, false, decodeAt(k) + 0.08)}>
          <rect x={promptX + (promptTokens - 1) * pitch + 6} y={topY} width={cell} height={cell} rx={3} className="fill-accent" />
        </motion.g>
      ))}
      <rect x={modelX} y={56} width={76} height={76} rx={6} className="fill-surface stroke-border" />
      <Label x={modelX + 38} y={92} anchor="middle" className="fill-text">model</Label>
      <Label x={modelX + 38} y={108} anchor="middle">22 layers</Label>
      <Label x={cacheX} y={cacheY - 8}>KV cache: keys and values of every position</Label>
      {Array.from({ length: promptTokens + decodeSteps }, (_, i) => {
        const prompt = i < promptTokens;
        const at = prompt ? 0.14 : decodeAt(i - promptTokens) + 0.07;
        return (
          <g key={`kv${i}`}>
            <rect x={cacheX + i * pitch} y={cacheY} width={cell} height={cell} rx={3} className="fill-none stroke-border" />
            <motion.rect x={cacheX + i * pitch} y={cacheY} width={cell} height={cell} rx={3} className={prompt ? "fill-series-2" : "fill-accent"} {...on(at, 0.97)} />
          </g>
        );
      })}
      <Label x={cacheX} y={outY + cell + 16}>tokens out, one per pass</Label>
      {Array.from({ length: decodeSteps + 1 }, (_, i) => {
        const at = i === 0 ? 0.2 : decodeAt(i - 1) + 0.1;
        return (
          <g key={`o${i}`}>
            <rect x={cacheX + i * pitch} y={outY} width={cell} height={cell} rx={3} className="fill-none stroke-border" />
            <motion.rect x={cacheX + i * pitch} y={outY} width={cell} height={cell} rx={3} className="fill-accent" {...on(at, 0.97)} />
          </g>
        );
      })}
      <Label x={20} y={182} className="fill-text">{`a ${fmt(exampleTokens)}-token prompt P as one step: A, B and C get nothing while it runs`}</Label>
      <Strip y={192} chunked={false} on={on} />
      <Label x={20} y={276} className="fill-text">{`in chunks under a budget of ${budget} rows: ${chunks} steps, and A, B and C get a token in each`}</Label>
      <Strip y={286} chunked on={on} />
      <Label x={20} y={378}>a column is a step, its width in proportion to its rows; sizes are illustrative</Label>
    </Frame>
  );
}

export function ChunkedPrefillAboutStory() {
  return <StickyStory kicker="the idea" steps={[chunkedPrefillAbout]} caption={() => chunkedPrefillAboutCaption} figure={() => <ChunkedPrefillAboutFigure />} />;
}
