"use client";

import { motion } from "motion/react";
import { chatRows, piece, repeatRows, rowTimes, spec, specFmt } from "@/content/kernels/speculative-decoding";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

const fmt = specFmt;
const X = 20;
const W = 520;
const example = spec.example;
const all = [...example.prompt_tokens, ...example.output_tokens];
const tokens = (n: number) => (n === 1 ? "token" : `${n} tokens`);
const short = (token: string, most = 6) => {
  const text = piece(token).trim() || "␣";
  return text.length > most ? `${text.slice(0, most - 1)}…` : text;
};

function Chip({ x, y, w, text, tone }: { x: number; y: number; w: number; text: string; tone: "plain" | "pattern" | "draft" | "kept" | "dropped" | "real" }) {
  const box = { plain: "fill-border", pattern: "fill-accent/30", draft: "fill-warn/20", kept: "fill-accent", dropped: "fill-none stroke-warn", real: "fill-text/70" }[tone];
  const ink = tone === "kept" || tone === "real" ? "fill-bg" : tone === "dropped" ? "fill-warn" : "fill-text";
  return (
    <g>
      <rect x={x} y={y} width={w - 3} height={24} rx={3} className={box} strokeDasharray={tone === "dropped" ? "3 2" : undefined} />
      <text x={x + (w - 3) / 2} y={y + 16} textAnchor="middle" className={`${ink} text-[10px]`}>
        {text}
      </text>
    </g>
  );
}

export function StepsFigure() {
  const { on, grow } = useLoop(14);
  const tokens = spec.outputTokens;
  const px = W / tokens;
  const blocks = [{ width: 1, kept: 0 }, ...example.steps.map((s) => ({ width: s.kept + 1, kept: s.kept }))];
  const starts = blocks.map((_, i) => blocks.slice(0, i).reduce((sum, b) => sum + b.width, 0));
  return (
    <Frame label={`${fmt(tokens)} output tokens: plain decoding takes ${fmt(tokens)} steps, speculative decoding took ${fmt(blocks.length)}, keeping ${fmt(spec.kept)} of ${fmt(spec.drafted)} guessed tokens`}>
      <Label x={X} y={22} className="fill-text">{`the same ${fmt(tokens)} tokens, step by step`}</Label>
      <Label x={X} y={58}>{`plain decoding: ${fmt(tokens)} steps, one token each`}</Label>
      <motion.g {...grow(0.04, 0.3)}>
        {Array.from({ length: tokens }, (_, i) => (
          <rect key={i} x={X + i * px} y={66} width={Math.max(0.6, px - 0.8)} height={40} className="fill-text/60" />
        ))}
      </motion.g>
      <Label x={X} y={150}>{`with guesses: ${fmt(blocks.length)} steps, a block is one step`}</Label>
      <motion.g {...grow(0.34, 0.62)}>
        {blocks.map((b, i) => (
          <rect key={i} x={X + starts[i] * px} y={158} width={Math.max(0.6, b.width * px - 0.8)} height={40} className={b.kept > 0 ? "fill-accent" : "fill-text/60"} />
        ))}
      </motion.g>
      <motion.g {...on(0.66)}>
        <rect x={X} y={236} width={10} height={10} className="fill-accent" />
        <Label x={X + 18} y={245} className="fill-text">{`a step whose guesses the model agreed with: it wrote 2 to ${fmt(Math.max(...blocks.map((b) => b.width)))} tokens`}</Label>
        <rect x={X} y={258} width={10} height={10} className="fill-text/60" />
        <Label x={X + 18} y={267} className="fill-text">
          a step that wrote one token: no guess, or the first guess was wrong
        </Label>
      </motion.g>
      <motion.g {...on(0.74)}>
        <Label x={X} y={308} className="fill-accent">{`${fmt(spec.kept)} of ${fmt(spec.drafted)} guessed tokens kept (${fmt(spec.keptPct)}%): ${fmt(tokens / blocks.length, 2)} tokens per step`}</Label>
        <Label x={X} y={328}>the two rows end at the same place: the output is the same</Label>
      </motion.g>
      <Label x={X} y={384}>{`recorded ${spec.date}; width is tokens, not time (measured)`}</Label>
    </Frame>
  );
}

export function RowsFigure() {
  const { on, grow } = useLoop(14);
  const bars = [
    { label: "1 request in the step", ms: spec.step1, tone: "fill-text/70" },
    { label: `${spec.maxBatch} requests in the step`, ms: spec.step16, tone: "fill-accent" },
  ];
  const most = Math.max(spec.step16, ...rowTimes.map((r) => r.ms));
  const px = (W - 150 - 64) / most;
  const perRow = (spec.step16 - spec.step1) / (spec.maxBatch - 1);
  return (
    <Frame label={`A decode step for 1 request takes ${fmt(spec.step1, 2)} ms and for ${spec.maxBatch} requests ${fmt(spec.step16, 2)} ms; with one request and guesses in the other rows ${rowTimes.map((r) => `a step of ${r.rows} rows took ${fmt(r.ms, 2)} ms`).join(", ")}`}>
      <Label x={X} y={22} className="fill-text">
        what a step costs, by how many rows it carries
      </Label>
      {bars.map((b, i) => (
        <g key={b.label}>
          <Label x={X} y={56 + i * 40}>{b.label}</Label>
          <motion.rect x={X + 150} y={44 + i * 40} width={b.ms * px} height={18} rx={2} className={b.tone} {...grow(0.05 + i * 0.1, 0.3 + i * 0.1)} />
          <motion.g {...on(0.3 + i * 0.1)}>
            <Label x={X + 156 + b.ms * px} y={57 + i * 40} className="fill-text">{`${fmt(b.ms, 2)} ms`}</Label>
          </motion.g>
        </g>
      ))}
      <motion.g {...on(0.5)}>
        <Label x={X} y={140} className="fill-accent">{`${spec.maxBatch} rows for ${fmt(spec.rowCost, 2)}× the price of one: about ${fmt(perRow, 2)} ms per added row`}</Label>
      </motion.g>
      <Label x={X} y={182} className="fill-text">
        one request, the other rows used for guesses
      </Label>
      {rowTimes.slice(0, 6).map((r, i) => (
        <motion.g key={r.rows} {...on(0.56 + i * 0.05)}>
          <Label x={X} y={208 + i * 26}>{`${r.rows} ${r.rows === 1 ? "row" : "rows"} · ${fmt(r.steps)} ${r.steps === 1 ? "step" : "steps"}`}</Label>
          <rect x={X + 150} y={197 + i * 26} width={r.ms * px} height={14} rx={2} className={r.rows === 1 ? "fill-text/70" : "fill-accent/60"} />
          <Label x={X + 156 + r.ms * px} y={208 + i * 26} className="fill-text">{`${fmt(r.ms, 2)} ms`}</Label>
        </motion.g>
      ))}
      <Label x={X} y={384}>top: time per token at max batch 1 and 16; below: median step time, traced runs (measured)</Label>
    </Frame>
  );
}

export function GuessFigure() {
  const { on, move } = useLoop(14);
  const s = spec.bestStep;
  const before = 2;
  const found = all.slice(Math.max(0, s.match - before), s.match + s.ngram + s.draft.length);
  const lead = Math.min(before, s.match);
  const tail = all.slice(s.at - Math.max(s.ngram, 3) - 1, s.at);
  const cells = Math.max(found.length, tail.length + s.draft.length);
  const w = Math.min(56, W / cells);
  const yTop = 70;
  const yEnd = 230;
  const endX = X + tail.length * w;
  return (
    <Frame label={`Step ${s.index}: the last ${tokens(s.ngram)} of the context appeared ${fmt(s.at - s.ngram - s.match)} tokens earlier; the ${tokens(s.draft.length)} after that place are copied as the guess, and the model agreed with ${s.kept}`}>
      <Label x={X} y={22} className="fill-text">{`step ${s.index}: where did the last ${tokens(s.ngram)} appear before?`}</Label>
      <Label x={X} y={yTop - 10}>{`earlier in the context, ${fmt(s.at - s.ngram - s.match)} tokens back`}</Label>
      {found.map((t, i) => {
        const tone = i < lead ? "plain" : i < lead + s.ngram ? "pattern" : "draft";
        return (
          <motion.g key={i} {...on(tone === "draft" ? 0.3 : 0.18)}>
            <Chip x={X + i * w} y={yTop} w={w} text={short(t)} tone={tone} />
          </motion.g>
        );
      })}
      <motion.g {...on(0.3)}>
        <Label x={X + (lead + s.ngram) * w} y={yTop + 42} className="fill-warn">{`what followed: the guess, ${tokens(s.draft.length)}`}</Label>
      </motion.g>
      <Label x={X} y={yEnd - 10}>the end of the context</Label>
      {tail.map((t, i) => (
        <motion.g key={i} {...on(0.05)}>
          <Chip x={X + i * w} y={yEnd} w={w} text={short(t)} tone={i >= tail.length - s.ngram ? "pattern" : "plain"} />
        </motion.g>
      ))}
      <motion.g {...on(0.1)}>
        <Label x={X + (tail.length - s.ngram) * w} y={yEnd + 42} className="fill-accent">{`the last ${tokens(s.ngram)}: the pattern`}</Label>
      </motion.g>
      {s.draft.map((t, i) => (
        <motion.g key={i} {...move(0.38 + i * 0.03, 0.52 + i * 0.03, 0, 0)}>
          <Chip x={endX + i * w} y={yEnd} w={w} text={short(t)} tone="draft" />
        </motion.g>
      ))}
      <motion.g {...on(0.7)}>
        <Label x={X} y={330} className="fill-text">{`the guess is placed after the end, as rows ${s.draft.length > 1 ? `1 to ${s.draft.length}` : "1"} of the next step`}</Label>
        <Label x={X} y={350} className="fill-accent">{`the model agreed with ${s.kept} of ${s.draft.length}: the step wrote ${s.kept + 1} tokens`}</Label>
      </motion.g>
      <Label x={X} y={384}>{`recorded ${spec.date}; longest pattern first (${spec.maxNgram} tokens, then shorter), latest match first`}</Label>
    </Frame>
  );
}

export function CheckFigure() {
  const { on } = useLoop(14);
  const s = spec.missStep;
  const rows = 1 + s.draft.length;
  const pitch = Math.min(30, 290 / rows);
  const top = 62;
  const input = (i: number) => (i === 0 ? all[s.at - 1] : s.draft[i - 1]);
  return (
    <Frame label={`Step ${s.index} ran ${rows} rows of one sequence: the last real token and ${s.draft.length} guesses. The model agreed with ${s.kept} guesses, so ${s.kept + 1} tokens were kept and ${s.draft.length - s.kept} rows dropped`}>
      <Label x={X} y={22} className="fill-text">{`step ${s.index}: ${rows} rows of the same sequence in one step`}</Label>
      <Label x={X + 60} y={top - 8}>input</Label>
      <Label x={X + 190} y={top - 8}>the model samples</Label>
      <Label x={X + 320} y={top - 8}>against the next guess</Label>
      {Array.from({ length: rows }, (_, i) => {
        const y = top + i * pitch;
        const keptRow = i <= s.kept;
        const wrong = i === s.kept && s.kept < s.draft.length;
        return (
          <motion.g key={i} {...on(0.06 + i * 0.07)}>
            <Label x={X} y={y + 16}>{`row ${i}`}</Label>
            <Chip x={X + 60} y={y} w={90} text={short(input(i), 11)} tone={i === 0 ? "real" : keptRow ? "pattern" : "dropped"} />
            <text x={X + 165} y={y + 16} className="fill-muted text-[11px]">
              →
            </text>
            {keptRow ? <Chip x={X + 190} y={y} w={90} text={short(all[s.at + i], 11)} tone="kept" /> : <Chip x={X + 190} y={y} w={90} text="dropped" tone="dropped" />}
            {keptRow && (
              <Label x={X + 320} y={y + 16} className={wrong ? "fill-warn" : i < s.draft.length ? "fill-accent" : "fill-muted"}>
                {i >= s.draft.length ? "no guess left to check" : wrong ? `guess was “${short(s.draft[i], 11)}”: stop here` : "same as the guess: go on"}
              </Label>
            )}
            {!keptRow && (
              <Label x={X + 320} y={y + 16} className="fill-warn">
                computed from a wrong token
              </Label>
            )}
          </motion.g>
        );
      })}
      <motion.g {...on(0.75)}>
        <Label x={X} y={top + rows * pitch + 22} className="fill-accent">{`${s.kept + 1} tokens kept from one step: every one is the token the model sampled`}</Label>
      </motion.g>
      <Label x={X} y={384}>{`recorded ${spec.date} (measured); dropped rows' tokens are not recorded`}</Label>
    </Frame>
  );
}

export function TruncateFigure() {
  const { on } = useLoop(14);
  const s = spec.missStep;
  const block = 16;
  const first = Math.floor((s.at - 1) / block) * block;
  const last = s.at - 1 + s.draft.length;
  const blocks = Math.floor(last / block) - Math.floor((s.at - 1) / block) + 1;
  const cell = W / block;
  const keptEnd = s.at - 1 + s.kept;
  const freed = Math.floor(last / block) - Math.floor(keptEnd / block);
  return (
    <Frame label={`Step ${s.index} reserved ${1 + s.draft.length} cache slots, positions ${fmt(s.at - 1)} to ${fmt(last)}. ${s.kept + 1} stay; ${s.draft.length - s.kept} are given back${freed > 0 ? `, and ${freed} block returns to the free list` : ""}`}>
      <Label x={X} y={22} className="fill-text">{`step ${s.index}: the cache slots of one sequence, in blocks of ${block}`}</Label>
      {Array.from({ length: blocks }, (_, b) => (
        <g key={b}>
          <Label x={X} y={58 + b * 86}>{`block ${fmt(first / block + b)} of the sequence: positions ${fmt(first + b * block)} to ${fmt(first + (b + 1) * block - 1)}`}</Label>
          {Array.from({ length: block }, (_, i) => {
            const pos = first + b * block + i;
            const tone = pos < s.at - 1 ? "fill-text/25" : pos === s.at - 1 ? "fill-text/70" : pos <= keptEnd ? "fill-accent" : pos <= last ? "fill-none stroke-warn" : "fill-none stroke-border";
            return (
              <motion.g key={i} {...on(pos < s.at - 1 ? 0.04 : pos <= keptEnd ? 0.2 : 0.36)}>
                <rect x={X + i * cell} y={66 + b * 86} width={cell - 3} height={34} rx={2} className={tone} strokeDasharray={pos > keptEnd && pos <= last ? "3 2" : undefined} />
              </motion.g>
            );
          })}
        </g>
      ))}
      <motion.g {...on(0.5)}>
        <rect x={X} y={250} width={10} height={10} className="fill-text/70" />
        <Label x={X + 18} y={259} className="fill-text">{`row 0, the last real token, position ${fmt(s.at - 1)}`}</Label>
        <rect x={X} y={272} width={10} height={10} className="fill-accent" />
        <Label x={X + 18} y={281} className="fill-text">{`${s.kept} guessed ${s.kept === 1 ? "row" : "rows"} the model agreed with: their keys and values stay`}</Label>
        <rect x={X} y={294} width={10} height={10} className="fill-none stroke-warn" strokeDasharray="3 2" />
        <Label x={X + 18} y={303} className="fill-text">{`${s.draft.length - s.kept} rejected ${s.draft.length - s.kept === 1 ? "row" : "rows"}: the slots are given back`}</Label>
      </motion.g>
      <motion.g {...on(0.62)}>
        <Label x={X} y={338} className="fill-accent">{`length after the step: ${fmt(keptEnd + 1)} cached tokens${freed > 0 ? `; ${freed} whole block goes back to the free list` : "; the block stays, the next step overwrites the stale slots"}`}</Label>
      </motion.g>
      <Label x={X} y={384}>{`positions from the recorded answer of ${spec.date}; block size ${block} from include/engine/kv/cache_size.h`}</Label>
    </Frame>
  );
}

export function PayoffFigure() {
  const { on, grow } = useLoop(14);
  const most = Math.max(...repeatRows.map((r) => r.ms), ...chatRows.map((r) => r.ms));
  const left = X + 96;
  const px = (W - 96 - 236) / most;
  const pitch = Math.min(46, 276 / repeatRows.length);
  return (
    <Frame label={`Time per token by the most guesses per step. Repeat prompts: ${repeatRows.map((r) => `${r.k} guesses ${fmt(r.ms, 2)} ms`).join(", ")}. Chat prompts: ${chatRows.map((r) => `${r.k} guesses ${fmt(r.ms, 2)} ms`).join(", ")}`}>
      <Label x={X} y={22} className="fill-text">
        time per token, by the most guesses a step may carry
      </Label>
      <rect x={left} y={32} width={10} height={10} className="fill-accent" />
      <Label x={left + 16} y={41}>answers that repeat their prompt</Label>
      <rect x={left + 220} y={32} width={10} height={10} className="fill-baseline" />
      <Label x={left + 236} y={41}>chat answers</Label>
      {repeatRows.map((r, i) => {
        const chat = chatRows.find((c) => c.k === r.k);
        const y = 62 + i * pitch;
        return (
          <g key={r.k}>
            <Label x={X} y={y + 14} className="fill-text">
              {r.k === 0 ? "no guesses" : `up to ${r.k}`}
            </Label>
            <motion.rect x={left} y={y} width={r.ms * px} height={12} rx={2} className="fill-accent" {...grow(0.05 + i * 0.07, 0.3 + i * 0.07)} />
            <motion.g {...on(0.3 + i * 0.07)}>
              <Label x={left + r.ms * px + 6} y={y + 10} className="fill-text">{`${fmt(r.ms, 2)} ms · ${fmt(r.tokensPerStep, 2)} tokens per step${r.k > 0 ? ` · ${fmt(r.acceptedPct)}% kept` : ""}`}</Label>
            </motion.g>
            {chat && (
              <>
                <motion.rect x={left} y={y + 15} width={chat.ms * px} height={12} rx={2} className="fill-baseline" {...grow(0.08 + i * 0.07, 0.33 + i * 0.07)} />
                <motion.g {...on(0.33 + i * 0.07)}>
                  <Label x={left + chat.ms * px + 6} y={y + 25}>{`${fmt(chat.ms, 2)} ms${chat.k > 0 ? ` · ${fmt(chat.acceptedPct)}% kept` : ""}`}</Label>
                </motion.g>
              </>
            )}
          </g>
        );
      })}
      <motion.g {...on(0.8)}>
        <Label x={X} y={360} className="fill-accent">{`${fmt(spec.best.speedup, 2)}× the tokens per second with up to ${spec.best.k} guesses, when there is something to copy`}</Label>
      </motion.g>
      <Label x={X} y={384}>docs/results.md, Speculative decoding: one request at a time, median of the runs (measured)</Label>
    </Frame>
  );
}
