"use client";

import { motion } from "motion/react";
import { chapterNumber } from "@/content/book-order";
import { kvReplay, modelShape } from "@/lib/data";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";
import { StickyStory, type StoryStep } from "../sticky-story";

const shape = modelShape();
const replay = kvReplay();
const headDim = shape.hidden / shape.heads;
const valueBytes = 2;

export const kvAbout = {
  block: replay.meta.block_size,
  layers: shape.layers,
  rowBytes: shape.kv_heads * headDim * valueBytes,
  tokenBytes: 2 * shape.layers * shape.kv_heads * headDim * valueBytes,
  pool: [1, 2, 3, 4, 5, 6, 7, 8],
  others: new Map([
    [1, 16],
    [2, 11],
    [4, 16],
    [6, 7],
  ]),
  tokens: 37,
};

export const lowestFree = kvAbout.pool.filter((id) => !kvAbout.others.has(id));

const fmt = (v: number) => v.toLocaleString("en-US");

export const pagedKvAbout: StoryStep = {
  id: "about",
  title: "what a paged KV cache is",
  body: (
    <>
      <p>
        A transformer writes one token at a time, and each new token has to look back at every token before it. In each of the {kvAbout.layers} layers, attention turns every token into three
        vectors: a query, a key and a value. The new token&apos;s query is compared with the key of every earlier token, and the scores decide how much of each earlier token&apos;s value goes into
        its output.
      </p>
      <p>
        The keys and values of earlier tokens never change, so the engine keeps them instead of computing them again on every step. That store is the KV cache, K for keys and V for values. Each new
        token is computed once, adds one row of keys and one row of values in every layer, and from then on is only read. For TinyLlama a row is the KV heads of one layer in
        bfloat16, so every token a request sees adds <span className="text-accent">{`${fmt(kvAbout.tokenBytes)} bytes`}</span> over all the layers.
      </p>
      <p>
        Paged means the cache is not one long array per request. Its memory is cut into blocks of {kvAbout.block} tokens, handed out one at a time as a request grows, wherever one is free, and each
        request keeps a block table that says which block holds each run of its tokens, the first block&apos;s worth, the next and so on, the way an operating system finds a
        program&apos;s memory through page tables. That lets many requests share one pool. How the engine decides which of them run together is chapter {chapterNumber("serving")}.
      </p>
    </>
  ),
};

export const pagedKvAboutCaption = `illustrative: one request in one of ${kvAbout.layers} layers and a pool of ${kvAbout.pool.length} blocks; the engine's blocks are ${kvAbout.block} tokens, as here, and its pool holds ${fmt(replay.meta.kv_blocks)} of them; a row of keys is ${shape.kv_heads} KV heads of ${headDim} numbers, the block table maps tokens 0 to ${kvAbout.block - 1}, ${kvAbout.block} to ${2 * kvAbout.block - 1} and so on`;

const B = kvAbout.block;
const used = Math.ceil(kvAbout.tokens / B);
const slot = 9;
const lane = 14;
const logicalLeft = 40;
const logicalW = B * slot;
const logicalGap = 22;
const logicalTop = 92;
const poolTop = 250;
const poolLeft = 20;
const poolW = 60;
const poolGap = 6;
const poolSlot = poolW / B;
const start = 0.06;
const span = 0.24;
const windowOf = (j: number) => {
  const a = start + j * span;
  const fill = Math.min(B, kvAbout.tokens - j * B) / B;
  return [a, a + span * fill] as const;
};
const logicalX = (j: number) => logicalLeft + j * (logicalW + logicalGap);
const poolX = (i: number) => poolLeft + i * (poolW + poolGap);

export function PagedKvAboutFigure() {
  const { on, grow } = useLoop(16);
  const ours = lowestFree.slice(0, used);
  return (
    <Frame
      label={`One request's keys and values, ${kvAbout.tokens} tokens in order, kept in blocks of ${B}. Its block table maps logical blocks 0 to ${used - 1} to physical blocks ${ours.join(", ")} of a shared pool, whichever were free.`}
    >
      <Label x={20} y={24}>{"KV cache: one row of keys (K) and one of values (V) for every token, in each layer"}</Label>
      <Label x={20} y={44} className="fill-text">{"this request's tokens, in order"}</Label>
      {Array.from({ length: used }, (_, j) => {
        const [a, b] = windowOf(j);
        const x = logicalX(j);
        const fill = Math.min(B, kvAbout.tokens - j * B);
        return (
          <g key={`l${j}`}>
            <Label x={x} y={logicalTop - 22}>{`logical ${j}: tokens ${j * B} to ${j * B + B - 1}`}</Label>
            {[0, 1].map((k) => (
              <g key={k}>
                <rect x={x} y={logicalTop + k * (lane + 3)} width={logicalW} height={lane} rx={2} className="fill-surface stroke-border" strokeWidth={0.75} />
                <motion.rect x={x} y={logicalTop + k * (lane + 3)} width={fill * slot} height={lane} rx={2} className="fill-accent" {...grow(a, b)} />
              </g>
            ))}
          </g>
        );
      })}
      <Label x={20} y={logicalTop + 11}>K</Label>
      <Label x={20} y={logicalTop + lane + 14}>V</Label>
      <Label x={20} y={164}>block table</Label>
      {ours.map((id, j) => (
        <motion.g key={`t${j}`} {...on(windowOf(j)[0])}>
          <Label x={logicalX(j)} y={182} className="fill-accent">{`logical ${j} → block ${id}`}</Label>
          <line x1={logicalX(j) + logicalW / 2} y1={190} x2={poolX(id - 1) + poolW / 2} y2={poolTop - 6} className="stroke-accent" strokeWidth={1.25} strokeDasharray="3 3" />
        </motion.g>
      ))}
      {kvAbout.pool.map((id, i) => {
        const x = poolX(i);
        const other = kvAbout.others.get(id);
        const j = ours.indexOf(id);
        return (
          <g key={`p${id}`}>
            {[0, 1].map((k) => (
              <g key={k}>
                <rect x={x} y={poolTop + k * (lane + 3)} width={poolW} height={lane} rx={2} className="fill-surface stroke-border" strokeWidth={0.75} />
                {other !== undefined && <rect x={x} y={poolTop + k * (lane + 3)} width={other * poolSlot} height={lane} rx={2} className="fill-baseline" />}
                {j >= 0 && (
                  <motion.rect
                    x={x}
                    y={poolTop + k * (lane + 3)}
                    width={Math.min(B, kvAbout.tokens - j * B) * poolSlot}
                    height={lane}
                    rx={2}
                    className="fill-accent"
                    {...grow(windowOf(j)[0], windowOf(j)[1])}
                  />
                )}
              </g>
            ))}
            <Label x={x} y={poolTop + 2 * lane + 18}>{`block ${id}`}</Label>
            {j >= 0 ? (
              <>
                <motion.g {...on(0, windowOf(j)[0], false)}>
                  <Label x={x} y={poolTop + 2 * lane + 32} className="fill-text">
                    free
                  </Label>
                </motion.g>
                <motion.g {...on(windowOf(j)[0])}>
                  <Label x={x} y={poolTop + 2 * lane + 32} className="fill-accent">{`logical ${j}`}</Label>
                </motion.g>
              </>
            ) : (
              <Label x={x} y={poolTop + 2 * lane + 32} className={other !== undefined ? "fill-muted" : "fill-text"}>
                {other !== undefined ? "other" : "free"}
              </Label>
            )}
          </g>
        );
      })}
      <Label x={20} y={334}>{`the shared pool: physical blocks of ${B} slots, grey ones held by other requests`}</Label>
      <motion.g {...on(windowOf(used - 1)[1])}>
        <Label x={20} y={354}>{`${kvAbout.tokens} tokens in ${used} blocks; a new block is taken only when the last one is full`}</Label>
      </motion.g>
      <Label x={20} y={374}>{`each token: ${fmt(kvAbout.rowBytes)} bytes of K and ${fmt(kvAbout.rowBytes)} of V per layer, ${fmt(kvAbout.tokenBytes)} bytes over ${kvAbout.layers} layers`}</Label>
    </Frame>
  );
}

export function PagedKvAboutStory() {
  return <StickyStory kicker="the idea" steps={[pagedKvAbout]} caption={() => pagedKvAboutCaption} figure={() => <PagedKvAboutFigure />} />;
}
