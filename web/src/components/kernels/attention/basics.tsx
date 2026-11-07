"use client";

import { motion } from "motion/react";
import { kvReplay, modelShape } from "@/lib/data";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";
import { StickyStory, type StoryStep } from "../sticky-story";

const m = modelShape();
const pool = kvReplay().meta;
const headDim = m.hidden / m.heads;
const kvDim = m.kv_heads * headDim;
const qkvRow = m.hidden + 2 * kvDim;
const group = m.heads / m.kv_heads;
const kvLayer = 2 * kvDim * 2;
const kvToken = m.layers * kvLayer;
const mhaToken = kvToken * group;
const poolTokens = pool.kv_blocks * pool.block_size;
const poolBytes = poolTokens * kvToken;
const mhaPoolTokens = Math.floor(poolBytes / mhaToken);
const scale = Math.sqrt(headDim);

const int = (v: number) => Math.round(v).toLocaleString("en-US");

const words = ["the", "cat", "sat", "on", "the", "mat"];
const scores = [0.4, 2.1, 0.9, 0.2, 0.6, 1.5];
const exps = scores.map((s) => Math.exp(s - Math.max(...scores)));
const total = exps.reduce((a, b) => a + b, 0);
const weights = exps.map((e) => e / total);
const top = weights.indexOf(Math.max(...weights));
const starts = weights.map((_, i) => weights.slice(0, i).reduce((a, b) => a + b, 0));

export const basicsNumbers = { headDim, kvDim, qkvRow, group, kvLayer, kvToken, mhaToken, poolTokens, mhaPoolTokens };

const steps: StoryStep[] = [
  {
    id: "basics-qkv",
    title: "three vectors per token",
    body: (
      <>
        <p>
          Every token reaches attention as a vector of {int(m.hidden)} values. One matrix multiply turns it into three vectors: a <span className="text-text">query</span>, what this token is looking for; a <span className="text-text">key</span>, what it offers to tokens that look at it; and a <span className="text-text">value</span>, what it passes on when it is picked.
        </p>
        <p>
          In TinyLlama the query has {int(m.hidden)} values and the key and the value <span className="text-accent">{int(kvDim)}</span> each, {int(qkvRow)} in all. That is the QKV projection of the GEMM chapter, and the key and value are what the KV cache stores.
        </p>
      </>
    ),
  },
  {
    id: "basics-scores",
    title: "score, weigh, sum",
    body: (
      <>
        <p>
          The newest token compares its query with the key of every token so far: one dot product each, divided by √{headDim} = {scale} so the scores do not grow with the vector length. A high score means a good match.
        </p>
        <p>
          Softmax turns the scores into weights that are all positive and add up to 1, so the best match gets most of the weight: here &quot;{words[top]}&quot; gets <span className="text-accent">{Math.round(weights[top] * 100)}%</span>. The output is the values added up with those weights, one vector of {headDim} numbers per head.
        </p>
        <p>That is the whole formula, out = softmax(q · Kᵀ / √d) · V. Everything later in this chapter is about reading K and V fast.</p>
      </>
    ),
  },
  {
    id: "basics-past",
    title: "only the past",
    body: (
      <>
        <p>
          A token may only look at itself and the tokens before it; the future is masked out. In prefill every prompt token does this at once, a triangle of scores: 8 tokens give 36.
        </p>
        <p>
          In decode only the newest token is new, so a step adds one row to the triangle: <span className="text-accent">one query</span> against every key so far. Old queries are never used again, but every old key and value is read again at every step. That is why the cache keeps K and V, and not Q.
        </p>
      </>
    ),
  },
  {
    id: "basics-heads",
    title: "heads, and heads that share",
    body: (
      <>
        <p>
          Attention runs as {m.heads} heads side by side. The query is cut into {m.heads} pieces of {headDim} values, each head scores and sums on its own, looking for its own pattern, and the {m.heads} outputs are joined back into {int(m.hidden)} values.
        </p>
        <p>
          TinyLlama uses grouped-query attention: it has only {m.kv_heads} key heads and {m.kv_heads} value heads, and each serves {group} query heads. So a token caches {int(kvLayer)} bytes per layer, <span className="text-accent">{int(kvToken)}</span> over {m.layers} layers, where one key head per query head would need {int(mhaToken)}.
        </p>
        <p>The same memory then holds {group} times the tokens: the recorded run&apos;s pool fits {int(poolTokens)} tokens, and would fit {int(mhaPoolTokens)} without sharing.</p>
      </>
    ),
  },
];

export const basicsCaptions = [
  `sizes from config.json: ${int(m.hidden)} hidden values, ${m.heads} query heads and ${m.kv_heads} key and value heads of ${headDim}; bars drawn to scale`,
  "illustrative: six tokens and made-up scores; the softmax and the sum are computed from them",
  "illustrative: an 8-token prompt, then one decode step",
  `${int(kvToken)} bytes = ${m.layers} layers × K and V × ${m.kv_heads} heads × ${headDim} values × 2 bytes; the pool is the ${int(pool.kv_blocks)} blocks of ${pool.block_size} tokens of the recorded chat run`,
];

export const basicsInstructions = [
  "Watch one token's vector split into a long query and a short key and value.",
  "Watch the scores fill, then the softmax hand most of the weight to one token.",
  "Watch prefill fill the triangle, then decode add a single row.",
  "Follow each group of query heads to the one key and value head it shares.",
];

const ink = "fill-text";

function QkvFigure() {
  const { on, grow } = useLoop(11);
  const x0 = 40;
  const px = 470 / m.hidden;
  const q = m.hidden * px;
  const kv = kvDim * px;
  return (
    <Frame label={`One token's ${m.hidden} values become a query of ${m.hidden}, a key of ${kvDim} and a value of ${kvDim}`}>
      <Label x={20} y={30} className={ink}>
        one token, after RMSNorm
      </Label>
      <rect x={x0} y={44} width={q} height={22} rx={4} className="fill-baseline/60" />
      <Label x={x0 + q} y={84} anchor="end">{`${int(m.hidden)} values`}</Label>
      <motion.g {...on(0.08, 0.94)}>
        <rect x={x0 + 140} y={100} width={190} height={46} rx={6} className="fill-none stroke-text" strokeWidth={1.25} />
        <Label x={x0 + 235} y={120} anchor="middle" className={ink}>
          × QKV weights
        </Label>
        <Label x={x0 + 235} y={137} anchor="middle">{`[${int(qkvRow)} × ${int(m.hidden)}]`}</Label>
      </motion.g>
      <Label x={20} y={186} className={ink}>
        query: what this token looks for
      </Label>
      <motion.rect x={x0} y={196} width={q} height={22} rx={4} className="fill-accent" {...grow(0.2, 0.42)} />
      <motion.g {...on(0.42, 0.94)}>
        {Array.from({ length: m.heads - 1 }, (_, i) => (
          <line key={i} x1={x0 + (i + 1) * headDim * px} x2={x0 + (i + 1) * headDim * px} y1={196} y2={218} className="stroke-bg" strokeWidth={1} />
        ))}
        <Label x={x0 + q} y={236} anchor="end">{`${int(m.hidden)} values = ${m.heads} heads × ${headDim}`}</Label>
      </motion.g>
      <Label x={20} y={270} className={ink}>
        key: what it offers
      </Label>
      <motion.rect x={x0} y={280} width={kv} height={22} rx={4} className="fill-warn" {...grow(0.45, 0.55)} />
      <Label x={x0 + kv + 10} y={296}>{`${int(kvDim)} values = ${m.kv_heads} heads × ${headDim}`}</Label>
      <Label x={20} y={330} className={ink}>
        value: what it passes on
      </Label>
      <motion.rect x={x0} y={340} width={kv} height={22} rx={4} className="fill-warn" {...grow(0.55, 0.65)} />
      <Label x={x0 + kv + 10} y={356}>{`${int(kvDim)} values = ${m.kv_heads} heads × ${headDim}`}</Label>
      <motion.g {...on(0.68, 0.94)}>
        <Label x={300} y={326} className="fill-warn">
          K and V go into the KV cache
        </Label>
        <Label x={300} y={386}>
          Q is used once and dropped
        </Label>
      </motion.g>
    </Frame>
  );
}

function ScoresFigure() {
  const { on, grow } = useLoop(13);
  const y0 = 64;
  const row = 40;
  const sx = 190;
  const wx = 330;
  const sMax = Math.max(...scores);
  return (
    <Frame label={`Six tokens scored against the newest query; softmax gives ${words[top]} ${Math.round(weights[top] * 100)}% of the weight; the output is the weighted sum of the values`}>
      <Label x={20} y={28} className={ink}>
        the newest token&apos;s query against every key so far
      </Label>
      <Label x={sx} y={50}>
        score q · k / √d
      </Label>
      <Label x={wx} y={50}>
        weight after softmax
      </Label>
      {words.map((w, i) => {
        const y = y0 + i * row;
        return (
          <g key={i}>
            <Label x={40} y={y + 15} className={ink}>
              {w}
            </Label>
            <circle cx={120} cy={y + 11} r={6} className="fill-warn" />
            <Label x={132} y={y + 15}>
              k
            </Label>
            <rect x={sx} y={y + 2} width={(scores[i] / sMax) * 110} height={18} rx={3} className="fill-baseline/20" />
            <motion.rect x={sx} y={y + 2} width={(scores[i] / sMax) * 110} height={18} rx={3} className="fill-text" {...grow(0.04 + i * 0.05, 0.08 + i * 0.05)} />
            <motion.g {...on(0.1 + i * 0.05, 0.94)}>
              <Label x={sx + (scores[i] / sMax) * 110 + 6} y={y + 15}>
                {scores[i].toFixed(1)}
              </Label>
            </motion.g>
            <motion.rect x={wx} y={y + 2} width={weights[i] * 300} height={18} rx={3} className={i === top ? "fill-accent" : "fill-accent/50"} {...grow(0.42, 0.56)} />
            <motion.g {...on(0.56, 0.94)}>
              <Label x={wx + weights[i] * 300 + 6} y={y + 15} className={i === top ? "fill-accent" : "fill-muted"}>{`${Math.round(weights[i] * 100)}%`}</Label>
            </motion.g>
          </g>
        );
      })}
      <motion.g {...on(0.62, 0.94)}>
        <Label x={40} y={322} className={ink}>
          output = the values, added up with those weights
        </Label>
        {weights.map((w, i) => (
          <rect key={i} x={40 + starts[i] * 480} y={334} width={Math.max(1, w * 480 - 2)} height={22} rx={3} className={i === top ? "fill-accent" : "fill-accent/40"} />
        ))}
        <Label x={40} y={378}>{`share of each token's value in the output; ${headDim} numbers per head`}</Label>
      </motion.g>
    </Frame>
  );
}

function PastFigure() {
  const { on } = useLoop(13);
  const n = 8;
  const c = 20;
  const g = 4;
  const x0 = 60;
  const x1 = 330;
  const y0 = 82;
  return (
    <Frame label="Prefill computes a triangle of 36 scores for 8 tokens; a decode step adds one row: one query against 9 keys">
      <Label x={20} y={28} className={ink}>
        rows are queries, columns are keys; a token never looks ahead
      </Label>
      <Label x={x0} y={66} className={ink}>
        prefill: 8 prompt tokens at once
      </Label>
      {Array.from({ length: n }, (_, r) => (
        <motion.g key={r} {...on(0.04, 0.94)}>
          {Array.from({ length: n }, (_, k) => (
            <rect key={k} x={x0 + k * (c + g)} y={y0 + r * (c + g)} width={c} height={c} rx={3} className={k <= r ? "fill-accent" : "fill-baseline/15"} />
          ))}
        </motion.g>
      ))}
      <motion.g {...on(0.2, 0.94)}>
        <Label x={x0} y={y0 + n * (c + g) + 20}>
          36 scores; the grey half is masked out
        </Label>
      </motion.g>
      <motion.g {...on(0.42, 0.94)}>
        <Label x={x1} y={66} className={ink}>
          decode: the 9th token
        </Label>
        {Array.from({ length: n }, (_, r) => (
          <g key={r}>
            {Array.from({ length: r + 1 }, (_, k) => (
              <rect key={k} x={x1 + k * (c + g)} y={y0 + r * (c + g)} width={c} height={c} rx={3} className="fill-baseline/30" />
            ))}
          </g>
        ))}
      </motion.g>
      <motion.g {...on(0.55, 0.94)}>
        {Array.from({ length: n + 1 }, (_, k) => (
          <rect key={k} x={x1 + k * (c + g)} y={y0 + n * (c + g)} width={c} height={c} rx={3} className="fill-accent" />
        ))}
        <Label x={x1} y={y0 + (n + 1) * (c + g) + 16}>
          one new row: 1 query, 9 keys
        </Label>
      </motion.g>
      <motion.g {...on(0.68, 0.94)}>
        <Label x={20} y={360} className={ink}>
          old rows are never computed again, so old queries can go;
        </Label>
        <Label x={20} y={380} className={ink}>
          every column is read again each step, so keys and values stay cached
        </Label>
      </motion.g>
    </Frame>
  );
}

function HeadsFigure() {
  const { on, grow } = useLoop(13);
  const x0 = 30;
  const w = 13;
  const gap = 2;
  const groupGap = 12;
  const qx = (i: number) => x0 + i * (w + gap) + Math.floor(i / group) * groupGap;
  const groupWidth = group * (w + gap) - gap;
  const kvx = (k: number) => qx(k * group) + groupWidth / 2;
  const bar = 330 / mhaToken;
  return (
    <Frame label={`${m.heads} query heads in ${m.kv_heads} groups of ${group}, each group sharing one key head and one value head; ${int(kvToken)} bytes of cache per token against ${int(mhaToken)} without sharing`}>
      <Label x={20} y={28} className={ink}>{`${m.heads} query heads of ${headDim} values, ${m.kv_heads} key and value heads`}</Label>
      {Array.from({ length: m.heads }, (_, i) => (
        <motion.rect key={i} x={qx(i)} y={44} width={w} height={44} rx={2} className={Math.floor(i / group) % 2 ? "fill-accent/60" : "fill-accent"} {...on(0.03 + (i / m.heads) * 0.2, 0.94)} />
      ))}
      {Array.from({ length: m.kv_heads }, (_, k) => (
        <motion.g key={k} {...on(0.28 + k * 0.05, 0.94)}>
          {Array.from({ length: group }, (_, j) => (
            <line key={j} x1={qx(k * group + j) + w / 2} y1={90} x2={kvx(k)} y2={140} className="stroke-warn/60" strokeWidth={1} />
          ))}
          <rect x={kvx(k) - 22} y={140} width={20} height={34} rx={3} className="fill-warn" />
          <rect x={kvx(k) + 2} y={140} width={20} height={34} rx={3} className="fill-warn/60" />
          <Label x={kvx(k)} y={190} anchor="middle">{`K, V head ${k + 1}`}</Label>
          <Label x={kvx(k)} y={206} anchor="middle">{`serves ${group} query heads`}</Label>
        </motion.g>
      ))}
      <motion.g {...on(0.48, 0.94)}>
        <Label x={20} y={236} className={ink}>
          KV cache per token, all {m.layers} layers
        </Label>
      </motion.g>
      <Label x={20} y={268}>
        shared ({m.kv_heads} heads)
      </Label>
      <motion.rect x={180} y={254} width={kvToken * bar} height={20} rx={3} className="fill-accent" {...grow(0.52, 0.6)} />
      <motion.g {...on(0.6, 0.94)}>
        <Label x={186 + kvToken * bar} y={268} className="fill-accent">{`${int(kvToken)} B`}</Label>
      </motion.g>
      <Label x={20} y={300}>
        one per query head
      </Label>
      <motion.rect x={180} y={286} width={mhaToken * bar} height={20} rx={3} className="fill-baseline/60" {...grow(0.62, 0.76)} />
      <motion.g {...on(0.76, 0.94)}>
        <Label x={176 + mhaToken * bar} y={300} anchor="end" className="fill-bg">{`${int(mhaToken)} B`}</Label>
      </motion.g>
      <motion.g {...on(0.8, 0.94)}>
        <Label x={20} y={350} className={ink}>{`the same pool holds ${int(poolTokens)} tokens shared, ${int(mhaPoolTokens)} without`}</Label>
        <Label x={20} y={372}>{`${group} times the conversations in the same memory`}</Label>
      </motion.g>
    </Frame>
  );
}

const figures = [QkvFigure, ScoresFigure, PastFigure, HeadsFigure];

export function BasicsFigure({ index }: { index: number }) {
  const Figure = figures[index] ?? QkvFigure;
  return <Figure />;
}

export const attentionBasics = steps;

export function AttentionBasicsStory() {
  return <StickyStory steps={steps} kicker="attention basics" caption={(i) => basicsCaptions[i]} figure={(i) => <BasicsFigure index={i} />} />;
}
