"use client";

import { motion } from "motion/react";
import { pairs, pcFmt as fmt, prefix } from "@/content/kernels/prefix-caching";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

const X = 20;
const W = 520;
const p = prefix.largest;

function Cell({ x, y, w = 34, h = 22, tone, text }: { x: number; y: number; w?: number; h?: number; tone: "computed" | "cached" | "blank" | "warn" | "plain"; text?: string }) {
  const box = { computed: "fill-accent", cached: "fill-text/35", blank: "fill-none stroke-border", warn: "fill-warn/70", plain: "fill-text/10 stroke-border" }[tone];
  const ink = tone === "computed" || tone === "warn" ? "fill-bg" : "fill-text";
  return (
    <g>
      <rect x={x} y={y} width={w - 3} height={h} rx={3} className={box} strokeDasharray={tone === "blank" ? "3 2" : undefined} />
      {text && (
        <text x={x + (w - 3) / 2} y={y + h / 2 + 4} textAnchor="middle" className={`${ink} text-[10px]`}>
          {text}
        </text>
      )}
    </g>
  );
}

export function ShareFigure() {
  const { on, grow, move } = useLoop(16);
  const shared = 4;
  const cw = 52;
  const scale = 300 / p.quiet.off.later_ttft_ms;
  return (
    <Frame label={`Two requests start with the same ${shared} blocks. The first prefills all of them; the second points at the same keys and values and prefills only its own block. At a ${fmt(p.shared)}-token shared prefix the time to the first token fell from ${fmt(p.quiet.off.later_ttft_ms)} to ${fmt(p.quiet.on.later_ttft_ms)} ms (measured)`}>
      <Label x={X} y={22} className="fill-text">two requests that start the same way</Label>
      <Label x={X} y={56}>request 1: the shared start, then its own question</Label>
      <motion.g {...grow(0.04, 0.24)}>
        {Array.from({ length: shared }, (_, i) => (
          <Cell key={i} x={X + i * cw} y={64} w={cw} tone="computed" text={`block ${i + 1}`} />
        ))}
        <Cell x={X + shared * cw} y={64} w={cw} tone="warn" text="own" />
      </motion.g>
      <motion.g {...on(0.26)}>
        <Label x={X + (shared + 1) * cw + 10} y={80} className="fill-accent">all prefilled, then kept</Label>
      </motion.g>
      <Label x={X} y={130}>request 2: the same start, another question</Label>
      <motion.g {...move(0.32, 0.5, 0, 74, true, 0.94)}>
        {Array.from({ length: shared }, (_, i) => (
          <Cell key={i} x={X + i * cw} y={64} w={cw} tone="cached" text={`block ${i + 1}`} />
        ))}
      </motion.g>
      <motion.g {...on(0.54)}>
        <motion.g {...grow(0.56, 0.66)}>
          <Cell x={X + shared * cw} y={138} w={cw} tone="warn" text="own" />
        </motion.g>
        <Label x={X + (shared + 1) * cw + 10} y={154} className="fill-text">only its own block is prefilled</Label>
      </motion.g>
      <motion.g {...on(0.66)}>
        <Label x={X} y={214} className="fill-text">{`measured at a ${fmt(p.shared)}-token shared start, ${prefix.own} tokens of its own`}</Label>
        <Label x={X} y={244}>cache off</Label>
        <rect x={X + 80} y={232} width={p.quiet.off.later_ttft_ms * scale} height={16} rx={2} className="fill-text/60" />
        <Label x={X + 86 + p.quiet.off.later_ttft_ms * scale} y={244} className="fill-text">{`${fmt(p.quiet.off.later_ttft_ms)} ms to the first token`}</Label>
        <Label x={X} y={274}>cache on</Label>
        <motion.rect x={X + 80} y={262} width={p.quiet.on.later_ttft_ms * scale} height={16} rx={2} className="fill-accent" {...grow(0.7, 0.76)} />
        <Label x={X + 86 + p.quiet.on.later_ttft_ms * scale} y={274} className="fill-accent">{`${fmt(p.quiet.on.later_ttft_ms)} ms`}</Label>
      </motion.g>
      <rect x={X} y={330} width={10} height={10} rx={1} className="fill-accent" />
      <Label x={X + 16} y={339}>prefilled in this request</Label>
      <rect x={X + 170} y={330} width={10} height={10} rx={1} className="fill-text/35" />
      <Label x={X + 186} y={339}>reused from the cache</Label>
      <rect x={X + 320} y={330} width={10} height={10} rx={1} className="fill-warn/70" />
      <Label x={X + 336} y={339}>the request&apos;s own tokens</Label>
      <Label x={X} y={380}>a cell is one block of 16 tokens; block counts illustrative, times measured</Label>
    </Frame>
  );
}

const words = ["You", "are", "a", "kind", "tutor", ".", "What"];
const other = ["You", "are", "a", "kind", "tutor", ".", "Name"];

export function SameKeysFigure() {
  const { on, grow } = useLoop(14);
  const cw = 64;
  const split = words.findIndex((w, i) => w !== other[i]);
  return (
    <Frame label={`Two prompts share their first ${split} tokens. A token's keys and values are computed from it and the tokens before it, so the first ${split} are the same in both prompts and only the tokens from the first difference on are not`}>
      <Label x={X} y={22} className="fill-text">a token&apos;s keys and values look only back</Label>
      <Label x={X} y={56}>prompt A</Label>
      <Label x={X} y={166}>prompt B</Label>
      {words.map((w, i) => (
        <g key={i}>
          <Cell x={X + i * cw} y={64} w={cw} tone="plain" text={w} />
          <Cell x={X + i * cw} y={174} w={cw} tone="plain" text={other[i]} />
          <motion.g {...on(0.08 + i * 0.07)}>
            <Cell x={X + i * cw} y={94} w={cw} h={16} tone={i < split ? "computed" : "warn"} text="K, V" />
            <Cell x={X + i * cw} y={204} w={cw} h={16} tone={i < split ? "computed" : "warn"} text="K, V" />
            <text x={X + i * cw + (cw - 3) / 2} y={146} textAnchor="middle" className={`${i < split ? "fill-accent" : "fill-warn"} text-[16px]`}>
              {i < split ? "=" : "≠"}
            </text>
          </motion.g>
        </g>
      ))}
      <motion.path d={`M${X + 4 * cw + 30} 92 C ${X + 4 * cw + 30} 76, ${X + 30} 76, ${X + 30} 88`} fill="none" strokeWidth={1.2} className="stroke-accent" {...on(0.4)} />
      <motion.g {...on(0.4)}>
        <Label x={X + 2 * cw} y={250} className="fill-text">{`"tutor" reads its own token and the ${4} before it`}</Label>
      </motion.g>
      <motion.g {...on(0.6)}>
        <motion.g {...grow(0.6, 0.7)}>
          <rect x={X} y={278} width={split * cw - 3} height={6} rx={2} className="fill-accent" />
        </motion.g>
        <Label x={X} y={304} className="fill-accent">{`the first ${split} tokens: the same keys and values in both prompts`}</Label>
        <Label x={X} y={324}>from the first different token on, every key and value differs</Label>
      </motion.g>
      <Label x={X} y={380}>illustrative words; in the engine a token is a SentencePiece piece and a block holds 16</Label>
    </Frame>
  );
}

export function ChainFigure() {
  const { on, grow } = useLoop(16);
  const bw = 120;
  const blocks = ["t0 … t15", "t16 … t31", "t32 … t47", "t48 … t63"];
  return (
    <Frame label="A block's key is every token from the start of the sequence to the end of that block. When one token in block 2 differs, the keys of block 2 and of every later block differ, although block 3 holds the same tokens">
      <Label x={X} y={22} className="fill-text">a block&apos;s key is every token up to its end</Label>
      {blocks.map((b, i) => (
        <g key={i}>
          <Cell x={X + i * bw} y={44} w={bw} tone="plain" text={`block ${i}: ${b}`} />
          <motion.g {...on(0.06 + i * 0.1)}>
            <motion.g {...grow(0.06 + i * 0.1, 0.12 + i * 0.1)}>
              <rect x={X} y={80 + i * 22} width={(i + 1) * bw - 3} height={12} rx={2} className="fill-accent/40" />
            </motion.g>
            <Label x={X + (i + 1) * bw - 10} y={90 + i * 22} anchor="end" className="fill-text">{`key of block ${i}`}</Label>
          </motion.g>
        </g>
      ))}
      <motion.g {...on(0.5)}>
        <Label x={X} y={200} className="fill-text">another prompt: one token of block 2 differs</Label>
        {blocks.map((b, i) => (
          <Cell key={i} x={X + i * bw} y={210} w={bw} tone={i < 2 ? "computed" : i === 2 ? "warn" : "plain"} text={i === 2 ? "t32 … t47, one differs" : `block ${i}: ${b}`} />
        ))}
      </motion.g>
      <motion.g {...on(0.6)}>
        <Label x={X} y={256} className="fill-accent">blocks 0 and 1: same keys, found in the cache</Label>
        <Label x={X} y={276} className="fill-warn">block 2: a new key, and block 3&apos;s key covers block 2 as well</Label>
        <Label x={X} y={296}>so a match stops at the first different block, and the blocks after it are computed</Label>
      </motion.g>
      <motion.g {...on(0.74)}>
        <Label x={X} y={330} className="fill-text">the key is hashed to find the block fast, and the tokens are kept to compare on a hit</Label>
      </motion.g>
      <Label x={X} y={380}>illustrative token numbers; the engine&apos;s blocks hold 16 tokens</Label>
    </Frame>
  );
}

const pool = 12;
const tableA = [1, 2, 3, 4, 5];
const tableB = [1, 2, 3, 4, 6];

export function TablesFigure() {
  const { on, draw } = useLoop(16);
  const pw = 42;
  const px = (b: number) => X + 10 + (b - 1) * pw;
  const tw = 40;
  return (
    <Frame label="Request A holds blocks 1 to 5 of the pool. Request B starts with the same 4 full blocks, so its block table points at blocks 1 to 4 too, each now held by 2 sequences, and only its fifth block, block 6, is new">
      <Label x={X} y={22} className="fill-text">two block tables, the same physical blocks</Label>
      <Label x={X} y={50}>block table of A</Label>
      {tableA.map((b, i) => (
        <Cell key={i} x={X + 110 + i * tw} y={36} w={tw} h={20} tone={i < 4 ? "computed" : "warn"} text={String(b)} />
      ))}
      <Label x={X} y={174}>the KV pool</Label>
      {Array.from({ length: pool }, (_, i) => {
        const b = i + 1;
        const tone = b <= 4 ? "computed" : b === 5 || b === 6 ? "warn" : "blank";
        return (
          <g key={b}>
            <Cell x={px(b)} y={182} w={pw} h={30} tone={tone} text={String(b)} />
          </g>
        );
      })}
      {tableA.map((b, i) => (
        <motion.line key={`a${i}`} x1={X + 110 + i * tw + 18} y1={58} x2={px(b) + 19} y2={180} strokeWidth={1} className="stroke-text/50" {...draw(0.04 + i * 0.03, 0.2 + i * 0.03)} />
      ))}
      <motion.g {...on(0.36)}>
        <Label x={X} y={320}>block table of B</Label>
        {tableB.map((b, i) => (
          <Cell key={i} x={X + 110 + i * tw} y={306} w={tw} h={20} tone={i < 4 ? "computed" : "warn"} text={String(b)} />
        ))}
      </motion.g>
      {tableB.map((b, i) => (
        <motion.line key={`b${i}`} x1={X + 110 + i * tw + 18} y1={304} x2={px(b) + 19} y2={214} strokeWidth={1.2} className={i < 4 ? "stroke-accent" : "stroke-warn"} {...draw(0.4 + i * 0.03, 0.56 + i * 0.03)} />
      ))}
      <motion.g {...on(0.62)}>
        {[1, 2, 3, 4].map((b) => (
          <text key={b} x={px(b) + 19} y={234} textAnchor="middle" className="fill-accent text-[10px]">
            2
          </text>
        ))}
        <Label x={px(7)} y={234} className="fill-accent">← held by 2: the count</Label>
      </motion.g>
      <motion.g {...on(0.7)}>
        <Label x={X} y={360} className="fill-text">B&apos;s prefill starts at token 64: its rows read the shared keys through its own table</Label>
      </motion.g>
      <Label x={X} y={390}>illustrative block numbers; blue full blocks are shared, pink ones hold a request&apos;s last tokens</Label>
    </Frame>
  );
}

export function PublishFigure() {
  const { on, grow } = useLoop(16);
  const sw = 150;
  const steps = [
    { label: "step 1", rows: ["A: system prompt + question", "B: system prompt + question"], tone: "computed" as const },
    { label: "step 2", rows: ["A, B: decode"], tone: "plain" as const },
    { label: "step 3", rows: ["C: system prompt reused", "A, B: decode"], tone: "cached" as const },
  ];
  return (
    <Frame label="A and B are admitted in step 1 and both compute the system prompt, because nothing is published until the step finishes. At the end of step 1 their full blocks are published, and C, admitted in step 3, reuses them">
      <Label x={X} y={22} className="fill-text">published when the step that wrote it ends</Label>
      {steps.map((s, i) => (
        <motion.g key={s.label} {...on(0.06 + i * 0.22)}>
          <Label x={X + i * (sw + 20)} y={56} className="fill-text">{s.label}</Label>
          <motion.g {...grow(0.06 + i * 0.22, 0.14 + i * 0.22)}>
            {s.rows.map((r, k) => (
              <Cell key={k} x={X + i * (sw + 20)} y={66 + k * 30} w={sw} tone={k === 0 ? s.tone : "plain"} text={r} />
            ))}
          </motion.g>
        </motion.g>
      ))}
      <motion.g {...on(0.2)}>
        <line x1={X + sw + 8} y1={50} x2={X + sw + 8} y2={140} strokeDasharray="3 3" strokeWidth={1.5} className="stroke-accent" />
        <Label x={X + sw + 14} y={156} className="fill-accent">publish</Label>
      </motion.g>
      <motion.g {...on(0.12)}>
        <Label x={X} y={190} className="fill-warn">A and B, the same step: both prefill the whole system prompt</Label>
      </motion.g>
      <motion.g {...on(0.56)}>
        <Label x={X} y={212} className="fill-accent">C, a later step: its system prompt is already in the cache</Label>
      </motion.g>
      <motion.g {...on(0.68)}>
        <Label x={X} y={256} className="fill-text">what a finished step publishes</Label>
        <Label x={X} y={278}>every full block whose keys and values it wrote: prompt blocks after a prefill,</Label>
        <Label x={X} y={296}>and a block of generated tokens when a decode step fills it</Label>
        <Label x={X} y={318}>so a second chat turn, which repeats the first turn and its answer, finds both</Label>
      </motion.g>
      <Label x={X} y={380}>illustrative steps; the rule is Scheduler::update in src/sched/scheduler.cpp</Label>
    </Frame>
  );
}

type Slot = "blank" | "held" | "cached" | "taken";
const slots: Slot[] = ["cached", "cached", "cached", "cached", "held", "held", "blank", "blank", "cached", "cached", "held", "held"];
const order = [3, 4, 1, 2, 0, 0, 0, 0, 7, 8, 0, 0];

export function PoolFigure() {
  const { on } = useLoop(16);
  const pw = 42;
  const bx = (i: number) => X + 10 + i * pw;
  const tone = (s: Slot) => (s === "held" ? "computed" : s === "cached" ? "cached" : s === "taken" ? "warn" : "blank");
  return (
    <Frame label="A freed request's published blocks stay in the pool as cached blocks and count as free. A new request takes blank blocks first; only when none is left does it evict a cached block, the one released longest ago, the end of a chain before its start">
      <Label x={X} y={22} className="fill-text">a released block stays cached, and counts as free</Label>
      {slots.map((s, i) => (
        <Cell key={i} x={bx(i)} y={52} w={pw} h={30} tone={tone(s)} text={String(i + 1)} />
      ))}
      <rect x={X} y={104} width={10} height={10} rx={1} className="fill-accent" />
      <Label x={X + 16} y={113}>held by a running request</Label>
      <rect x={X + 180} y={104} width={10} height={10} rx={1} className="fill-text/35" />
      <Label x={X + 196} y={113}>cached: released, keys kept</Label>
      <rect x={X + 370} y={104} width={10} height={10} rx={1} className="fill-none stroke-border" strokeDasharray="3 2" />
      <Label x={X + 386} y={113}>blank</Label>
      <motion.g {...on(0.12)}>
        <Label x={X} y={152} className="fill-text">a new request needs 4 blocks; 2 are blank</Label>
        {[6, 7].map((i) => (
          <Cell key={i} x={bx(i)} y={162} w={pw} h={30} tone="warn" text="1st" />
        ))}
      </motion.g>
      <motion.g {...on(0.34)}>
        <Label x={X} y={222} className="fill-text">then it evicts: the block released longest ago, the end of its chain first</Label>
        {order.map((n, i) =>
          n > 0 ? (
            <text key={i} x={bx(i) + 19} y={248} textAnchor="middle" className={`${n <= 2 ? "fill-warn" : "fill-muted"} text-[10px]`}>
              {n <= 2 ? `evict ${n}` : `then ${n}`}
            </text>
          ) : null,
        )}
        {[2, 3].map((i) => (
          <Cell key={i} x={bx(i)} y={256} w={pw} h={30} tone="warn" text="2nd" />
        ))}
      </motion.g>
      <motion.g {...on(0.56)}>
        <Label x={X} y={318}>blocks 1 to 4 were one request&apos;s chain, released before blocks 9 and 10; the chain&apos;s end goes first,</Label>
        <Label x={X} y={336}>so what stays cached is still a start that a later prompt can match</Label>
      </motion.g>
      <Label x={X} y={380}>{`illustrative pool; the engine's holds ${fmt(prefix.pool)} blocks of ${prefix.block} tokens`}</Label>
    </Frame>
  );
}

export function TtftFigure() {
  const { on, grow } = useLoop(14);
  const most = Math.max(...pairs.map((q) => q.quiet.off.later_ttft_ms));
  const scale = 300 / most;
  const row = 92;
  return (
    <Frame label={`Time to the first token of the later requests, cache off and on: ${pairs.map((q) => `${fmt(q.quiet.off.later_ttft_ms, 1)} against ${fmt(q.quiet.on.later_ttft_ms, 1)} ms at a ${fmt(q.shared)}-token prefix`).join(", ")} (measured)`}>
      <Label x={X} y={22} className="fill-text">{`a later request's time to the first token, arriving ${prefix.quietArrivals.replace("/s", " a second")}`}</Label>
      {pairs.map((q, i) => {
        const y = 54 + i * row;
        return (
          <g key={q.shared}>
            <Label x={X} y={y + 12} className="fill-text">{`${fmt(q.shared)} shared + ${prefix.own}`}</Label>
            <motion.rect x={X + 130} y={y} width={q.quiet.off.later_ttft_ms * scale} height={16} rx={2} className="fill-text/60" {...grow(0.04 + i * 0.1, 0.14 + i * 0.1)} />
            <motion.rect x={X + 130} y={y + 24} width={q.quiet.on.later_ttft_ms * scale} height={16} rx={2} className="fill-accent" {...grow(0.36 + i * 0.08, 0.44 + i * 0.08)} />
            <motion.g {...on(0.14 + i * 0.1)}>
              <Label x={X + 136 + q.quiet.off.later_ttft_ms * scale} y={y + 12} className="fill-text">{`off ${fmt(q.quiet.off.later_ttft_ms, 1)} ms`}</Label>
            </motion.g>
            <motion.g {...on(0.44 + i * 0.08)}>
              <Label x={X + 136 + q.quiet.on.later_ttft_ms * scale} y={y + 36} className="fill-accent">{`on ${fmt(q.quiet.on.later_ttft_ms, 1)} ms`}</Label>
            </motion.g>
          </g>
        );
      })}
      <motion.g {...on(0.7)}>
        <Label x={X} y={330} className="fill-text">{`with the cache on, a prefill step takes ${fmt(prefix.prefillLow)} to ${fmt(prefix.prefillHigh)} ms whatever the prefix: the request's own ${prefix.own} tokens`}</Label>
      </motion.g>
      <Label x={X} y={380}>{`measured ${prefix.date} on the ${prefix.gpu}; median of the ${p.quiet.on.requests - 1} later requests, median of ${p.quiet.on.runs} runs`}</Label>
    </Frame>
  );
}

export function BurstFigure() {
  const { on, grow } = useLoop(14);
  const most = Math.max(...pairs.map((q) => q.burst.on.output_tps));
  const scale = 300 / most;
  const row = 92;
  return (
    <Frame label={`All ${p.quiet.on.requests} requests at once: output tokens per second ${pairs.map((q) => `${fmt(q.burst.off.output_tps)} against ${fmt(q.burst.on.output_tps)} at a ${fmt(q.shared)}-token prefix, ${fmt(prefix.speedup(q), 1)} times`).join(", ")} (measured)`}>
      <Label x={X} y={22} className="fill-text">{`all ${p.quiet.on.requests} requests at once: output tokens per second`}</Label>
      {pairs.map((q, i) => {
        const y = 54 + i * row;
        return (
          <g key={q.shared}>
            <Label x={X} y={y + 12} className="fill-text">{`${fmt(q.shared)} shared + ${prefix.own}`}</Label>
            <Label x={X} y={y + 36}>{`hit rate ${fmt(100 * q.burst.on.hit_rate, 1)}%`}</Label>
            <motion.rect x={X + 130} y={y} width={q.burst.off.output_tps * scale} height={16} rx={2} className="fill-text/60" {...grow(0.04 + i * 0.1, 0.14 + i * 0.1)} />
            <motion.rect x={X + 130} y={y + 24} width={q.burst.on.output_tps * scale} height={16} rx={2} className="fill-accent" {...grow(0.36 + i * 0.08, 0.44 + i * 0.08)} />
            <motion.g {...on(0.14 + i * 0.1)}>
              <Label x={X + 136 + q.burst.off.output_tps * scale} y={y + 12} className="fill-text">{`off ${fmt(q.burst.off.output_tps)}`}</Label>
            </motion.g>
            <motion.g {...on(0.44 + i * 0.08)}>
              <Label x={X + 136 + q.burst.on.output_tps * scale} y={y + 36} className="fill-accent">{`on ${fmt(q.burst.on.output_tps)}, ${fmt(prefix.speedup(q), 1)}×`}</Label>
            </motion.g>
          </g>
        );
      })}
      <motion.g {...on(0.7)}>
        <Label x={X} y={330} className="fill-text">{`a hit counts only its ${prefix.own} uncached rows against the ${fmt(prefix.budget)}-token prefill budget, so the batch fills sooner`}</Label>
      </motion.g>
      <Label x={X} y={380}>{`measured ${prefix.date} on the ${prefix.gpu}, max batch ${prefix.maxBatch}; median of ${p.burst.on.runs} runs`}</Label>
    </Frame>
  );
}

export function SameOutputFigure() {
  const { on } = useLoop(14);
  const s = prefix.sameOut;
  const lw = Math.floor(W / s.layers);
  const per = 29;
  return (
    <Frame label={`The keys and values a hit reuses are bit-identical to the ones the request would write itself, in all ${s.layers} layers. Of ${s.outputs} greedy outputs compared with the cache off, ${s.same} are identical and ${s.split} split at a near tie, within ${s.ulps} BF16 ulps`}>
      <Label x={X} y={22} className="fill-text">the same keys and values, nearly the same answers</Label>
      <Label x={X} y={56}>{`the reused keys and values, layer by layer, against the request's own prefill`}</Label>
      {Array.from({ length: s.layers }, (_, i) => (
        <motion.g key={i} {...on(0.04 + i * 0.012)}>
          <rect x={X + i * lw} y={66} width={lw - 3} height={22} rx={2} className="fill-accent" />
          <text x={X + i * lw + (lw - 3) / 2} y={81} textAnchor="middle" className="fill-bg text-[10px]">
            =
          </text>
        </motion.g>
      ))}
      <motion.g {...on(0.32)}>
        <Label x={X} y={110} className="fill-accent">{`bit-identical in all ${s.layers} layers`}</Label>
      </motion.g>
      <Label x={X} y={152}>{`${s.outputs} greedy outputs, cache on against cache off`}</Label>
      {Array.from({ length: s.outputs }, (_, i) => {
        const split = i >= s.same;
        return (
          <motion.circle key={i} cx={X + 8 + (i % per) * 18} cy={172 + Math.floor(i / per) * 20} r={6} className={split ? "fill-warn" : "fill-accent"} {...on(0.38 + (i / s.outputs) * 0.2)} />
        );
      })}
      <motion.g {...on(0.62)}>
        <Label x={X} y={240} className="fill-accent">{`${s.same} identical`}</Label>
        <Label x={X + 110} y={240} className="fill-warn">{`${s.split} split at a near tie, within ${s.ulps} BF16 ulps of the best logit`}</Label>
      </motion.g>
      <motion.g {...on(0.72)}>
        <Label x={X} y={284} className="fill-text">why: the request&apos;s own rows run as a separate chunk, and a GEMM&apos;s rounding</Label>
        <Label x={X} y={302} className="fill-text">depends on how many rows a step holds; chunked prefill splits the same way</Label>
      </motion.g>
      <Label x={X} y={380}>tests/gpu/test_serve_prefix_cache.cpp, as results.md reports it</Label>
    </Frame>
  );
}
