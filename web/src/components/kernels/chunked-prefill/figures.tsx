"use client";

import { motion } from "motion/react";
import { type ChunkStep } from "@/content/kernels/chunked-prefill";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

const names = ["A", "B", "C"];

function Cell({ x, y, w, h = 24, text, className }: { x: number; y: number; w: number; h?: number; text?: string; className: string }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={3} className={className} />
      {text && (
        <text x={x + w / 2} y={y + h / 2 + 4} textAnchor="middle" className="fill-bg text-[11px]">
          {text}
        </text>
      )}
    </g>
  );
}

export function AllPrefillFigure({ ms }: { ms: string }) {
  const { on, grow } = useLoop(9);
  const lane = (i: number) => 64 + i * 44;
  const steps = [
    { x: 50, w: 40, at: 0.04 },
    { x: 94, w: 40, at: 0.12 },
    { x: 402, w: 40, at: 0.58 },
    { x: 446, w: 40, at: 0.66 },
  ];
  return (
    <Frame label="schedule() as lanes over time: three sequences decode for two steps, then one step holds only the new prompt and all three lanes wait, then decoding resumes. Step widths are illustrative">
      <Label x={20} y={28}>schedule(): a step is all prefill or all decode</Label>
      {[...names, "P"].map((n, i) => (
        <g key={n}>
          <text x={20} y={lane(i) + 18} className={i === 3 ? "fill-series-2 text-[12px]" : "fill-accent text-[12px]"}>
            {n}
          </text>
          <line x1={46} y1={lane(i) + 32} x2={540} y2={lane(i) + 32} className="stroke-border" />
        </g>
      ))}
      {steps.map((s) =>
        names.map((n, i) => (
          <motion.g key={`${s.x}-${n}`} {...on(s.at)}>
            <Cell x={s.x} y={lane(i)} w={s.w} className="fill-accent" />
          </motion.g>
        )),
      )}
      {names.map((n, i) => (
        <motion.g key={n} {...on(0.22, 0.94)}>
          <rect x={138} y={lane(i)} width={260} height={24} rx={3} className="fill-none stroke-warn" strokeDasharray="4 3" />
          <text x={268} y={lane(i) + 16} textAnchor="middle" className="fill-warn text-[11px]">
            waiting
          </text>
        </motion.g>
      ))}
      <motion.rect x={138} y={lane(3)} width={260} height={24} rx={3} className="fill-series-2" {...grow(0.2, 0.5)} />
      <motion.g {...on(0.42)}>
        <text x={268} y={lane(3) + 16} textAnchor="middle" className="fill-bg text-[11px]">
          the whole prompt
        </text>
      </motion.g>
      <Label x={20} y={270}>one step, one prompt, no decode rows:</Label>
      <motion.g {...on(0.45)}>
        <Label x={20} y={290} className="fill-warn">A, B and C get no token until it ends</Label>
      </motion.g>
      <Label x={20} y={372} className="fill-text">{`a 2,048-token prompt alone: ${ms}`}</Label>
    </Frame>
  );
}

export function BudgetBarFigure({ budget, decoders }: { budget: number; decoders: number }) {
  const { on, grow } = useLoop(9);
  const slot = 520 / budget;
  const chunk = budget - decoders;
  const bar = (y: number, first: number, at: number) => (
    <g>
      {Array.from({ length: budget }, (_, i) => (
        <rect key={i} x={20 + i * slot + 1} y={y} width={slot - 2} height={34} rx={2} className="fill-none stroke-border" />
      ))}
      {Array.from({ length: budget }, (_, i) => {
        const d = i < decoders;
        const start = at + i * 0.02;
        return (
          <motion.rect key={i} x={20 + i * slot + 1} y={y} width={slot - 2} height={34} rx={2} className={d ? "fill-accent" : "fill-series-2"} {...grow(start, start + 0.03)} />
        );
      })}
      {names.map((n, i) => (
        <text key={n} x={20 + i * slot + slot / 2} y={y + 22} textAnchor="middle" className="fill-bg text-[11px]">
          {n}
        </text>
      ))}
      <Label x={20 + decoders * slot} y={y + 52}>{`prompt tokens ${first} to ${first + chunk - 1}`}</Label>
    </g>
  );
  return (
    <Frame label={`One step as a bar of ${budget} token slots: ${decoders} decode rows fill it first, the prompt's chunk takes the remaining ${chunk}, and the next step continues the prompt where this one stopped. The budget of 16 is illustrative`}>
      <Label x={20} y={28}>{`token budget ${budget} = rows in one step`}</Label>
      <Label x={20} y={58}>step 1</Label>
      {bar(66, 0, 0.04)}
      <motion.g {...on(0.46)}>
        <Label x={20} y={170}>step 2: the decode rows again, the prompt continues</Label>
        {bar(178, chunk, 0.5)}
      </motion.g>
      <motion.g {...on(0.2)}>
        <Label x={20} y={262} className="fill-accent">1  every decoding sequence gets its row first</Label>
      </motion.g>
      <motion.g {...on(0.28)}>
        <Label x={20} y={282} className="fill-series-2">2  the rest of the budget goes to the sequence in prefill</Label>
      </motion.g>
      <motion.g {...on(0.36)}>
        <Label x={20} y={302} className="fill-text">3  a new request is admitted only if budget is left</Label>
      </motion.g>
      <Label x={20} y={372}>token slots of an illustrative step, not engine output</Label>
    </Frame>
  );
}

export function SpreadFigure({ steps, budget, prompt }: { steps: ChunkStep[]; budget: number; prompt: number }) {
  const { grow, on } = useLoop(10);
  const unit = 3.6;
  const base = 320;
  const pitch = Math.min(64, 520 / steps.length);
  const w = pitch - 6;
  return (
    <Frame label={`A ${prompt}-token prompt with a budget of ${budget} and three decoders: ${steps.length} steps, each with the three decode rows and a chunk of the prompt. Column height is the tokens in the step`}>
      <Label x={20} y={28}>{`${prompt}-token prompt, budget ${budget}: ${steps.length} steps`}</Label>
      <line x1={20} y1={base} x2={540} y2={base} className="stroke-border" />
      <line x1={20} y1={base - budget * unit} x2={540} y2={base - budget * unit} className="stroke-baseline" strokeDasharray="4 3" />
      <Label x={540} y={base - budget * unit - 6} anchor="end">{`budget ${budget}`}</Label>
      {steps.map((s, k) => {
        const x = 20 + k * pitch;
        const at = 0.04 + (k * 0.8) / steps.length;
        const to = at + 0.8 / steps.length;
        return (
          <g key={k}>
            <motion.rect x={x} y={base - s.decode * unit} width={w} height={s.decode * unit} className="fill-accent" {...grow(at, to, "y")} />
            <motion.rect x={x} y={base - (s.decode + s.chunk) * unit} width={w} height={s.chunk * unit} className="fill-series-2" {...grow(at, to, "y")} />
            <Label x={x + w / 2} y={base + 16} anchor="middle">{k + 1}</Label>
          </g>
        );
      })}
      <motion.g {...on(0.88, 0.97)}>
        <Label x={20} y={362} className="fill-text">A, B and C get a row in every step</Label>
        <Label x={20} y={382} className="fill-series-2">{`the prompt's first token comes at step ${steps.length}`}</Label>
      </motion.g>
    </Frame>
  );
}

export function PreemptFigure() {
  const { on, move } = useLoop(10);
  const xs = [20, 150, 280, 410];
  const box = (i: number, y: number, className: string, text: string) => (
    <g>
      <rect x={xs[i]} y={y} width={120} height={44} rx={3} className={className} />
      <text x={xs[i] + 10} y={y + 18} className="fill-text text-[12px]">
        {text}
      </text>
    </g>
  );
  return (
    <Frame label="The running list is A, B, C, then P, still in prefill. A needs a new block and none is free, so the victim is the back of the list, P. P's blocks are freed, P goes to the front of the waiting queue with its progress forgotten, and A gets its block">
      <Label x={20} y={28}>running list, in admission order</Label>
      {box(0, 40, "fill-none stroke-border", "A")}
      {box(1, 40, "fill-none stroke-border", "B")}
      {box(2, 40, "fill-none stroke-border", "C")}
      <motion.g {...on(0, 0.4)}>
        {box(3, 40, "fill-none stroke-series-2", "P")}
        <rect x={xs[3] + 10} y={74} width={75} height={6} rx={2} className="fill-series-2" />
        <Label x={xs[3] + 10} y={100}>5 of 8 chunks</Label>
      </motion.g>
      <motion.g {...on(0.1, 0.94)}>
        <Label x={20} y={124} className="fill-warn">A needs a new block, none is free</Label>
      </motion.g>
      <Label x={20} y={172}>waiting queue, front first</Label>
      <rect x={20} y={180} width={520} height={44} rx={3} className="fill-none stroke-border" strokeDasharray="4 3" />
      <motion.g {...move(0.4, 0.6, -390, 140, true, 0.94)}>
        {box(3, 40, "fill-none stroke-series-2", "P")}
        <rect x={xs[3] + 10} y={74} width={75} height={6} rx={2} className="fill-border" />
        <Label x={xs[3] + 10} y={100}>0 chunks</Label>
      </motion.g>
      <motion.g {...on(0.55, 0.94)}>
        <rect x={xs[0] + 90} y={44} width={24} height={36} rx={2} className="fill-accent" />
        <Label x={xs[0] + 10} y={100} className="fill-accent">A gets its slot</Label>
      </motion.g>
      <motion.g {...on(0.6, 0.94)}>
        <Label x={20} y={264} className="fill-text">victim = back of the running list</Label>
        <Label x={20} y={284}>that is the sequence in prefill, admitted last</Label>
      </motion.g>
      <motion.g {...on(0.72, 0.94)}>
        <Label x={20} y={320}>P restarts at position 0 over its prompt plus output,</Label>
        <Label x={20} y={340}>its blocks were all taken at admission and are freed now</Label>
      </motion.g>
    </Frame>
  );
}

export function SampledFigure() {
  const { on } = useLoop(10);
  const pitch = 16;
  const step = (y: number, chunk: number, last: boolean, at: number) => {
    const chunkX = 128;
    const samples = last ? [...names, "P"] : names;
    return (
      <g>
        <motion.g {...on(at)}>
          {names.map((n, i) => (
            <Cell key={n} x={20 + i * 36} y={y} w={32} text={n} className="fill-accent" />
          ))}
          {Array.from({ length: chunk }, (_, i) => (
            <rect key={i} x={chunkX + i * pitch} y={y} width={pitch - 2} height={24} rx={2} className="fill-series-2" />
          ))}
          {last && <Label x={chunkX + chunk * pitch + 6} y={y + 16} className="fill-text">end of prompt</Label>}
        </motion.g>
        <motion.g {...on(at + 0.14)}>
          <Label x={20} y={y + 46}>sampled, in row order</Label>
          {samples.map((n, i) => (
            <Cell key={n} x={20 + i * 36} y={y + 54} w={32} text={n} className={n === "P" ? "fill-series-2" : "fill-accent"} />
          ))}
          {!last && (
            <g>
              <rect x={20 + 3 * 36} y={y + 54} width={32} height={24} rx={3} className="fill-none stroke-warn" strokeDasharray="4 3" />
              <Label x={20 + 4 * 36 + 4} y={y + 70} className="fill-warn">P: no logits, no entry</Label>
            </g>
          )}
        </motion.g>
      </g>
    );
  };
  return (
    <Frame label="Two steps. In the first the prompt's chunk is not its last, so only the three decode rows are sampled. In the second the chunk reaches the end of the prompt, so P joins the sampled list after the decoders">
      <Label x={20} y={28}>a step with a chunk that stops short</Label>
      {step(36, 13, false, 0.04)}
      <Label x={20} y={168}>the step whose chunk reaches the end of the prompt</Label>
      {step(176, 5, true, 0.5)}
      <motion.g {...on(0.8, 0.94)}>
        <Label x={20} y={320}>update() takes one token per sampled entry, in order</Label>
        <Label x={20} y={340}>a step of only unfinished chunks has none, and still gets its update</Label>
      </motion.g>
    </Frame>
  );
}
