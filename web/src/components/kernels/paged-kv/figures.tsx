"use client";

import { motion } from "motion/react";
import { kvReplay } from "@/lib/data";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

const replay = kvReplay();
const B = replay.meta.block_size;
export const fullContext = 2048;
const fullBlocks = fullContext / B;

export const reqs = replay.requests;
export const first = reqs[0];
const lender = reqs.find((q) => q.id !== first.id && q.finished >= 0 && q.blocks.some((b) => b.id === first.blocks[3].id && b.step < first.blocks[3].step));

export type HeldBlock = { id: number; req: number; fill: number };

export function heldAt(step: number): HeldBlock[] {
  const out: HeldBlock[] = [];
  const decodes = replay.steps[step][3];
  for (const q of reqs) {
    if (q.admitted > step || (q.finished >= 0 && step >= q.finished)) continue;
    const length = q.prompt + decodes - q.decodes;
    const blocks = q.blocks.filter((b) => b.step <= step);
    blocks.forEach((b, j) =>
      out.push({
        id: b.id,
        req: q.id,
        fill: j < blocks.length - 1 ? 1 : (length - j * B) / B,
      }),
    );
  }
  return out;
}

const pos = 100;
const logical = Math.floor(pos / B);
export const lookup = {
  pos,
  logical,
  phys: first.blocks[logical].id,
  slot: first.blocks[logical].id * B + (pos % B),
  row: pos % B,
  step: first.blocks[logical].step,
};

const fmt = (v: number, d = 0) =>
  v.toLocaleString("en-US", {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
  });

export function WhyBlocksFigure() {
  const { grow, on } = useLoop(9);
  const rows = reqs.slice(0, 4);
  const reserved = rows.length * fullBlocks;
  const held = rows.reduce((n, q) => n + q.blocks.length, 0);
  return (
    <Frame
      label={`The first four requests of the recorded run reserved for a full ${fullContext}-token context each against the blocks of 16 tokens they actually filled: ${reserved} blocks reserved, ${held} needed`}
    >
      <Label x={20} y={24}>{`one contiguous cache per request, sized for the full ${fmt(fullContext)} tokens`}</Label>
      {rows.map((q, i) => {
        const y = 40 + i * 30;
        const used = (q.final_length / fullContext) * 520;
        return (
          <g key={q.id}>
            <rect x={20} y={y} width={520} height={18} rx={2} className="fill-warn/15 stroke-warn/40" />
            <motion.rect x={20} y={y} width={used} height={18} rx={2} className="fill-accent" {...grow(0.05 + i * 0.04, 0.3 + i * 0.04)} />
            <Label x={26 + used} y={y + 13} className="fill-muted">{`request ${q.id}: ${q.final_length} tokens`}</Label>
          </g>
        );
      })}
      <motion.g {...on(0.3)}>
        <Label x={20} y={176} className="fill-warn">{`${fmt(reserved)} blocks of room reserved, the rest of each row never written`}</Label>
      </motion.g>
      <Label x={20} y={218}>
        the same requests in blocks of 16 tokens, taken as they grow
      </Label>
      {rows.map((q, i) => {
        const y = 234 + i * 30;
        return (
          <g key={q.id}>
            {q.blocks.map((b, j) => {
              const fill = j < q.blocks.length - 1 ? 1 : (q.final_length - j * B) / B;
              const t = 0.42 + (j / q.blocks.length) * 0.3;
              return (
                <motion.g key={j} {...on(t)}>
                  <rect x={20 + j * 26} y={y} width={23} height={18} rx={2} className="fill-surface stroke-accent/60" />
                  <rect x={20 + j * 26} y={y} width={23 * fill} height={18} rx={2} className="fill-accent" />
                </motion.g>
              );
            })}
            <Label x={300} y={y + 13}>{`${q.blocks.length} blocks, ${q.blocks.length * B - q.final_length} ${q.blocks.length * B - q.final_length === 1 ? "slot" : "slots"} spare`}</Label>
          </g>
        );
      })}
      <motion.g {...on(0.76)}>
        <Label x={20} y={372} className="fill-accent">{`${held} blocks held: at most 15 slots unused per request`}</Label>
      </motion.g>
    </Frame>
  );
}

const poolCols = 16;
const poolX = (id: number) => 20 + (id % poolCols) * 32.5;
const poolY = (id: number, top: number) => top + Math.floor(id / poolCols) * 28;

export function TableFigure() {
  const { on } = useLoop(10);
  const table = first.blocks.map((b) => b.id);
  const { pos, logical: k, phys, slot, step } = lookup;
  const others = new Set(
    heldAt(step)
      .filter((b) => b.req !== first.id)
      .map((b) => b.id),
  );
  const top = 230;
  const w = 50;
  return (
    <Frame
      label={`Request ${first.id}'s ${table.length} logical blocks map through its block table to blocks ${table.join(", ")} of the shared pool; position ${pos} is logical block ${k}, physical block ${phys}, slot ${slot}`}
    >
      <Label x={20} y={24}>{`request ${first.id}: ${first.final_length} tokens, ${table.length} logical blocks of ${B}`}</Label>
      {table.map((id, j) => (
        <g key={j}>
          <rect x={20 + j * (w + 2)} y={34} width={w} height={24} rx={3} className={j === k ? "fill-accent/15 stroke-accent" : "fill-surface stroke-border"} />
          <Label x={20 + j * (w + 2) + w / 2} y={50} anchor="middle" className="fill-text">
            {`${j * B}–${j * B + B - 1}`}
          </Label>
          <rect x={20 + j * (w + 2)} y={92} width={w} height={24} rx={3} className={j === k ? "fill-accent/15 stroke-accent" : "fill-bg stroke-border"} />
          <Label x={20 + j * (w + 2) + w / 2} y={108} anchor="middle" className="fill-text">
            {id}
          </Label>
          <motion.line x1={20 + j * (w + 2) + w / 2} y1={116} x2={poolX(id) + 15} y2={poolY(id, top)} className="stroke-accent/50" strokeWidth={1} {...on(0.04 + j * 0.02)} />
        </g>
      ))}
      <Label x={20} y={84}>
        block table, from first-fit allocation
      </Label>
      <motion.g {...on(0.3)}>
        <Label x={20} y={146} className="fill-text">{`position ${pos}: table[${pos} / ${B}] = table[${k}] = ${phys}`}</Label>
      </motion.g>
      <motion.g {...on(0.45)}>
        <Label x={20} y={164} className="fill-text">{`slot = ${phys} · ${B} + ${pos} mod ${B} = ${slot}`}</Label>
      </motion.g>
      <motion.g {...on(0.6)}>
        <Label x={20} y={182} className="fill-accent">{`its K and V go to row ${pos % B} of block ${phys}, in every layer`}</Label>
      </motion.g>
      <Label x={20} y={top - 12}>{`the pool at step ${step}, when request ${first.id} took block ${phys}: blocks 0 to 63 of ${fmt(replay.meta.kv_blocks)}`}</Label>
      {Array.from({ length: 64 }, (_, id) => {
        const mine = table.includes(id);
        const cls = id === 0 ? "fill-warn/20 stroke-warn/50" : mine ? "fill-accent/25 stroke-accent" : others.has(id) ? "fill-baseline/30 stroke-border" : "fill-surface stroke-border";
        return (
          <g key={id}>
            <rect x={poolX(id)} y={poolY(id, top)} width={30} height={24} rx={2} className={cls} />
            <Label x={poolX(id) + 15} y={poolY(id, top) + 16} anchor="middle" className={mine ? "fill-text" : "fill-muted"}>
              {id}
            </Label>
          </g>
        );
      })}
      <motion.rect x={poolX(phys) - 2} y={poolY(phys, top) - 2} width={34} height={28} rx={3} className="fill-none stroke-accent" strokeWidth={2} {...on(0.5)} />
      {[
        {
          x: 20,
          label: `request ${first.id}`,
          cls: "fill-accent/25 stroke-accent",
        },
        {
          x: 140,
          label: "other requests",
          cls: "fill-baseline/30 stroke-border",
        },
        {
          x: 280,
          label: "block 0, never handed out",
          cls: "fill-warn/20 stroke-warn/50",
        },
      ].map((l) => (
        <g key={l.label}>
          <rect x={l.x} y={358} width={10} height={10} rx={2} className={l.cls} />
          <Label x={l.x + 16} y={367}>
            {l.label}
          </Label>
        </g>
      ))}
    </Frame>
  );
}

export function WriteKvFigure({ launchUs, share }: { launchUs: number; share: number }) {
  const { on, move } = useLoop(9);
  const width = 2560;
  const kv = 256;
  const px = (c: number) => 20 + (c / width) * 520;
  const kx = px(width - 2 * kv);
  const vx = px(width - kv);
  const seg = (px(kv) - px(0)) / 4;
  const tileW = 120;
  const tileX = (h: number) => 20 + h * (tileW + 14);
  const { row, phys, slot } = lookup;
  return (
    <Frame label="write_kv copies the K and V slices from the end of a token's fused QKV row into one row of its block, one thread per element, for each of the four KV heads">
      <Label x={20} y={24}>{`one token's fused QKV row, ${fmt(width)} values (bf16)`}</Label>
      <rect x={20} y={34} width={kx - 20} height={22} rx={2} className="fill-baseline/40" />
      <Label x={(20 + kx) / 2} y={49} anchor="middle" className="fill-text">
        Q: 32 heads × 64, read by attention
      </Label>
      {[0, 1, 2, 3].map((h) => (
        <g key={h}>
          <rect x={kx + h * seg} y={34} width={seg - 1} height={22} className="fill-accent" />
          <rect x={vx + h * seg} y={34} width={seg - 1} height={22} className="fill-accent/50" />
        </g>
      ))}
      <Label x={kx + 2 * seg} y={72} anchor="middle" className="fill-accent">
        K
      </Label>
      <Label x={vx + 2 * seg} y={72} anchor="middle" className="fill-accent">
        V
      </Label>
      <Label x={20} y={104}>{`K cache, block ${phys} of layer l: [KV head, ${B} rows, 64 dims]`}</Label>
      {[0, 1, 2, 3].map((h) => (
        <g key={h}>
          <Label x={tileX(h)} y={124}>{`head ${h}`}</Label>
          {Array.from({ length: B }, (_, r) => (
            <rect key={r} x={tileX(h)} y={132 + r * 8} width={tileW} height={7} className={r < row ? "fill-accent/30" : "fill-surface stroke-border"} strokeWidth={0.5} />
          ))}
          <motion.rect x={kx + h * seg} y={34} width={seg - 1} height={7} className="fill-accent" {...move(0.1 + h * 0.08, 0.3 + h * 0.08, tileX(h) - kx - h * seg, 132 + row * 8 - 34, false, 0.85)} />
          <motion.rect x={tileX(h)} y={132 + row * 8} width={tileW} height={7} className="fill-accent" {...on(0.3 + h * 0.08)} />
        </g>
      ))}
      <Label x={20} y={276} className="fill-text">{`row ${row}: slot ${slot} = block ${phys} · ${B} + ${row}; V goes to the same place in the V cache`}</Label>
      <motion.g {...on(0.55)}>
        <Label x={20} y={300}>{`one thread per (token, KV head, dim): 4 · 64 = 256 threads a token,`}</Label>
        <Label x={20} y={316}>
          each copying one K and one V value bit for bit
        </Label>
        <Label x={20} y={332}>{`index = ((block · 4 + head) · ${B} + row) · 64 + dim; a slot of -1 writes nothing`}</Label>
      </motion.g>
      <motion.g {...on(0.72)}>
        <Label x={20} y={366} className="fill-accent">{`22 launches per batch-1 decode step: ${fmt(launchUs, 1)} µs, ${share}% of the kernel time`}</Label>
      </motion.g>
    </Frame>
  );
}

export function AppendFigure() {
  const { on } = useLoop(12);
  const start = first.prompt;
  const blocks = first.blocks.slice(0, 4);
  const cw = 27;
  const rowY = (j: number) => 64 + j * 44;
  const fillEnd = 3 * B + 6;
  const span = 0.7;
  const at = (n: number) => 0.06 + ((n - start) / (fillEnd - start)) * span;
  const newBlock = blocks[3];
  return (
    <Frame
      label={`Request ${first.id} decodes one token per step: slots fill one by one from position ${start}, and the block manager takes block ${newBlock.id} only when position 48 starts a new block`}
    >
      <Label x={20} y={24}>{`request ${first.id} after its ${start}-token prompt, one decode step at a time`}</Label>
      {blocks.map((b, j) => (
        <g key={j}>
          <motion.g {...(j === 3 ? on(at(3 * B)) : {})}>
            <Label x={20} y={rowY(j) + 14} className="fill-text">{`block ${b.id}`}</Label>
            {Array.from({ length: B }, (_, s) => {
              const n = j * B + s;
              return (
                <g key={s}>
                  <rect x={86 + s * cw} y={rowY(j)} width={cw - 3} height={22} rx={2} className="fill-surface stroke-border" />
                  {n < start ? (
                    <rect x={86 + s * cw} y={rowY(j)} width={cw - 3} height={22} rx={2} className="fill-accent/45" />
                  ) : n < fillEnd ? (
                    <motion.rect x={86 + s * cw} y={rowY(j)} width={cw - 3} height={22} rx={2} className="fill-accent" {...on(at(n), 0.94, n < fillEnd)} />
                  ) : null}
                </g>
              );
            })}
          </motion.g>
        </g>
      ))}
      <Label x={86} y={rowY(4) + 6}>
        prompt (prefill)
      </Label>
      <rect x={70} y={rowY(4) - 3} width={10} height={10} rx={2} className="fill-accent/45" />
      <rect x={220} y={rowY(4) - 3} width={10} height={10} rx={2} className="fill-accent" />
      <Label x={236} y={rowY(4) + 6}>
        one new token per decode step
      </Label>
      <motion.g {...on(at(3 * B))}>
        <Label x={20} y={276} className="fill-text">{`position 48: length % ${B} == 0, so append_slot calls allocate_block()`}</Label>
        <Label x={20} y={294}>{`which takes the lowest free block: ${newBlock.id}, at step ${newBlock.step}`}</Label>
      </motion.g>
      <motion.g {...on(at(3 * B) + 0.08)}>
        <Label x={20} y={326}>
          every other step only writes into a slot the request already holds:
        </Label>
        <Label x={20} y={342}>{`${first.blocks.length - first.blocks.filter((b) => b.step === first.admitted).length} allocations in its ${first.final_length - first.prompt} decode steps`}</Label>
      </motion.g>
    </Frame>
  );
}

export function ReuseFigure() {
  const { on, move } = useLoop(12);
  const lent = lender ?? reqs[1];
  const lentIds = lent.blocks.map((b) => b.id);
  const takers = reqs
    .filter((q) => q.id !== lent.id)
    .flatMap((q) => q.blocks.filter((b) => lentIds.includes(b.id) && b.step > lent.finished).map((b) => ({ q: q.id, ...b })))
    .sort((a, b) => a.step - b.step);
  const firstTake = lentIds.map((id) => takers.find((t) => t.id === id)).filter((t) => t !== undefined);
  const x = (j: number) => 200 + j * 70;
  return (
    <Frame
      label={`Request ${lent.id} finishes at step ${lent.finished} and its blocks ${lentIds.join(" and ")} go back to the pool; the next requests that need a block take them first. When the pool is full, the newest request is preempted instead`}
    >
      <Label x={20} y={24}>
        from the recorded run
      </Label>
      <Label x={20} y={58} className="fill-text">{`request ${lent.id}`}</Label>
      {lentIds.map((id, j) => (
        <g key={id}>
          <motion.g {...on(0.02, 0.24, false)}>
            <rect x={x(j)} y={42} width={60} height={26} rx={3} className="fill-baseline/40 stroke-border" />
          </motion.g>
          <Label x={x(j) + 30} y={60} anchor="middle" className="fill-text">{`block ${id}`}</Label>
        </g>
      ))}
      <motion.g {...on(0.02, 0.24, false)}>
        <Label x={200} y={88}>{`${lent.prompt}-token prompt, ${lent.final_length - lent.prompt} tokens out`}</Label>
      </motion.g>
      <motion.g {...on(0.24)}>
        <Label x={200} y={88} className="fill-accent">{`finished, step ${lent.finished}: blocks freed`}</Label>
      </motion.g>
      {firstTake.map((t, j) => (
        <g key={t.id}>
          <motion.rect x={x(j)} y={42} width={60} height={26} rx={3} className="fill-accent/30 stroke-accent" {...move(0.36 + j * 0.12, 0.46 + j * 0.12, 0, 60, true, 0.94)} />
          <motion.g {...on(0.46 + j * 0.12)}>
            <Label x={x(j) + 30} y={138} anchor="middle" className="fill-accent">{`request ${t.q}`}</Label>
            <Label x={x(j) + 30} y={154} anchor="middle">{`step ${t.step}`}</Label>
          </motion.g>
        </g>
      ))}
      <Label x={20} y={118} className="fill-text">
        taken again
      </Label>
      <Label x={20} y={196}>
        first fit hands out the lowest free id, so freed blocks are reused first
      </Label>
      <Label x={20} y={212}>{`and the whole run stays inside blocks 1 to ${replay.max_block}`}</Label>
      <line x1={20} x2={540} y1={236} y2={236} className="stroke-border" />
      <Label x={20} y={260} className="fill-text">
        when the pool is full
      </Label>
      {Array.from({ length: 14 }, (_, i) => (
        <rect key={i} x={20 + i * 37} y={272} width={34} height={20} rx={2} className={i >= 11 ? "fill-warn/40" : "fill-baseline/40"} />
      ))}
      <motion.g {...on(0.55)}>
        <Label x={20} y={312}>{`no block left for the next token: the newest request is preempted,`}</Label>
        <Label x={20} y={328}>
          its blocks freed and its prompt plus output recomputed when there is room
        </Label>
      </motion.g>
      <motion.g {...on(0.72)}>
        <Label x={20} y={360} className="fill-accent">
          a 56-block cache, 6 preemptions: all 16 outputs equal the roomy run&apos;s
        </Label>
      </motion.g>
    </Frame>
  );
}
