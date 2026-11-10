"use client";

import { motion } from "motion/react";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

const fmt = (v: number) => v.toLocaleString("en-US");

function gridPath(x: number, y: number, cols: number, rows: number, cell: number) {
  let d = "";
  for (let c = 0; c <= cols; c++) d += `M${x + c * cell} ${y}V${y + rows * cell}`;
  for (let r = 0; r <= rows; r++) d += `M${x} ${y + r * cell}H${x + cols * cell}`;
  return d;
}

export const fragmentCells = (lane: number) => {
  const group = Math.floor(lane / 4);
  const quad = lane % 4;
  return { group, quad, rows: [group, group + 8], pairs: [2 * quad, 2 * quad + 1, 2 * quad + 8, 2 * quad + 9] };
};

const shownLanes = [0, 5, 18, 31];
const laneWindow = (i: number) => [0.03 + i * 0.235, 0.03 + (i + 1) * 0.235] as const;

export function TileSweepFigure({ mode, tile, tableMb, tileBytes, warps }: { mode: "about" | "kernel"; tile: number; tableMb?: number; tileBytes?: number; warps?: number }) {
  const { on } = useLoop(16);
  const n = 8;
  const s = 26;
  const x0 = 30;
  const y0 = 64;
  const row = 5;
  const slot = 0.125;
  const at = (c: number) => 0.06 + c * slot;
  const cells: [number, number][] = [];
  for (let r = 0; r < n; r++) for (let c = 0; c <= r; c++) cells.push([r, c]);
  const px = 284;
  return (
    <Frame label={mode === "about" ? `FlashAttention cuts the triangle of scores into tiles of ${tile} query rows by ${tile} keys: one tile of scores exists at a time while each row keeps a running maximum, sum and weighted values` : `One warp owns a tile of ${tile} query rows and the key tiles of the sequence stream past it, ${tile} keys at a time; the last tile is masked where a key comes after its query`}>
      <Label x={20} y={24} className="fill-text">{mode === "about" ? "the scores of a prompt: query rows down, keys across" : `one warp, ${tile} query rows of one head`}</Label>
      <Label x={x0} y={y0 - 10}>{`keys, ${tile} per tile →`}</Label>
      <Label x={x0 + 4.4 * s} y={y0 + 1.6 * s}>a row only sees</Label>
      <Label x={x0 + 4.4 * s} y={y0 + 1.6 * s + 16}>keys up to itself</Label>
      {cells.map(([r, c]) => (
        <rect key={`${r}-${c}`} x={x0 + c * s} y={y0 + r * s} width={s - 2} height={s - 2} rx={2} className="fill-surface stroke-border" />
      ))}
      <rect x={x0 - 3} y={y0 + row * s - 3} width={(row + 1) * s + 4} height={s + 4} rx={3} className="fill-none stroke-accent" />
      <Label x={x0 + (row + 1) * s + 8} y={y0 + row * s + 16} className="fill-accent">{`${tile} query rows`}</Label>
      {Array.from({ length: row + 1 }, (_, c) => (
        <g key={c}>
          <motion.rect x={x0 + c * s} y={y0 + row * s} width={s - 2} height={s - 2} rx={2} className="fill-accent/35" {...on(at(c) + slot, 0.94)} />
          <motion.rect x={x0 + c * s} y={y0 + row * s} width={s - 2} height={s - 2} rx={2} className="fill-accent" {...on(at(c), c === row ? 0.94 : at(c) + slot, c === row)} />
        </g>
      ))}
      <polygon points={`${x0 + row * s + 5},${y0 + row * s} ${x0 + row * s + s - 2},${y0 + row * s} ${x0 + row * s + s - 2},${y0 + row * s + s - 7}`} className="fill-warn/80" />
      <Label x={x0} y={y0 + n * s + 16}>{`${n} × ${n} tiles drawn: a ${n * tile}-token prompt`}</Label>
      <Label x={x0} y={y0 + n * s + 34} className="fill-warn">
        pink corner: keys after the query, masked
      </Label>
      <rect x={px} y={52} width={258} height={92} rx={3} className="fill-none stroke-border" />
      <Label x={px + 12} y={72} className="fill-text">
        one tile of scores at a time
      </Label>
      <path d={gridPath(px + 12, 84, 7, 4, 12)} className="fill-none stroke-border" />
      {Array.from({ length: row + 1 }, (_, c) => (
        <motion.rect key={c} x={px + 12} y={84} width={84} height={48} className="fill-accent/50" {...on(at(c), at(c) + slot * 0.7, c === row)} />
      ))}
      <Label x={px + 108} y={100}>{`${tile} × ${tile}, computed,`}</Label>
      <Label x={px + 108} y={116}>used and dropped</Label>
      {tileBytes !== undefined && <Label x={px + 108} y={132}>{`${fmt(tileBytes)} bytes in FP32`}</Label>}
      <rect x={px} y={156} width={258} height={mode === "about" ? 96 : 80} rx={3} className="fill-none stroke-accent" />
      <Label x={px + 12} y={176} className="fill-text">
        kept for every query row
      </Label>
      <Label x={px + 12} y={196}>m, the largest score so far</Label>
      <Label x={px + 12} y={212}>l, the running sum of e^(s − m)</Label>
      <Label x={px + 12} y={228}>acc, the running Σ weight × value</Label>
      {mode === "about" && <Label x={px + 12} y={244} className="fill-accent">the online softmax, once per row</Label>}
      {mode === "about" ? (
        <>
          <Label x={px} y={280}>the direct way stores the whole table first:</Label>
          {tableMb !== undefined && <Label x={px} y={298} className="fill-warn">{`${fmt(tableMb)} MB per head at 2,048 tokens`}</Label>}
          <Label x={px} y={326}>a tile of queries × a tile of keys is a</Label>
          <Label x={px} y={342}>small matrix multiply: tensor core work</Label>
        </>
      ) : (
        <>
          <Label x={px} y={262}>keys already in the KV cache: read through</Label>
          <Label x={px} y={278}>the chunk&apos;s one block table row</Label>
          <Label x={px} y={300}>keys of this chunk: read from qkv</Label>
          <Label x={px} y={322} className="fill-warn">keys past the tile&apos;s last row: zeros</Label>
          <Label x={px} y={350}>{`a thread block is ${warps} warps, one head each`}</Label>
        </>
      )}
    </Frame>
  );
}

export function SquareFigure({ tile, perRow, tiled, tokens, ms }: { tile: number; perRow: number; tiled: number; tokens: string; ms: number }) {
  const { grow } = useLoop(14);
  const n = 16;
  const c = 10;
  const y0 = 62;
  const left = 30;
  const right = 300;
  const band = 4;
  return (
    <Frame label={`With the decode attention every prompt row walks its own context alone, so the key rows read grow with the square of the prompt; with tiles, ${tile} rows share one pass over the keys`}>
      <Label x={left} y={24} className="fill-text">
        one walk per row
      </Label>
      <Label x={left} y={42}>row t reads its t + 1 keys alone</Label>
      <Label x={right} y={24} className="fill-accent">
        one pass per tile of rows
      </Label>
      <Label x={right} y={42}>{`${tile} rows share each tile of keys`}</Label>
      {Array.from({ length: n }, (_, t) => (
        <g key={t}>
          <rect x={left} y={y0 + t * c} width={(t + 1) * c} height={c - 2} rx={1} className="fill-border" />
          <motion.rect x={left} y={y0 + t * c} width={(t + 1) * c} height={c - 2} rx={1} className="fill-baseline" {...grow(0.03 + t * 0.052, 0.03 + (t + 1) * 0.052)} />
        </g>
      ))}
      {Array.from({ length: n / band }, (_, b) => (
        <g key={b}>
          {Array.from({ length: band }, (_, k) => (
            <rect key={k} x={right} y={y0 + (b * band + k) * c} width={(b * band + k + 1) * c} height={c - 2} rx={1} className="fill-border" />
          ))}
          <motion.g {...grow(0.03 + b * 0.208, 0.03 + (b + 1) * 0.208)}>
            {Array.from({ length: band }, (_, k) => (
              <rect key={k} x={right} y={y0 + (b * band + k) * c} width={(b * band + k + 1) * c} height={c - 2} rx={1} className="fill-accent" />
            ))}
          </motion.g>
          <rect x={right - 3} y={y0 + b * band * c - 2} width={(b + 1) * band * c + 6} height={band * c + 2} rx={2} className="fill-none stroke-accent/50" />
        </g>
      ))}
      <Label x={left} y={y0 + n * c + 18}>16 rows drawn, one bar = the keys a row reads</Label>
      <Label x={right} y={y0 + n * c + 18}>4 tiles drawn, each bar group fills in one pass</Label>
      <Label x={left} y={270} className="fill-text">{`key rows read per head, ${tokens}-token prompt (calculated)`}</Label>
      <Label x={left} y={292}>one walk per row</Label>
      <rect x={left + 130} y={282} width={300} height={12} rx={2} className="fill-border" />
      <motion.rect x={left + 130} y={282} width={300} height={12} rx={2} className="fill-baseline" {...grow(0.1, 0.5)} />
      <Label x={left + 438} y={292}>{fmt(perRow)}</Label>
      <Label x={left} y={314} className="fill-accent">{`one pass per ${tile} rows`}</Label>
      <rect x={left + 130} y={304} width={300} height={12} rx={2} className="fill-border" />
      <motion.rect x={left + 130} y={304} width={(300 * tiled) / perRow} height={12} rx={2} className="fill-accent" {...grow(0.1, 0.1 + (0.4 * tiled) / perRow)} />
      <Label x={left + 438} y={314} className="fill-accent">
        {fmt(tiled)}
      </Label>
      <Label x={left} y={346}>the score and value products are the same number either way;</Label>
      <Label x={left} y={362}>what changes is how often each key and value is fetched</Label>
      <Label x={left} y={388} className="fill-text">{`one walk per row, ${tokens} tokens: ${fmt(ms)} ms to the first token (measured)`}</Label>
    </Frame>
  );
}

export type SpecGroup = { title: string; lines: string[] };

export function SpecFigure({ tests, groups }: { tests: number; groups: SpecGroup[] }) {
  const { on } = useLoop(14);
  return (
    <Frame label={`The ${tests} tests written before the kernel pin what it must match, what must be bit for bit the same, what it must not read and which numeric cases it must survive`}>
      <Label x={20} y={24} className="fill-text">{`${tests} tests, written before any kernel`}</Label>
      {groups.map((g, i) => {
        const x = 20 + (i % 2) * 266;
        const y = 40 + Math.floor(i / 2) * 136;
        return (
          <g key={g.title}>
            <rect x={x} y={y} width={254} height={124} rx={3} className="fill-none stroke-border" />
            <motion.rect x={x} y={y} width={254} height={124} rx={3} className="fill-accent/10 stroke-accent" {...on(0.04 + i * 0.2, 0.04 + (i + 1) * 0.2, false)} />
            <Label x={x + 12} y={y + 22} className="fill-accent">
              {g.title}
            </Label>
            {g.lines.map((line, k) => (
              <Label key={line} x={x + 12} y={y + 44 + k * 17}>
                {line}
              </Label>
            ))}
          </g>
        );
      })}
      {["the stub: every test fails", "the first kernel: all pass", "the tiled kernel: all pass, unchanged"].map((text, i) => (
        <g key={text}>
          <circle cx={28 + i * 178} cy={340} r={5} className={i === 0 ? "fill-border" : "fill-accent"} />
          {i < 2 && <line x1={36 + i * 178} y1={340} x2={198 + i * 178} y2={340} className="stroke-border" />}
          <Label x={22 + i * 178} y={362} className={i === 0 ? "fill-muted" : "fill-text"}>
            {text}
          </Label>
        </g>
      ))}
      <Label x={20} y={388}>the same file judged both kernels: a rewrite could not change an answer</Label>
    </Frame>
  );
}

export function FirstKernelFigure({ paged, first, tokens, loss }: { paged: number; first: number; tokens: string; loss: number }) {
  const { on, grow } = useLoop(12);
  const keys = 20;
  const kw = 21;
  const x0 = 30;
  const y = 176;
  return (
    <Frame label={`The first kernel read the chunk's keys and values from the QKV buffer instead of the paged cache but walked them the same way, one key at a time per row and head, and measured ${loss.toFixed(0)}% slower`}>
      <Label x={20} y={24} className="fill-text">
        what changed: where the bytes come from
      </Label>
      <rect x={30} y={38} width={236} height={56} rx={3} className="fill-none stroke-border" />
      <Label x={42} y={60}>the paged KV cache,</Label>
      <Label x={42} y={78}>through the block table</Label>
      <rect x={290} y={38} width={236} height={56} rx={3} className="fill-accent/10 stroke-accent" />
      <Label x={302} y={60} className="fill-accent">
        the QKV buffer of this step:
      </Label>
      <Label x={302} y={78}>the chunk&apos;s own keys and values</Label>
      <line x1={148} y1={94} x2={148} y2={122} className="stroke-border" strokeDasharray="3 3" />
      <line x1={408} y1={94} x2={408} y2={122} className="stroke-accent" />
      <Label x={20} y={140} className="fill-text">
        what did not: the walk
      </Label>
      <Label x={20} y={158}>one thread block per row and head, one key at a time</Label>
      {Array.from({ length: keys }, (_, i) => (
        <g key={i}>
          <rect x={x0 + i * kw} y={y} width={kw - 5} height={20} rx={2} className="fill-surface stroke-border" />
          <motion.rect x={x0 + i * kw} y={y} width={kw - 5} height={20} rx={2} className="fill-baseline" {...on(0.03 + i * 0.03, 0.94)} />
          <line x1={x0 + i * kw + kw - 3.5} y1={y - 3} x2={x0 + i * kw + kw - 3.5} y2={y + 23} className="stroke-warn" />
          <line x1={x0 + i * kw + kw - 1.5} y1={y - 3} x2={x0 + i * kw + kw - 1.5} y2={y + 23} className="stroke-warn" />
        </g>
      ))}
      <Label x={x0 + keys * kw + 6} y={y + 14}>…</Label>
      <Label x={20} y={218} className="fill-warn">
        pink lines: two barriers after every key, the whole block waits
      </Label>
      <Label x={20} y={252} className="fill-text">{`one ${tokens}-token prompt, ms to the first token (measured)`}</Label>
      <Label x={20} y={278}>decode kernels</Label>
      <rect x={130} y={268} width={320} height={12} rx={2} className="fill-border" />
      <motion.rect x={130} y={268} width={(320 * paged) / first} height={12} rx={2} className="fill-baseline" {...grow(0.5, 0.5 + (0.3 * paged) / first)} />
      <Label x={458} y={278}>{`${fmt(paged)} ms`}</Label>
      <Label x={20} y={302} className="fill-warn">
        first kernel
      </Label>
      <rect x={130} y={292} width={320} height={12} rx={2} className="fill-border" />
      <motion.rect x={130} y={292} width={320} height={12} rx={2} className="fill-warn" {...grow(0.5, 0.8)} />
      <Label x={458} y={302} className="fill-warn">{`${fmt(first)} ms`}</Label>
      <Label x={20} y={340}>{`${loss.toFixed(0)}% slower (calculated): every row still walks every key alone,`}</Label>
      <Label x={20} y={356}>so a shorter path to each byte saves nothing</Label>
    </Frame>
  );
}

function LaneCells({ x, y, cell, rows, cols, on, className = "fill-accent" }: { x: number; y: number; cell: number; rows: number[]; cols: number[]; on: object; className?: string }) {
  return (
    <motion.g {...on}>
      {rows.flatMap((r) => cols.map((c) => <rect key={`${r}-${c}`} x={x + c * cell + 0.5} y={y + r * cell + 0.5} width={cell - 1} height={cell - 1} className={className} />))}
    </motion.g>
  );
}

export function MmaFigure({ tile, half, calls, headDim }: { tile: number; half: number; calls: number; headDim: number }) {
  const { on } = useLoop(18);
  const c = 10;
  const ax = 30;
  const ay = 208;
  const bx = ax + tile * c + 14;
  const by = 30;
  const tx = 318;
  return (
    <Frame label={`One mma call multiplies a ${tile} by ${tile} fragment of the query tile with a ${tile} by ${half} fragment of the keys and adds the result to ${tile} by ${half} scores; every lane of the warp holds a fixed few elements of each`}>
      <path d={gridPath(ax, ay, tile, tile, c)} className="fill-none stroke-border" />
      <path d={gridPath(bx, by, half, tile, c)} className="fill-none stroke-border" />
      <path d={gridPath(bx, ay, half, tile, c)} className="fill-none stroke-accent/60" />
      <Label x={ax} y={ay - 26} className="fill-text">A: the Q tile</Label>
      <Label x={ax} y={ay - 10}>{`${tile} query rows × ${tile} dims`}</Label>
      <Label x={bx - 12} y={by + 70} anchor="end" className="fill-text">B: half a key tile</Label>
      <Label x={bx - 12} y={by + 86} anchor="end">{`${tile} dims × ${half} keys`}</Label>
      <Label x={bx} y={ay + tile * c + 16} className="fill-accent">C: scores</Label>
      <Label x={ax} y={ay + tile * c + 16}>one key per column of B and C →</Label>
      {shownLanes.map((lane, i) => {
        const f = fragmentCells(lane);
        const [from, to] = laneWindow(i);
        const win = on(from, to, i === 1);
        return (
          <g key={lane}>
            <LaneCells x={ax} y={ay} cell={c} rows={f.rows} cols={f.pairs} on={win} className="fill-baseline" />
            <LaneCells x={bx} y={by} cell={c} rows={f.pairs} cols={[f.group]} on={win} className="fill-baseline" />
            <LaneCells x={bx} y={ay} cell={c} rows={f.rows} cols={f.pairs.slice(0, 2)} on={win} />
            <motion.g {...win}>
              <Label x={tx} y={150} className="fill-accent">{`lane ${lane} of 32 holds`}</Label>
              <Label x={tx} y={170}>{`A: rows ${f.rows[0]} and ${f.rows[1]}, dims ${f.pairs.join(", ")}`}</Label>
              <Label x={tx} y={188}>{`B: key ${f.group}, the same 4 dims`}</Label>
              <Label x={tx} y={206}>{`C: rows ${f.rows[0]} and ${f.rows[1]} × keys ${f.pairs[0]}, ${f.pairs[1]}`}</Label>
            </motion.g>
          </g>
        );
      })}
      <Label x={tx} y={44} className="fill-text">
        mma.sync.aligned.m16n8k16
      </Label>
      <Label x={tx} y={66}>C += A × B on the tensor cores</Label>
      <Label x={tx} y={84}>A and B in BF16, C in FP32</Label>
      <Label x={tx} y={102}>one instruction, the whole warp</Label>
      <Label x={tx} y={240} className="fill-text">
        per lane, in registers
      </Label>
      <Label x={tx} y={260}>A: 8 values, 4 words</Label>
      <Label x={tx} y={278}>B: 4 values, 2 words</Label>
      <Label x={tx} y={296}>C: 4 scores</Label>
      <Label x={tx} y={330}>{`${calls} calls cover the ${headDim} dims of a head`}</Label>
      <Label x={tx} y={348}>{`2 halves cover the ${tile} keys of a tile`}</Label>
    </Frame>
  );
}

export function FragmentFigure({ tile, lanesPerRow }: { tile: number; lanesPerRow: number }) {
  const { on } = useLoop(18);
  const c = 9;
  const y0 = 56;
  const sx = 30;
  const px = 386;
  const by = 268;
  const bw = 112;
  const bx = (i: number) => 30 + i * 130;
  const arc = (a: number, b: number, lift: number) => `M${bx(a) + bw / 2} ${by} Q${(bx(a) + bx(b) + bw) / 2} ${by - lift} ${bx(b) + bw / 2} ${by}`;
  return (
    <Frame label={`After both halves a lane holds the scores of its two query rows against four keys, exactly the elements the P times V product wants from it, so scores become weights in place; a row's maximum and sum are reduced over the ${lanesPerRow} lanes that hold it`}>
      <Label x={sx} y={24} className="fill-text">
        scores of one key tile
      </Label>
      <Label x={sx} y={42}>{`${tile} rows × ${tile} keys`}</Label>
      <Label x={px} y={24} className="fill-accent">
        P, first operand of P·V
      </Label>
      <Label x={px} y={42}>the same shape, in BF16</Label>
      <path d={gridPath(sx, y0, tile, tile, c)} className="fill-none stroke-border" />
      <path d={gridPath(px, y0, tile, tile, c)} className="fill-none stroke-accent/60" />
      {shownLanes.map((lane, i) => {
        const f = fragmentCells(lane);
        const [from, to] = laneWindow(i);
        const win = on(from, to, i === 1);
        return (
          <g key={lane}>
            <LaneCells x={sx} y={y0} cell={c} rows={f.rows} cols={f.pairs} on={win} className="fill-baseline" />
            <LaneCells x={px} y={y0} cell={c} rows={f.rows} cols={f.pairs} on={win} />
            <motion.g {...win}>
              <Label x={280} y={74} anchor="middle" className="fill-accent">{`lane ${lane}: rows ${f.rows[0]}, ${f.rows[1]}`}</Label>
              <Label x={280} y={92} anchor="middle">{`keys ${f.pairs.join(", ")}`}</Label>
            </motion.g>
          </g>
        );
      })}
      <line x1={186} y1={128} x2={374} y2={128} className="stroke-accent" />
      <path d="M374 128l-7 -4v8z" className="fill-accent" />
      <Label x={280} y={120} anchor="middle" className="fill-text">
        same lane, same cells
      </Label>
      <Label x={280} y={148} anchor="middle">round, scale, mask</Label>
      <Label x={280} y={166} anchor="middle">w = e^(s − m), rounded to BF16</Label>
      <Label x={280} y={184} anchor="middle" className="fill-accent">
        nothing moves between lanes
      </Label>
      <Label x={30} y={226} className="fill-text">{`one query row is spread over ${lanesPerRow} lanes: its max and sum take two shuffles`}</Label>
      {Array.from({ length: lanesPerRow }, (_, q) => (
        <g key={q}>
          <rect x={bx(q)} y={by} width={bw} height={34} rx={3} className="fill-surface stroke-border" />
          <Label x={bx(q) + bw / 2} y={by + 15} anchor="middle" className="fill-text">{`lane ${q}`}</Label>
          <Label x={bx(q) + bw / 2} y={by + 29} anchor="middle">{`keys ${[2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9].join(", ")}`}</Label>
        </g>
      ))}
      <motion.g {...on(0.1, 0.5, false)}>
        <path d={arc(0, 1, 26)} className="fill-none stroke-accent" />
        <path d={arc(2, 3, 26)} className="fill-none stroke-accent" />
        <Label x={30} y={324} className="fill-accent">shuffle 1: each lane takes its neighbour&apos;s value (xor 1)</Label>
      </motion.g>
      <motion.g {...on(0.5, 0.94)}>
        <path d={arc(0, 2, 34)} className="fill-none stroke-accent" />
        <path d={arc(1, 3, 34)} className="fill-none stroke-accent" />
        <Label x={30} y={324} className="fill-accent">shuffle 2: the pair two lanes away (xor 2): all 4 hold the row&apos;s result</Label>
      </motion.g>
      <Label x={30} y={354}>then, per row: m′ = max(m, tile max), l′ = l·e^(m − m′) + Σ w,</Label>
      <Label x={30} y={372}>and acc is scaled by e^(m − m′) before P·V is added</Label>
    </Frame>
  );
}

export function RegistersFigure({ headDim, qWords, accs, group }: { headDim: number; qWords: number; accs: number; group: number }) {
  const { on } = useLoop(16);
  const cols = headDim / 8;
  const vw = 7.5;
  const vx = 40;
  const vy = 254;
  const mine = 2;
  return (
    <Frame label={`A lane keeps ${qWords} words of the query tile and ${accs} FP32 accumulators in registers; the value columns are interleaved so the 8 values it needs from one key sit side by side`}>
      <Label x={20} y={24} className="fill-text">{`what one lane keeps in registers (head_dim ${headDim})`}</Label>
      <Label x={20} y={52}>its piece of the Q tile</Label>
      {Array.from({ length: qWords }, (_, i) => (
        <rect key={i} x={190 + i * 11} y={42} width={9} height={12} rx={1} className="fill-baseline" />
      ))}
      <Label x={190 + qWords * 11 + 8} y={52}>{`${qWords} words`}</Label>
      <Label x={20} y={80} className="fill-accent">
        acc, the running Σ w · v
      </Label>
      {Array.from({ length: accs }, (_, i) => (
        <rect key={i} x={190 + (i % 16) * 11} y={70 + Math.floor(i / 16) * 15} width={9} height={12} rx={1} className="fill-accent" />
      ))}
      <Label x={190 + 16 * 11 + 8} y={88} className="fill-accent">{`${accs} × FP32`}</Label>
      <Label x={20} y={124}>m and l of its 2 rows</Label>
      {Array.from({ length: 4 }, (_, i) => (
        <rect key={i} x={190 + i * 11} y={114} width={9} height={12} rx={1} className="fill-warn" />
      ))}
      <Label x={20} y={158} className="fill-text">
        shared memory: none. barriers: none. warps share nothing.
      </Label>
      <Label x={20} y={206} className="fill-text">{`one key's value row, ${headDim} dims`}</Label>
      <Label x={20} y={224}>{`P·V call n writes output dims n, ${cols} + n, ${2 * cols} + n … one per group`}</Label>
      {Array.from({ length: headDim }, (_, d) => (
        <rect key={d} x={vx + d * vw} y={vy} width={vw - 1.5} height={18} rx={1} className={Math.floor(d / cols) === mine ? "fill-accent/50" : Math.floor(d / cols) % 2 ? "fill-border" : "fill-baseline/50"} />
      ))}
      {Array.from({ length: cols }, (_, n) => (
        <motion.g key={n} {...on(0.04 + n * 0.11, 0.04 + (n + 1) * 0.11, n === 0)}>
          {Array.from({ length: group }, (_, g) => (
            <rect key={g} x={vx + (g * cols + n) * vw - 1} y={vy - 3} width={vw + 0.5} height={24} rx={1} className="fill-none stroke-accent" strokeWidth={1.5} />
          ))}
          <Label x={vx} y={vy + 40} className="fill-accent">{`call ${n}: dims ${Array.from({ length: 3 }, (_, g) => g * cols + n).join(", ")} … ${(group - 1) * cols + n}`}</Label>
        </motion.g>
      ))}
      <rect x={vx + mine * cols * vw - 2} y={vy - 6} width={cols * vw + 2.5} height={30} rx={2} className="fill-none stroke-text" strokeDasharray="3 2" />
      <Label x={vx} y={vy + 62}>{`dashed: the ${cols} values a lane of group ${mine} needs, side by side, loaded as 4 words`}</Label>
      <Label x={20} y={356}>template &lt;int HeadDim&gt;: the loops unroll at compile time,</Label>
      <Label x={20} y={372}>so these arrays are registers, not memory</Label>
    </Frame>
  );
}

export type TtftSeries = { label: string; values: number[]; tone: "baseline" | "warn" | "accent" | "muted" };

const tones = { baseline: { stroke: "stroke-baseline", fill: "fill-baseline", text: "fill-muted" }, warn: { stroke: "stroke-warn", fill: "fill-warn", text: "fill-warn" }, accent: { stroke: "stroke-accent", fill: "fill-accent", text: "fill-accent" }, muted: { stroke: "stroke-text", fill: "fill-text", text: "fill-text" } };

export function TtftFigure({ lengths, series }: { lengths: number[]; series: TtftSeries[] }) {
  const { on } = useLoop(14);
  const x0 = 58;
  const x1 = 330;
  const y0 = 330;
  const y1 = 50;
  const lo = 8;
  const hi = 2400;
  const X = (n: number) => x0 + ((x1 - x0) * Math.log(n / lengths[0])) / Math.log(lengths[lengths.length - 1] / lengths[0]);
  const Y = (ms: number) => y0 - ((y0 - y1) * Math.log(ms / lo)) / Math.log(hi / lo);
  const last = lengths.length - 1;
  return (
    <Frame label={`Time to first token against prompt length for ${series.map((s) => s.label).join(", ")}: ${series.map((s) => `${s.label} ${fmt(s.values[last])} ms`).join(", ")} at ${fmt(lengths[last])} tokens`}>
      <Label x={20} y={24} className="fill-text">
        ms to the first token, one prompt alone (measured)
      </Label>
      <line x1={x0} y1={y0} x2={x1 + 8} y2={y0} className="stroke-border" />
      <line x1={x0} y1={y0} x2={x0} y2={y1 - 8} className="stroke-border" />
      {[10, 100, 1000].map((ms) => (
        <g key={ms}>
          <line x1={x0} y1={Y(ms)} x2={x1 + 8} y2={Y(ms)} className="stroke-border" strokeDasharray="2 4" />
          <Label x={x0 - 6} y={Y(ms) + 4} anchor="end">
            {fmt(ms)}
          </Label>
        </g>
      ))}
      {lengths.map((n) => (
        <g key={n}>
          <line x1={X(n)} y1={y0} x2={X(n)} y2={y0 + 5} className="stroke-border" />
          <Label x={X(n)} y={y0 + 18} anchor="middle">
            {fmt(n)}
          </Label>
        </g>
      ))}
      <Label x={x1 + 8} y={y0 + 36} anchor="end">
        prompt tokens, both axes logarithmic
      </Label>
      {series.map((s, i) => (
        <motion.g key={s.label} {...on(0.04 + i * 0.14)}>
          <polyline points={s.values.map((v, k) => `${X(lengths[k])},${Y(v)}`).join(" ")} className={`fill-none ${tones[s.tone].stroke}`} strokeWidth={s.tone === "accent" ? 2.5 : 1.8} strokeDasharray={s.tone === "muted" ? "5 4" : undefined} />
          {s.values.map((v, k) => (
            <circle key={k} cx={X(lengths[k])} cy={Y(v)} r={s.tone === "accent" ? 3.5 : 2.5} className={tones[s.tone].fill} />
          ))}
        </motion.g>
      ))}
      <Label x={356} y={62} className="fill-text">{`at ${fmt(lengths[last])} tokens`}</Label>
      {series.map((s, i) => (
        <motion.g key={s.label} {...on(0.04 + i * 0.14)}>
          <line x1={356} y1={88 + i * 44} x2={380} y2={88 + i * 44} className={tones[s.tone].stroke} strokeWidth={s.tone === "accent" ? 2.5 : 1.8} strokeDasharray={s.tone === "muted" ? "5 4" : undefined} />
          <Label x={388} y={92 + i * 44} className={tones[s.tone].text}>
            {s.label}
          </Label>
          <Label x={388} y={108 + i * 44} className={tones[s.tone].text}>{`${fmt(s.values[last])} ms`}</Label>
        </motion.g>
      ))}
      <Label x={356} y={282}>a straight line that climbs</Label>
      <Label x={356} y={298}>one step per doubling has a</Label>
      <Label x={356} y={314}>flat cost per token</Label>
    </Frame>
  );
}

type OpenParts = { attention: number; gemms: number; rest: number; span: number; blocks: number };

export function OpenFigure({ whole, chunked, atOnce, registers, budget, chunks, tokens, items }: { whole: OpenParts; chunked: OpenParts; atOnce: number; registers: number; budget: string; chunks: number; tokens: string; items: string[] }) {
  const { on, grow } = useLoop(12);
  const px = 320 / chunked.span;
  const rows = [
    { label: "one step", parts: whole, y: 42, tone: "accent", end: 0.04 + (0.3 * whole.span) / chunked.span },
    { label: `${chunks} chunks of ${budget}`, parts: chunked, y: 66, tone: "warn", end: 0.34 },
  ] as const;
  return (
    <Frame label={`One ${tokens}-token prompt alone, kernel time from Nsight Systems: in one step ${whole.span} ms, of which attention ${whole.attention} and the GEMMs ${whole.gemms}; as ${chunks} chunks of ${budget} ${chunked.span} ms, of which attention ${chunked.attention} and the GEMMs ${chunked.gemms}; ${atOnce} thread blocks fit on the GPU at once and a chunk launches ${chunked.blocks}. Still open: ${items.join("; ")}`}>
      <Label x={20} y={24} className="fill-text">{`one ${tokens}-token prompt alone: where the ms go (measured, Nsight Systems)`}</Label>
      {rows.map((r) => (
        <g key={r.label}>
          <Label x={20} y={r.y + 10} className={r.tone === "warn" ? "fill-warn" : undefined}>
            {r.label}
          </Label>
          <rect x={130} y={r.y} width={320} height={12} rx={2} className="fill-border" />
          <motion.g {...grow(0.04, r.end)}>
            <rect x={130} y={r.y} width={r.parts.attention * px} height={12} rx={2} className={r.tone === "warn" ? "fill-warn" : "fill-accent"} />
            <rect x={130 + r.parts.attention * px + 1} y={r.y} width={r.parts.gemms * px - 1} height={12} className="fill-muted/60" />
            <rect x={130 + (r.parts.attention + r.parts.gemms) * px + 1} y={r.y} width={r.parts.rest * px - 1} height={12} rx={2} className="fill-muted/30" />
          </motion.g>
          <Label x={458} y={r.y + 10} className={r.tone === "warn" ? "fill-warn" : "fill-accent"}>{`${fmt(r.parts.span)} ms`}</Label>
        </g>
      ))}
      <Label x={20} y={102}>
        <tspan className="fill-warn">{`attention ${whole.attention.toFixed(1)} → ${chunked.attention.toFixed(1)} ms`}</tspan>
        {` · GEMMs ${whole.gemms.toFixed(1)} → ${chunked.gemms.toFixed(1)} · the rest ${whole.rest.toFixed(1)} → ${chunked.rest.toFixed(1)}`}
      </Label>
      <Label x={20} y={120}>{`${atOnce} thread blocks fit on the GPU at once (calculated, ${registers} registers per thread);`}</Label>
      <Label x={20} y={136}>{`a ${budget}-row chunk launches ${chunked.blocks}, so ${chunked.blocks - atOnce} run after the others`}</Label>
      <line x1={20} y1={154} x2={540} y2={154} className="stroke-border" />
      <Label x={20} y={178} className="fill-text">
        not done yet
      </Label>
      {items.map((item, i) => (
        <motion.g key={item} {...on(0.3 + i * 0.12)}>
          <rect x={22} y={196 + i * 50} width={12} height={12} rx={2} className="fill-none stroke-warn" />
          {item.split("\n").map((line, k) => (
            <Label key={line} x={46} y={206 + i * 50 + k * 16} className={k === 0 ? "fill-text" : "fill-muted"}>
              {line}
            </Label>
          ))}
        </motion.g>
      ))}
    </Frame>
  );
}
