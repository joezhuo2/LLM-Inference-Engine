"use client";

import { motion } from "motion/react";
import trace from "@/content/kernels/cuda-graphs-trace.json";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

export type Kernel = { op: string; start: number; end: number };

export const eagerLayer: Kernel[] = trace.eager.kernels_in_layer;
export const graphLayer: Kernel[] = trace.graphs.kernels_in_layer;
export const layerIndex = trace.eager.layer;

const short: Record<string, string> = {
  add2_norm1: "add+norm",
  qkv: "QKV",
  rope: "rope",
  write_kv: "write_kv",
  attention: "attn",
  o_proj: "O",
  add1_norm2: "add+norm",
  gate_up: "gate/up",
  swiglu: "swiglu",
  down: "down",
};

const gemm = new Set(["qkv", "o_proj", "gate_up", "down"]);

const x0 = 20;
const lane = 520;

export function LaneLayer({ kernels, total, y, sweep, showGaps }: { kernels: Kernel[]; total: number; y: number; sweep?: [number, number]; showGaps?: boolean }) {
  const { grow, on } = useLoop(11);
  const px = (t: number) => x0 + (lane * t) / total;
  const at = (t: number) => (sweep ? sweep[0] + ((sweep[1] - sweep[0]) * t) / total : 0);
  return (
    <g>
      {kernels.map((k, i) => {
        const next = kernels[i + 1];
        return (
          <g key={`${k.op}${i}`}>
            <motion.rect
              x={px(k.start)}
              y={y}
              width={Math.max(1.5, px(k.end) - px(k.start))}
              height={22}
              rx={2}
              className={gemm.has(k.op) ? "fill-baseline" : "fill-accent"}
              {...(sweep ? grow(at(k.start), at(k.end)) : {})}
            />
            {showGaps && next && (
              <motion.rect x={px(k.end)} y={y + 26} width={Math.max(1.5, px(next.start) - px(k.end))} height={5} className="fill-warn" {...(sweep ? on(at(next.start)) : {})} />
            )}
          </g>
        );
      })}
    </g>
  );
}

function LayerLabels({ kernels, total, y }: { kernels: Kernel[]; total: number; y: number }) {
  const px = (t: number) => x0 + (lane * t) / total;
  const big = kernels.filter((k) => gemm.has(k.op) || k.op === "attention");
  return (
    <g>
      {big.map((k) => (
        <Label key={k.op} x={(px(k.start) + px(k.end)) / 2} y={y} anchor="middle" className={gemm.has(k.op) ? "fill-text" : "fill-accent"}>
          {short[k.op]}
        </Label>
      ))}
    </g>
  );
}

const fmt = (v: number, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

export function EagerTimelineFigure({ idlePct, kernels, spanUs, busyUs }: { idlePct: number; kernels: number; spanUs: number; busyUs: number }) {
  const { on, reduced } = useLoop(11);
  const total = Math.ceil(eagerLayer[eagerLayer.length - 1].end / 10) * 10 + 10;
  const cpu = trace.eager.cpu_in_layer;
  const sweep: [number, number] = [0.04, 0.78];
  const px = (t: number) => x0 + (lane * t) / total;
  const at = (t: number) => sweep[0] + ((sweep[1] - sweep[0]) * Math.max(0, t)) / total;
  return (
    <Frame label={`Nsight Systems timeline of layer ${layerIndex} in an eager batch-1 decode step: the CPU lane is a row of back-to-back cudaLaunchKernel calls for kernels further ahead, the GPU lane runs the ten kernels of the layer with a gap of a few microseconds before each one`}>
      <Label x={20} y={24}>{`layer ${layerIndex} of 22 in an eager decode step, from Nsight Systems`}</Label>
      <Label x={20} y={58} className="fill-text">CPU</Label>
      <Label x={60} y={58}>{`cudaLaunchKernel, ${fmt(trace.eager.launches.cudaLaunchKernel.median_us)} µs each, for kernels ${fmt(trace.eager.cpu_ahead_us / 1000, 1)} ms ahead`}</Label>
      {cpu.map((c, i) => (
        <motion.rect
          key={i}
          x={Math.max(x0, px(c.start))}
          y={68}
          width={Math.max(1.5, px(Math.min(c.end, total)) - Math.max(x0, px(c.start)) - 1)}
          height={14}
          rx={2}
          className="fill-text/60"
          {...on(at(c.start))}
        />
      ))}
      <Label x={20} y={118} className="fill-text">GPU</Label>
      <Label x={60} y={118}>{"the layer's ten kernels, in stream order"}</Label>
      <LayerLabels kernels={eagerLayer} total={total} y={140} />
      <LaneLayer kernels={eagerLayer} total={total} y={148} sweep={sweep} showGaps />
      {!reduced && (
        <motion.line
          x1={x0}
          x2={x0}
          y1={60}
          y2={190}
          className="stroke-text"
          strokeWidth={1}
          initial={{ x: 0, opacity: 0 }}
          animate={{ x: [0, 0, lane, lane, lane], opacity: [0, 1, 1, 0, 0] }}
          transition={{ duration: 11, repeat: Infinity, ease: "linear", times: [0, sweep[0], sweep[1], sweep[1] + 0.03, 1] }}
        />
      )}
      {[0, 100, 200, 300].map((t) => (
        <Label key={t} x={px(t)} y={206} anchor={t === 0 ? undefined : "middle"}>
          {`${t} µs`}
        </Label>
      ))}
      <rect x={20} y={226} width={10} height={10} rx={2} className="fill-baseline" />
      <Label x={36} y={235}>cuBLAS GEMV</Label>
      <rect x={140} y={226} width={10} height={10} rx={2} className="fill-accent" />
      <Label x={156} y={235}>the engine&apos;s kernels</Label>
      <rect x={300} y={230} width={10} height={5} className="fill-warn" />
      <Label x={316} y={235} className="fill-warn">GPU idle</Label>
      <motion.g {...on(0.8)}>
        <Label x={20} y={290} className="fill-text">{`${kernels} kernels per step, each its own launch`}</Label>
        <Label x={20} y={308}>{`the CPU is never what the GPU waits for here, yet the step spans`}</Label>
        <Label x={20} y={324}>{`${fmt(spanUs)} µs and its kernels are busy for ${fmt(busyUs)} µs:`}</Label>
        <Label x={20} y={346} className="fill-warn">{`the GPU is idle for ${idlePct}% of a decode step`}</Label>
      </motion.g>
    </Frame>
  );
}

export function GapFigure({ boundaries, gapsOver, idleUs }: { boundaries: number; gapsOver: number; idleUs: number }) {
  const { grow, on } = useLoop(10);
  const ops = eagerLayer.filter((k) => ["rope", "write_kv", "attention"].includes(k.op));
  const t0 = ops[0].start - 1.5;
  const t1 = ops[2].end + 1;
  const px = (t: number) => x0 + (lane * (t - t0)) / (t1 - t0);
  const slot = 0.22;
  const g = trace.eager.gap_us;
  const r = trace.graphs.gap_us;
  const dx = (v: number) => 70 + v * 80;
  return (
    <Frame label="Zoom into three kernels of the eager layer: rope, write_kv and attention, each separated from the next by about three microseconds in which the GPU runs nothing; below, the spread of those gaps over every boundary of every decode step, eager against a graph replay">
      <Label x={20} y={24}>{`zoom: ${fmt(t1 - t0)} µs of the same layer`}</Label>
      {ops.map((k, i) => {
        const s = 0.05 + i * slot;
        const next = ops[i + 1];
        return (
          <g key={k.op}>
            <Label x={(px(k.start) + px(k.end)) / 2} y={60} anchor="middle" className="fill-accent">
              {k.op}
            </Label>
            <motion.rect x={px(k.start)} y={70} width={px(k.end) - px(k.start)} height={30} rx={3} className="fill-accent" {...grow(s, s + slot * 0.6)} />
            <Label x={(px(k.start) + px(k.end)) / 2} y={118} anchor="middle">{`${fmt(k.end - k.start, 1)} µs`}</Label>
            {next && (
              <motion.g {...on(s + slot * 0.6)}>
                <rect x={px(k.end)} y={70} width={px(next.start) - px(k.end)} height={30} className="fill-warn/25" />
                <line x1={px(k.end)} x2={px(next.start)} y1={140} y2={140} className="stroke-warn" />
                <line x1={px(k.end)} x2={px(k.end)} y1={134} y2={146} className="stroke-warn" />
                <line x1={px(next.start)} x2={px(next.start)} y1={134} y2={146} className="stroke-warn" />
                <Label x={(px(k.end) + px(next.start)) / 2} y={160} anchor="middle" className="fill-warn">{`${fmt(next.start - k.end, 1)} µs idle`}</Label>
              </motion.g>
            )}
          </g>
        );
      })}
      <Label x={20} y={204} className="fill-text">{`the gap before every kernel, all ${fmt(trace.eager.steps)} decode steps of the profile`}</Label>
      {[0, 1, 2, 3, 4, 5].map((v) => (
        <g key={v}>
          <line x1={dx(v)} x2={dx(v)} y1={218} y2={282} className="stroke-border" strokeWidth={0.5} />
          <Label x={dx(v)} y={298} anchor="middle">{`${v} µs`}</Label>
        </g>
      ))}
      {[
        { label: "eager", q: g, y: 230, cls: "fill-warn" },
        { label: "graph", q: r, y: 260, cls: "fill-accent" },
      ].map((row) => (
        <g key={row.label}>
          <Label x={20} y={row.y + 9} className={row.label === "eager" ? "fill-warn" : "fill-accent"}>
            {row.label}
          </Label>
          <rect x={dx(row.q.p10)} y={row.y} width={Math.max(2, dx(row.q.p90) - dx(row.q.p10))} height={12} rx={3} className={row.cls} opacity={0.35} />
          <rect x={dx(row.q.p50) - 1} y={row.y - 2} width={2.5} height={16} className={row.cls} />
          <Label x={dx(row.q.p90) + 8} y={row.y + 9} className={row.label === "eager" ? "fill-warn" : "fill-accent"}>{`median ${row.q.p50} µs`}</Label>
        </g>
      ))}
      <motion.g {...on(0.72)}>
        <Label x={20} y={330}>{`bars: the middle 80% of gaps. ${boundaries} boundaries per step, most near 3 µs`}</Label>
        <Label x={20} y={346}>{`and ${gapsOver} past 5 µs, add up to ${fmt(idleUs)} µs of idle GPU per step`}</Label>
      </motion.g>
    </Frame>
  );
}

const gemmNode = (n: string) => ["QKV", "O", "gate/up", "down"].includes(n);

const nodes = ["add+norm", "QKV", "rope", "write_kv", "attn", "O", "add+norm", "gate/up", "swiglu", "down"];

export function CaptureFigure({ kernels }: { kernels: number }) {
  const { on, move } = useLoop(12);
  const code = [
    { text: "run_staged(b, b);", note: "eager, so cuBLAS picks its kernels", w: [0.02, 0.18] },
    { text: "cudaStreamBeginCapture(stream, Global);", note: "", w: [0.18, 0.26] },
    { text: "run_staged(b, b);", note: "recorded, not run", w: [0.26, 0.62] },
    { text: "cudaStreamEndCapture(stream, &graph);", note: "", w: [0.62, 0.72] },
    { text: "cudaGraphInstantiate(&exec, graph, 0);", note: "", w: [0.72, 0.94] },
  ];
  const nx = (i: number) => 24 + (i % 5) * 104;
  const ny = (i: number) => 236 + Math.floor(i / 5) * 46;
  return (
    <Frame label="Stream capture: after one eager run, the same run_staged call is recorded between cudaStreamBeginCapture and cudaStreamEndCapture; its kernel launches become nodes of a graph instead of running, and the graph is instantiated once into an executable graph">
      <Label x={20} y={24}>{"capture(): what DecodeGraphs does once per graph, at startup"}</Label>
      {code.map((c, i) => (
        <g key={i}>
          <motion.rect x={16} y={38 + i * 24} width={300} height={20} rx={3} className="fill-accent-dim" {...on(c.w[0], c.w[1], i === 4)} />
          <Label x={24} y={52 + i * 24} className="fill-text">
            {c.text}
          </Label>
          <Label x={326} y={52 + i * 24}>{c.note}</Label>
        </g>
      ))}
      <Label x={20} y={180}>{"while capturing, each launch on the stream becomes a node:"}</Label>
      {nodes.map((n, i) => {
        const s = 0.28 + i * 0.032;
        return (
          <g key={i}>
            <rect x={nx(i)} y={ny(i)} width={88} height={26} rx={4} className="fill-surface stroke-border" strokeWidth={0.75} />
            <motion.g {...move(s, s + 0.03, nx(i) - 24, ny(i) - 196, true, 0.94)}>
              <rect x={24} y={196} width={88} height={26} rx={4} className={gemmNode(n) ? "fill-baseline" : "fill-accent"} />
              <text x={68} y={213} textAnchor="middle" className={`${gemmNode(n) ? "fill-text" : "fill-bg"} text-[11px]`}>
                {n}
              </text>
            </motion.g>
            {i % 5 !== 4 && <line x1={nx(i) + 88} x2={nx(i) + 104} y1={ny(i) + 13} y2={ny(i) + 13} className="stroke-border" />}
          </g>
        );
      })}
      <Label x={20} y={346}>{"one layer drawn; the graph holds all 22 and the LM head"}</Label>
      <motion.g {...on(0.74)}>
        <rect x={16} y={222} width={528} height={100} rx={8} className="fill-none stroke-accent" strokeWidth={1.25} />
        <Label x={20} y={372} className="fill-accent">{`instantiated once: ${kernels} kernels, launched with one cudaGraphLaunch`}</Label>
      </motion.g>
    </Frame>
  );
}

const fields = ["token", "position", "slot", "context", "block table"];

export function StageFigure() {
  const { on } = useLoop(12);
  const rows = [
    { name: "request A", v: ["417", "88", "1,432", "89", "12 4 31 …"], pad: false },
    { name: "request B", v: ["29", "7", "520", "8", "2 …"], pad: false },
    { name: "request C", v: ["1,863", "301", "4,925", "302", "8 9 17 …"], pad: false },
    { name: "padding", v: ["0", "0", "-1", "0", "0 0 0 …"], pad: true },
  ];
  const cx = (j: number) => 112 + j * 86;
  return (
    <Frame label="A decode batch of three requests padded to the bucket of four rows, then copied by stage into one device buffer whose five arrays sit at offsets fixed by the row count, where the captured kernels read them on every replay">
      <Label x={20} y={24}>{"stage(): 3 requests padded to the 4-row bucket, then copied to the GPU"}</Label>
      {fields.map((f, j) => (
        <Label key={f} x={cx(j)} y={50}>
          {f}
        </Label>
      ))}
      {rows.map((r, i) => (
        <motion.g key={r.name} {...(r.pad ? on(0.08) : {})}>
          <Label x={20} y={74 + i * 20} className={r.pad ? "fill-warn" : "fill-text"}>
            {r.name}
          </Label>
          {r.v.map((v, j) => (
            <Label key={j} x={cx(j)} y={74 + i * 20} className={r.pad ? "fill-warn" : "fill-text"}>
              {v}
            </Label>
          ))}
        </motion.g>
      ))}
      <motion.g {...on(0.1)}>
        <Label x={20} y={168} className="fill-warn">{"a padded row writes nowhere (slot -1) and reads nothing (context 0)"}</Label>
      </motion.g>
      <motion.g {...on(0.3)}>
        <Label x={20} y={204}>{"one device buffer of int32, at the same address on every step (t = 4 rows):"}</Label>
        {fields.map((f, j) => {
          const w = j === 4 ? 152 : 86;
          return (
            <g key={f}>
              <rect x={20 + j * 92} y={214} width={w} height={26} rx={3} className="fill-accent/25 stroke-accent" strokeWidth={0.75} />
              <Label x={26 + j * 92} y={231} className="fill-text">
                {j === 4 ? "4t: block tables" : `${j === 0 ? "0" : `${j}t`}: ${f}`}
              </Label>
            </g>
          );
        })}
      </motion.g>
      <motion.g {...on(0.5)}>
        {[
          { label: "embedding reads tokens", j: 0 },
          { label: "rope reads positions", j: 1 },
          { label: "write_kv reads slots", j: 2 },
          { label: "attention reads 3t and 4t", j: 3 },
        ].map((a, i) => (
          <g key={a.label}>
            <line x1={50 + a.j * 92} x2={50 + a.j * 92} y1={244} y2={262 + i * 18} className="stroke-accent" />
            <Label x={56 + a.j * 92} y={266 + i * 18} className="fill-accent">
              {a.label}
            </Label>
          </g>
        ))}
      </motion.g>
      <motion.g {...on(0.7)}>
        <Label x={20} y={350} className="fill-text">{"the graph stores every pointer and grid size it was captured with;"}</Label>
        <Label x={20} y={366} className="fill-text">{"a replay gets new values at the same addresses, never new addresses"}</Label>
      </motion.g>
    </Frame>
  );
}

const buckets = [1, 2, 4, 8, 16];
const splitCounts = [1, 2, 4, 8];
const captured: Record<number, number[]> = { 1: [1, 2, 4, 8], 2: [1, 2, 4], 4: [1, 2], 8: [1], 16: [1] };
const thresholds = [
  { context: 0, splits: 1 },
  { context: 512, splits: 2 },
  { context: 1024, splits: 4 },
  { context: 2048, splits: 8 },
];

export function GraphGridFigure({ before, after, eager }: { before: number; after: number; eager: number }) {
  const { on } = useLoop(12);
  const gx = (j: number) => 140 + j * 74;
  const gy = (i: number) => 64 + i * 34;
  const count = Object.values(captured).reduce((s, v) => s + v.length, 0);
  const max = Math.max(before, after, eager);
  return (
    <Frame label={`The ${count} captured graphs as a grid of batch-size buckets by attention split counts; a one-row request's context grows from 0 to 2,048 tokens and its replays move from the 1-split to the 2-, 4- and 8-split graph`}>
      <Label x={20} y={24}>{`${count} graphs: one per (bucket, split count) at a 2,048-token max context`}</Label>
      <Label x={20} y={50}>bucket</Label>
      {splitCounts.map((s, j) => (
        <Label key={s} x={gx(j) + 30} y={50} anchor="middle">{`${s} split${s > 1 ? "s" : ""}`}</Label>
      ))}
      {buckets.map((b, i) => (
        <g key={b}>
          <Label x={20} y={gy(i) + 17} className="fill-text">{`${b} row${b > 1 ? "s" : ""}`}</Label>
          {splitCounts.map((s, j) =>
            captured[b].includes(s) ? (
              <rect key={s} x={gx(j)} y={gy(i)} width={60} height={24} rx={4} className="fill-accent/30 stroke-accent" strokeWidth={0.75} />
            ) : (
              <rect key={s} x={gx(j)} y={gy(i)} width={60} height={24} rx={4} className="fill-none stroke-border" strokeDasharray="3 3" strokeWidth={0.75} />
            ),
          )}
        </g>
      ))}
      {thresholds.map((t, j) => {
        const s = 0.06 + j * 0.2;
        return <motion.rect key={t.context} x={gx(j)} y={gy(0)} width={60} height={24} rx={4} className="fill-accent" {...on(s, s + 0.2, j === 2)} />;
      })}
      <Label x={460} y={gy(0) + 10} className="fill-accent">one request,</Label>
      <Label x={460} y={gy(0) + 24} className="fill-accent">context from</Label>
      {thresholds.map((t, j) => {
        const s = 0.06 + j * 0.2;
        return (
          <motion.g key={t.context} {...on(s, s + 0.2, j === 2)}>
            <Label x={460} y={gy(0) + 38} className="fill-accent">{`${t.context.toLocaleString("en-US")} tokens`}</Label>
          </motion.g>
        );
      })}
      <Label x={20} y={258}>{"dashed: a split count the rule never gives that bucket"}</Label>
      <Label x={20} y={290} className="fill-text">ms per token, one request with a 1,984-token prompt</Label>
      {[
        { label: "5 graphs, 1 split", v: before, cls: "fill-baseline" },
        { label: "eager, split", v: eager, cls: "fill-text/60" },
        { label: "11 graphs", v: after, cls: "fill-accent" },
      ].map((b, i) => (
        <g key={b.label}>
          <Label x={20} y={316 + i * 22}>{b.label}</Label>
          <rect x={160} y={306 + i * 22} width={300} height={12} rx={6} className="fill-border" />
          <motion.rect x={160} y={306 + i * 22} width={(300 * b.v) / max} height={12} rx={6} className={b.cls} {...on(0.1 + i * 0.06)} />
          <Label x={470} y={316 + i * 22} className={i === 2 ? "fill-accent" : "fill-text"}>{`${b.v.toFixed(1)} ms`}</Label>
        </g>
      ))}
    </Frame>
  );
}
