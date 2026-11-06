"use client";

import { motion } from "motion/react";
import trace from "@/content/kernels/cuda-graphs-trace.json";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";
import { StickyStory, type StoryStep } from "../sticky-story";
import { eagerLayer, short } from "./figures";

export const cudaGraphsAbout: StoryStep = {
  id: "about",
  title: "what a CUDA Graph is",
  body: (
    <>
      <p>
        Everything in the chapters so far runs on the GPU as kernels. A kernel is one function written for the GPU and run there by thousands of threads at once: a GEMV, RoPE, attention and a norm are each a kernel. CUDA is NVIDIA&apos;s platform for writing and running them; the name was first spelled out as Compute Unified Device Architecture. The engine&apos;s own C++ runs on the CPU, and the CPU decides which kernel comes next.
      </p>
      <p>
        The GPU does not start a kernel on its own. The CPU asks for each one with a kernel launch: which function, how many thread blocks, which arguments. The request costs the CPU microseconds of work in the CUDA driver and goes onto a stream, a queue the GPU works through in order. A kernel begins only when the one before it has finished and the GPU has set up the next one, so there is a short gap before every kernel. Next to a long kernel the gap is nothing; next to a short one it can take as long as the kernel.
      </p>
      <p>
        A CUDA Graph hands the GPU the same work another way. The chain of launches is written down once as a graph: every kernel is a node that keeps its function, its grid and its arguments, with an edge from the kernel that has to run before it. From then on the whole chain goes to the GPU with a single launch, and the GPU moves from one kernel to the next with almost no gap. A decode step is such a chain, hundreds of kernels in a fixed order, and the rest of this chapter measures what its gaps cost and what it takes to record it.
      </p>
    </>
  ),
};

const count = 6;
const eagerGap = trace.eager.gap_us.p50;
const graphGap = trace.graphs.gap_us.p50;
const lengths = eagerLayer.map((k) => ({ op: k.op, us: k.end - k.start }));
const shortest = lengths.reduce((a, b) => (b.us < a.us ? b : a));
const average = trace.eager.busy_us / trace.eager.kernels;

export const cudaGraphsAboutNumbers = { count, eagerGap, graphGap, shortest, longest: lengths.reduce((a, b) => (b.us > a.us ? b : a)), average, kernels: trace.eager.kernels };

export const idleShare = (us: number, gap: number) => (100 * gap) / (us + gap);

const fmt = (v: number, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v: number) => (v < 10 ? fmt(v, 1) : fmt(v));

export const cudaGraphsAboutCaption = `illustrative: ${count} kernels of one length drawn of the ${fmt(trace.eager.kernels)} in a decode step; the gaps are the median gaps of the eager and graph Nsight Systems profiles`;

const x0 = 60;
const lane = 480;
const sweep = 0.34;

function Lanes({ us, from, final }: { us: number; from: number; final: boolean }) {
  const { on, grow } = useLoop(16);
  const span = count * (us + eagerGap);
  const px = (t: number) => (lane * t) / span;
  const frac = (t: number) => from + (sweep * t) / span;
  const rows = [
    { gap: eagerGap, top: 92, launches: count, name: "launched one by one: one launch per kernel" },
    { gap: graphGap, top: 186, launches: 1, name: "as a CUDA Graph: one launch for the whole chain" },
  ];
  return (
    <g>
      {rows.map((row) => {
        const start = (i: number) => i * (us + row.gap) + row.gap;
        const end = count * (us + row.gap);
        return (
          <g key={row.top}>
            <Label x={20} y={row.top - 14} className="fill-text">
              {row.name}
            </Label>
            <Label x={20} y={row.top + 9}>
              CPU
            </Label>
            <Label x={20} y={row.top + 41}>
              GPU
            </Label>
            {row.launches === count ? (
              Array.from({ length: count }, (_, i) => (
                <motion.rect key={`t${i}`} x={x0 + px(start(i))} y={row.top} width={10} height={10} rx={2} className="fill-text/60" {...on(from, frac(start(i)), final)} />
              ))
            ) : (
              <motion.g {...on(from, frac(start(0)), final)}>
                <line x1={x0 + px(start(0)) + 5} x2={x0 + px(start(count - 1)) + 5} y1={row.top + 5} y2={row.top + 5} className="stroke-text/60" strokeWidth={1.5} />
                {Array.from({ length: count }, (_, i) => (
                  <circle key={`n${i}`} cx={x0 + px(start(i)) + 5} cy={row.top + 5} r={4} className="fill-text/60" />
                ))}
              </motion.g>
            )}
            {Array.from({ length: count }, (_, i) => (
              <g key={`k${i}`}>
                <motion.rect x={x0 + px(start(i) - row.gap)} y={row.top + 28} width={Math.max(1.5, px(row.gap))} height={22} className="fill-warn" {...on(frac(start(i) - row.gap), from + 0.48, final)} />
                <motion.rect x={x0 + px(start(i))} y={row.top + 28} width={Math.max(1.5, px(us))} height={22} rx={2} className="fill-accent" {...grow(frac(start(i)), frac(start(i) + us), "x", from + 0.47)} />
              </g>
            ))}
            <line x1={x0 + px(end)} x2={x0 + px(end)} y1={row.top + 22} y2={row.top + 56} className={row.launches === count ? "stroke-warn" : "stroke-accent"} strokeWidth={1.25} />
          </g>
        );
      })}
    </g>
  );
}

function Readout({ us, name }: { us: number; name: string }) {
  return (
    <g>
      <Label x={20} y={300} className="fill-text">{`${count} kernels of ${fmt(us, 1)} µs, ${name}`}</Label>
      <Label x={20} y={318} className="fill-warn">{`launched one by one: the GPU is idle ${pct(idleShare(us, eagerGap))}% of the time`}</Label>
      <Label x={20} y={336} className="fill-accent">{`as a graph: idle ${pct(idleShare(us, graphGap))}%, done ${fmt(count * (eagerGap - graphGap), 1)} µs sooner`}</Label>
    </g>
  );
}

export function CudaGraphsAboutFigure() {
  const { on } = useLoop(16);
  return (
    <Frame
      label={`The same ${count} kernels launched one by one and as one CUDA Graph. Launched one by one, each kernel waits a median ${eagerGap} µs gap first; in a graph the gap is ${graphGap} µs. With kernels of ${fmt(average)} µs the gaps are a small share; with kernels of ${fmt(shortest.us, 1)} µs they take most of the time`}
    >
      <Label x={20} y={24}>{"the same kernels, one launch each or one launch for all"}</Label>
      <motion.g {...on(0.02, 0.48, false)}>
        <Label x={20} y={48} className="fill-accent">{`kernels of ${fmt(average)} µs, the average in a decode step`}</Label>
        <Lanes us={average} from={0.02} final={false} />
        <Readout us={average} name="the average length" />
      </motion.g>
      <motion.g {...on(0.5, 0.97)}>
        <Label x={20} y={48} className="fill-accent">{`kernels of ${fmt(shortest.us, 1)} µs, as short as ${short[shortest.op]}, the shortest in a layer`}</Label>
        <Lanes us={shortest.us} from={0.5} final />
        <Readout us={shortest.us} name={`as short as ${short[shortest.op]}`} />
      </motion.g>
      <rect x={20} y={258} width={10} height={10} rx={2} className="fill-text/60" />
      <Label x={36} y={267}>a launch</Label>
      <rect x={110} y={258} width={10} height={10} rx={2} className="fill-accent" />
      <Label x={126} y={267}>a kernel running</Label>
      <rect x={240} y={261} width={10} height={5} className="fill-warn" />
      <Label x={256} y={267} className="fill-warn">the gap before it, GPU idle</Label>
      <Label x={20} y={372}>{`gaps: medians over every decode step of the two profiles, ${eagerGap} µs eager and ${graphGap} µs in a graph`}</Label>
    </Frame>
  );
}

export function CudaGraphsAboutStory() {
  return <StickyStory kicker="the idea" steps={[cudaGraphsAbout]} caption={() => cudaGraphsAboutCaption} figure={() => <CudaGraphsAboutFigure />} />;
}
