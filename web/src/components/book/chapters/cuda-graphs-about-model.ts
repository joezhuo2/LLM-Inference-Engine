import * as THREE from "three";
import { cudaGraphsAboutNumbers, idleShare } from "@/components/kernels/cuda-graphs/about";
import { short } from "@/components/kernels/cuda-graphs/figures";
import type { BookModel } from "../book-scene";
import { createCard, inkSegments, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";

const { count, eagerGap, graphGap, shortest, longest, average, kernels } = cudaGraphsAboutNumbers;

const LX0 = 0.2;
const LX1 = 0.92;
const WIDE = LX1 - LX0;
const H = 0.03;
const DEPTH = 0.06;
const LIFT = 0.05;
const LEAD = 0.9;
const SWEEP = 5.6;
const CYCLE = 9;
const DROP = 0.55;

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const fmt = (v: number, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v: number) => (v < 10 ? fmt(v, 1) : fmt(v));

type Pen = (text: string, x: number, y: number, size: number, color: string, serif?: boolean, align?: CanvasTextAlign) => void;
const pen =
  (ctx: CanvasRenderingContext2D, fonts: Fonts, w: number, h: number): Pen =>
  (text, x, y, size, color, serif = false, align = "left") => {
    ctx.fillStyle = color;
    ctx.font = `${h * size}px ${serif ? fonts.serif : fonts.sans}`;
    ctx.textAlign = align;
    ctx.fillText(text, w * x, h * y);
    ctx.textAlign = "left";
  };

function bar(material: THREE.Material) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
  mesh.castShadow = true;
  const set = (x: number, width: number, height: number, depth: number, z: number, y0 = 0) => {
    const w = Math.max(0.0005, width);
    const h = Math.max(0.0001, height);
    mesh.scale.set(w, h, depth);
    mesh.position.set(x + w / 2, y0 + h / 2, z);
    mesh.visible = width > 0.0004 && height > 0.0004;
  };
  return { mesh, set };
}

export function createCudaGraphsAboutModel(kit: Kit, fonts: Fonts): BookModel {
  const root = new THREE.Group();
  const wash = washPlane(kit.palette.blue, 1, 0.6, 0.12);
  wash.position.set(0.5, 0.002, -0.24);
  root.add(wash);

  const title = createCard(kit, 0.86, 0.12, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("CUDA Graph: a chain of kernel launches, recorded once, launched once", 0.03, 0.3, 0.19, kit.palette.ink, true);
    write("kernel: one function the GPU runs on thousands of threads · launch: the CPU asking for one", 0.03, 0.58, 0.13, kit.palette.ink);
    write("CUDA: NVIDIA's platform for GPU programs, first spelled out as Compute Unified Device Architecture", 0.03, 0.84, 0.12, kit.palette.faint);
  });
  title.group.position.set(0.5, 0, -0.58);
  root.add(title.group);

  const lanes = [
    { gap: eagerGap, cpuZ: -0.43, gpuZ: -0.35, cardZ: -0.29, head: "launched one by one", sub: `one launch per kernel; a median ${eagerGap} µs gap before each`, color: kit.palette.pink },
    { gap: graphGap, cpuZ: -0.19, gpuZ: -0.11, cardZ: -0.05, head: "as a CUDA Graph", sub: `one launch for the whole chain; a median ${graphGap} µs between kernels`, color: kit.palette.blue },
  ].map((lane) => {
    const card = createCard(kit, 0.72, 0.045, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      write(lane.head, 0.02, 0.68, 0.52, lane.color, true);
      write(lane.sub, 0.35, 0.66, 0.36, kit.palette.ink);
    });
    card.group.position.set(LX0 + WIDE / 2, 0, lane.cardZ);
    root.add(card.group);
    for (const [text, z] of [
      ["CPU", lane.cpuZ],
      ["GPU", lane.gpuZ],
    ] as const) {
      const tag = createCard(kit, 0.1, 0.04, (ctx, w, h) => pen(ctx, fonts, w, h)(text, 0.5, 0.72, 0.6, kit.palette.ink, true, "center"));
      tag.group.position.set(0.12, 0, z - 0.01);
      root.add(tag.group);
    }
    const floor = new THREE.Mesh(new THREE.BoxGeometry(WIDE, 0.002, DEPTH + 0.012), kit.ink);
    floor.position.set(LX0 + WIDE / 2, 0.001, lane.gpuZ);
    root.add(floor);
    const shells = Array.from({ length: count }, () => {
      const shell = kit.solid(new THREE.BoxGeometry(1, 1, 1), { material: kit.glass, threshold: 40 });
      root.add(shell);
      return shell;
    });
    const fills = Array.from({ length: count }, () => {
      const b = bar(kit.blue);
      root.add(b.mesh);
      return b;
    });
    const gaps = Array.from({ length: count }, () => {
      const b = bar(kit.pink);
      root.add(b.mesh);
      return b;
    });
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.003, 0.07, DEPTH + 0.03), kit.ink);
    root.add(post);
    return { ...lane, shells, fills, gaps, post };
  });

  const eager = lanes[0];
  const graph = lanes[1];

  const tokens = Array.from({ length: count }, () => {
    const t = kit.solid(new THREE.BoxGeometry(0.026, 0.02, 0.026), { material: kit.blue });
    root.add(t);
    return t;
  });

  const chain = new THREE.Group();
  const nodes = Array.from({ length: count }, () => {
    const n = new THREE.Mesh(new THREE.SphereGeometry(0.012, 18, 14), kit.blue);
    n.castShadow = true;
    chain.add(n);
    return n;
  });
  const edge = new THREE.Mesh(new THREE.BoxGeometry(1, 0.004, 0.004), kit.ink);
  chain.add(edge);
  root.add(chain);

  const lnMin = Math.log(shortest.us);
  const lnMax = Math.log(longest.us);
  const railZ = 0.23;
  const railX = [0.2, 0.8];
  const toX = (us: number) => railX[0] + ((Math.log(us) - lnMin) / (lnMax - lnMin)) * (railX[1] - railX[0]);
  const toUs = (x: number) => Math.exp(lnMin + ((x - railX[0]) / (railX[1] - railX[0])) * (lnMax - lnMin));

  let us = shortest.us;
  let shown = us;
  let written = "";
  const readout = createCard(kit, 0.86, 0.1, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write(`${count} kernels of ${fmt(shown, 1)} µs each`, 0.03, 0.34, 0.28, kit.palette.ink, true);
    write(`launched one by one: the GPU idle ${pct(idleShare(shown, eagerGap))}% of the time`, 0.03, 0.62, 0.2, kit.palette.pink);
    write(`as a graph: idle ${pct(idleShare(shown, graphGap))}%, done ${fmt(count * (eagerGap - graphGap), 1)} µs sooner`, 0.03, 0.88, 0.2, kit.palette.blue);
    write(`a decode step: ${fmt(kernels)} launches,`, 0.97, 0.62, 0.17, kit.palette.faint, false, "right");
    write("or one graph launch", 0.97, 0.88, 0.17, kit.palette.faint, false, "right");
  });
  readout.group.position.set(0.5, 0, 0.06);
  root.add(readout.group);

  const rail = kit.solid(new THREE.BoxGeometry(railX[1] - railX[0] + 0.04, 0.016, 0.02));
  rail.position.set(0.5, 0.008, railZ);
  const marks = [shortest.us, average, longest.us];
  const ticks: number[] = [];
  for (const m of marks) ticks.push(toX(m), 0.017, railZ - 0.018, toX(m), 0.017, railZ + 0.018);
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.028, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(toX(us), 0.045, railZ);
  const cardX0 = 0.1;
  const cardW = 0.8;
  const railCard = createCard(kit, cardW, 0.05, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    const at = (us: number) => (toX(us) - cardX0) / cardW;
    write(`${fmt(shortest.us, 1)} µs, ${short[shortest.op]}`, at(shortest.us), 0.42, 0.36, kit.palette.ink, false, "center");
    write("shortest in a layer", at(shortest.us), 0.86, 0.32, kit.palette.faint, false, "center");
    write(`${fmt(average)} µs`, at(average), 0.42, 0.36, kit.palette.ink, false, "center");
    write("average in a step", at(average), 0.86, 0.32, kit.palette.faint, false, "center");
    write(`${fmt(longest.us)} µs, ${short[longest.op]}`, at(longest.us), 0.42, 0.36, kit.palette.ink, false, "center");
    write("longest in a layer", at(longest.us), 0.86, 0.32, kit.palette.faint, false, "center");
  });
  railCard.group.position.set(cardX0 + cardW / 2, 0, railZ + 0.055);
  const note = createCard(kit, 0.86, 0.04, (ctx, w, h) => {
    pen(ctx, fonts, w, h)("illustrative: drag the kernel length (log scale); gaps are the measured medians of the two profiles", 0.03, 0.7, 0.46, kit.palette.faint);
  });
  note.group.position.set(0.5, 0, 0.36);
  root.add(rail, inkSegments(ticks, kit.thin), knob, railCard.group, note.group);

  let held = false;
  let knobX = toX(us);
  let time = 0;

  return {
    exhibits: [root],
    handles: [knob],
    dragStart: () => {
      held = true;
    },
    drag: (x) => {
      knobX = Math.min(railX[1], Math.max(railX[0], x));
      us = toUs(knobX);
    },
    dragEnd: () => {
      held = false;
    },
    enter: () => {
      time = 0;
    },
    update: (dt, _exhibit, _stop, reduced) => {
      knob.position.x = held ? knobX : toX(us);
      shown = reduced ? us : shown + (us - shown) * Math.min(1, dt * 2.2);
      const text = fmt(shown, 1);
      if (text !== written) {
        written = text;
        readout.redraw();
      }
      time = reduced ? CYCLE - 1 : (time + dt) % CYCLE;
      const span = count * (shown + eagerGap);
      const scale = WIDE / span;
      const secondsPerUs = SWEEP / span;
      const nowUs = reduced ? span : (time - LEAD) / secondsPerUs;
      const fade = reduced ? 1 : 1 - smooth(CYCLE - 0.7, CYCLE, time);
      const back = reduced ? 0 : smooth(CYCLE - 0.7, CYCLE, time);
      const startAt = (lane: typeof eager, i: number) => i * (shown + lane.gap) + lane.gap;

      for (const lane of lanes) {
        const end = count * (shown + lane.gap);
        for (let i = 0; i < count; i++) {
          const s = startAt(lane, i);
          const x = LX0 + s * scale;
          const w = shown * scale;
          lane.shells[i].scale.set(Math.max(0.002, w - 0.002), H, DEPTH);
          lane.shells[i].position.set(x + w / 2, H / 2, lane.gpuZ);
          const run = Math.min(1, Math.max(0, (nowUs - s) / shown));
          lane.fills[i].set(x, (w - 0.002) * run, (H - 0.003) * fade, DEPTH - 0.006, lane.gpuZ);
          const g0 = s - lane.gap;
          const open = Math.min(1, Math.max(0, (nowUs - g0) / lane.gap));
          const gw = Math.max(lane === eager ? 0.0025 : 0.0012, lane.gap * scale);
          lane.gaps[i].set(LX0 + g0 * scale + (lane.gap * scale - gw) / 2, gw * open, 0.016 * fade, DEPTH, lane.gpuZ);
        }
        const at = Math.min(end, Math.max(0, nowUs));
        lane.post.position.set(LX0 + at * scale, 0.035, lane.gpuZ);
        lane.post.visible = !reduced && nowUs > 0 && fade > 0.05;
      }

      tokens.forEach((t, i) => {
        const s = startAt(eager, i);
        const x = LX0 + (s + shown / 2) * scale;
        const arrive = LEAD + s * secondsPerUs;
        const p = reduced ? 1 : smooth(arrive - DROP, arrive, time);
        const size = reduced ? 0 : Math.max(back, 1 - smooth(0.85, 1, p));
        t.position.set(x, 0.01 + LIFT * Math.sin(Math.PI * p), eager.cpuZ + (eager.gpuZ - eager.cpuZ) * p * (1 - back));
        t.scale.setScalar(Math.max(0.0001, size));
        t.visible = size > 0.02;
      });

      const first = startAt(graph, 0);
      const last = startAt(graph, count - 1);
      nodes.forEach((n, i) => n.position.set(LX0 + (startAt(graph, i) + shown / 2) * scale, 0.014, 0));
      const x0 = LX0 + (first + shown / 2) * scale;
      const x1 = LX0 + (last + shown / 2) * scale;
      edge.scale.x = Math.max(0.001, x1 - x0);
      edge.position.set((x0 + x1) / 2, 0.014, 0);
      const arrive = LEAD + first * secondsPerUs;
      const p = reduced ? 1 : smooth(arrive - DROP - 0.2, arrive, time);
      const size = reduced ? 0 : Math.max(back, 1 - smooth(0.85, 1, p));
      chain.position.set(0, LIFT * Math.sin(Math.PI * p), graph.cpuZ + (graph.gpuZ - graph.cpuZ) * p * (1 - back));
      chain.scale.set(1, Math.max(0.0001, size), Math.max(0.0001, size));
      chain.visible = size > 0.02;
    },
  };
}
