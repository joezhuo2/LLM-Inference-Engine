import * as THREE from "three";
import { buckets, captured, eagerLayer, fields, gemm, type Kernel, nodes, short, splitCounts, stageRows, thresholds } from "@/components/kernels/cuda-graphs/figures";
import trace from "@/content/kernels/cuda-graphs-trace.json";
import { createCard, inkEdges, type Kit, washPlane } from "../ink";
import type { BookModel } from "../book-scene";
import type { Fonts } from "../typeset";

export type CudaGraphsBookNumbers = {
  kernels: number;
  eagerIdle: number;
  spanUs: number;
  busyUs: number;
  layersAhead: number;
  shortKernels: number;
  longRun: { before: number; after: number; eager: number };
};

type Line = { text: string; color: string; size: number; y: number; x?: number; serif?: boolean; align?: CanvasTextAlign };

const placed = <T extends THREE.Object3D>(object: T, x: number, y: number, z: number): T => {
  object.position.set(x, y, z);
  return object;
};
const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const fmt = (v: number, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

function textCard(kit: Kit, fonts: Fonts, width: number, height: number, lines: () => Line[], tilt = -0.42) {
  return createCard(
    kit,
    width,
    height,
    (ctx, w, h) => {
      for (const l of lines()) {
        ctx.fillStyle = l.color;
        ctx.font = `${h * l.size}px ${l.serif ? fonts.serif : fonts.sans}`;
        ctx.textAlign = l.align ?? "left";
        ctx.fillText(l.text, w * (l.x ?? 0.03), h * l.y);
      }
      ctx.textAlign = "left";
    },
    tilt,
  );
}

function fillBar(kit: Kit, material: THREE.Material, width: number, height: number, depth: number) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
  mesh.castShadow = true;
  const set = (amount: number, x0: number, y0: number, z: number) => {
    const w = Math.max(0.0005, width * amount);
    mesh.scale.set(w, height, depth);
    mesh.position.set(x0 + w / 2, y0 + height / 2, z);
    mesh.visible = amount > 0.001;
  };
  return { mesh, set };
}

function launchesExhibit(kit: Kit, fonts: Fonts, numbers: CudaGraphsBookNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 1, 0.8, 0.12), 0.5, 0.002, -0.2));
  const t0 = -14;
  const t1 = 350;
  const tx = (t: number) => 0.08 + (0.84 * (t - t0)) / (t1 - t0);
  const cpuZ = -0.43;
  const gpuZ = -0.12;
  const cpu = trace.eager.cpu_in_layer;
  const launch = trace.eager.launches.cudaLaunchKernel.median_us;

  const cpuCard = textCard(kit, fonts, 0.84, 0.05, () => [
    { text: "CPU", color: kit.palette.ink, size: 0.56, y: 0.7, serif: true },
    { text: `cudaLaunchKernel, ${fmt(launch)} µs each, for kernels ${numbers.layersAhead} layers ahead`, color: kit.palette.faint, size: 0.4, y: 0.66, x: 0.1 },
  ]);
  cpuCard.group.position.set(0.5, 0, cpuZ - 0.11);
  const cpuBlocks = cpu.map((c) => {
    const a = tx(Math.max(t0, c.start));
    const b = tx(Math.min(t1, c.end));
    const block = kit.solid(new THREE.BoxGeometry(Math.max(0.004, b - a - 0.003), 1, 0.05), { threshold: 40 });
    block.position.set((a + b) / 2, 0, cpuZ);
    root.add(block);
    return { block, at: c.start };
  });

  const gpuCard = textCard(kit, fonts, 0.84, 0.05, () => [
    { text: "GPU", color: kit.palette.ink, size: 0.56, y: 0.7, serif: true },
    { text: `layer ${trace.eager.layer} of 22: its ten kernels, in stream order, to scale`, color: kit.palette.faint, size: 0.4, y: 0.66, x: 0.1 },
  ]);
  gpuCard.group.position.set(0.5, 0, gpuZ - 0.15);

  const kernels = eagerLayer.map((k: Kernel) => {
    const a = tx(k.start);
    const b = tx(k.end);
    const height = gemm.has(k.op) ? 0.05 : 0.032;
    const w = Math.max(0.004, b - a);
    const shell = kit.solid(new THREE.BoxGeometry(w, height, 0.07), { material: kit.glass, threshold: 40 });
    shell.position.set(a + w / 2, height / 2, gpuZ);
    const fill = fillBar(kit, kit.blue, w, height * 0.92, 0.064);
    root.add(shell, fill.mesh);
    return { k, a, w, height, fill };
  });
  const gaps = eagerLayer.slice(0, -1).map((k, i) => {
    const a = tx(k.end);
    const b = tx(eagerLayer[i + 1].start);
    const slab = new THREE.Mesh(new THREE.BoxGeometry(Math.max(0.002, b - a), 0.006, 0.07), kit.pink);
    slab.position.set((a + b) / 2, 0.003, gpuZ);
    root.add(slab);
    return { slab, from: k.end, to: eagerLayer[i + 1].start };
  });
  const names = textCard(
    kit,
    fonts,
    0.84,
    0.04,
    () =>
      eagerLayer
        .filter((k) => gemm.has(k.op) || k.op === "attention")
        .map((k) => ({ text: short[k.op], color: gemm.has(k.op) ? kit.palette.ink : kit.palette.blue, size: 0.56, y: 0.7, x: (tx((k.start + k.end) / 2) - 0.08) / 0.84, align: "center" as CanvasTextAlign })),
    -0.9,
  );
  names.group.position.set(0.5, 0, gpuZ + 0.05);
  const axis = textCard(
    kit,
    fonts,
    0.84,
    0.035,
    () => [0, 100, 200, 300].map((t) => ({ text: `${t} µs`, color: kit.palette.faint, size: 0.5, y: 0.7, x: (tx(t) - 0.08) / 0.84, align: "center" as CanvasTextAlign })),
    -0.9,
  );
  axis.group.position.set(0.5, 0, gpuZ + 0.1);

  const sweep = kit.solid(new THREE.BoxGeometry(0.004, 0.09, 0.42), { material: kit.ink, edges: false });
  sweep.position.set(tx(t0), 0.045, (cpuZ + gpuZ) / 2 + 0.02);
  let idle = 0;
  const idleCard = textCard(kit, fonts, 0.6, 0.11, () => [
    { text: `${fmt(numbers.kernels)} kernels per step, each its own launch`, color: kit.palette.ink, size: 0.24, y: 0.3 },
    { text: `idle in this layer so far: ${fmt(idle, 1)} µs`, color: kit.palette.pink, size: 0.24, y: 0.6 },
    { text: `the GPU idles for ${numbers.eagerIdle}% of a decode step`, color: kit.palette.faint, size: 0.2, y: 0.88 },
  ]);
  idleCard.group.position.set(0.5, 0, 0.18);
  root.add(cpuCard.group, gpuCard.group, names.group, axis.group, sweep, idleCard.group);

  let time = 0;
  let shownIdle = -1;
  const run = 11;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? run : (time + dt) % (run + 3);
      const now = t0 + (t1 - t0) * smooth(0, 1, time / run);
      sweep.position.x = tx(now);
      for (const c of cpuBlocks) c.block.scale.y = now >= c.at ? 0.03 : 0.012;
      for (const c of cpuBlocks) c.block.position.y = c.block.scale.y / 2;
      for (const n of kernels) n.fill.set(Math.min(1, Math.max(0, (now - n.k.start) / (n.k.end - n.k.start))), n.a, 0.002, gpuZ);
      let sum = 0;
      for (const g of gaps) {
        const part = Math.min(1, Math.max(0, (now - g.from) / (g.to - g.from)));
        g.slab.scale.set(Math.max(0.001, part), 1, 1);
        g.slab.position.x = tx(g.from) + ((tx(g.to) - tx(g.from)) * part) / 2;
        g.slab.visible = part > 0;
        sum += part * (g.to - g.from);
      }
      const rounded = Math.round(sum * 10) / 10;
      if (rounded !== shownIdle) {
        shownIdle = rounded;
        idle = rounded;
        idleCard.redraw();
      }
    },
  };
}

function gapsExhibit(kit: Kit, fonts: Fonts, numbers: CudaGraphsBookNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.pink, 0.9, 0.8, 0.12), 0.45, 0.002, -0.2));
  const ops = eagerLayer.filter((k) => ["rope", "write_kv", "attention"].includes(k.op));
  const t0 = ops[0].start - 1.5;
  const t1 = ops[2].end + 1;
  const tx = (t: number) => 0.06 + (0.56 * (t - t0)) / (t1 - t0);
  const z = -0.36;
  const height = 0.05;
  const zoomCard = textCard(kit, fonts, 0.56, 0.05, () => [{ text: `zoom: ${fmt(t1 - t0)} µs of layer ${trace.eager.layer}, eager`, color: kit.palette.faint, size: 0.46, y: 0.68 }]);
  zoomCard.group.position.set(0.34, 0, z - 0.17);
  root.add(zoomCard.group);
  const blocks = ops.map((k) => {
    const a = tx(k.start);
    const w = tx(k.end) - a;
    const shell = kit.solid(new THREE.BoxGeometry(w, height, 0.09), { material: kit.glass, threshold: 40 });
    shell.position.set(a + w / 2, height / 2, z);
    const fill = fillBar(kit, kit.blue, w, height * 0.92, 0.084);
    root.add(shell, fill.mesh);
    return { a, fill };
  });
  const gapSlabs = ops.slice(0, -1).map((k, i) => {
    const a = tx(k.end);
    const b = tx(ops[i + 1].start);
    const slab = new THREE.Mesh(new THREE.BoxGeometry(b - a, 0.012, 0.09), kit.pink);
    slab.position.set((a + b) / 2, 0.006, z);
    root.add(slab);
    return slab;
  });
  const opCard = textCard(
    kit,
    fonts,
    0.56,
    0.07,
    () => [
      ...ops.map((k) => ({ text: `${k.op} ${fmt(k.end - k.start, 1)} µs`, color: kit.palette.blue, size: 0.3, y: 0.38, x: (tx((k.start + k.end) / 2) - 0.06) / 0.56, align: "center" as CanvasTextAlign })),
      ...ops.slice(0, -1).map((k, i) => ({ text: `${fmt(ops[i + 1].start - k.end, 1)} idle`, color: kit.palette.pink, size: 0.3, y: 0.82, x: (tx((k.end + ops[i + 1].start) / 2) - 0.06) / 0.56, align: "center" as CanvasTextAlign })),
    ],
    -0.8,
  );
  opCard.group.position.set(0.34, 0, z + 0.07);
  root.add(opCard.group);

  const g = trace.eager.gap_us;
  const steps = [
    "1. the kernel before finishes, every block",
    "2. the GPU takes the next launch from the queue",
    "3. it sets up the new grid",
    "4. the first block runs",
  ];
  let phase = -1;
  const stepsCard = textCard(kit, fonts, 0.56, 0.17, () => [
    { text: `inside a gap: median ${g.p50} µs, middle 80% ${g.p10} to ${g.p90} µs`, color: kit.palette.ink, size: 0.13, y: 0.17 },
    ...steps.map((s, i) => ({ text: s, color: i === phase ? kit.palette.pink : kit.palette.faint, size: 0.12, y: 0.37 + i * 0.18 })),
  ]);
  stepsCard.group.position.set(0.34, 0, 0.0);
  root.add(stepsCard.group);

  const colH = 0.34;
  const max = numbers.spanUs;
  const busyH = (colH * numbers.busyUs) / max;
  const idleUs = numbers.spanUs - numbers.busyUs;
  const idleH = (colH * idleUs) / max;
  const colX = 0.83;
  const colZ = -0.3;
  const jar = kit.solid(new THREE.BoxGeometry(0.1, colH, 0.1), { material: kit.glass, threshold: 40 });
  jar.position.set(colX, colH / 2, colZ);
  const busy = new THREE.Mesh(new THREE.BoxGeometry(0.092, busyH, 0.092), kit.blue);
  busy.position.set(colX, busyH / 2, colZ);
  busy.castShadow = true;
  const idleFill = new THREE.Mesh(new THREE.BoxGeometry(0.092, 1, 0.092), kit.pink);
  idleFill.castShadow = true;
  root.add(jar, busy, idleFill);
  let count = 0;
  const boundaries = numbers.kernels - 1;
  const stepCard = textCard(kit, fonts, 0.3, 0.17, () => [
    { text: "one decode step", color: kit.palette.faint, size: 0.13, y: 0.17 },
    { text: `kernels busy ${fmt(numbers.busyUs)} µs`, color: kit.palette.blue, size: 0.13, y: 0.38 },
    { text: `${fmt(count)} of ${fmt(boundaries)} gaps`, color: kit.palette.pink, size: 0.13, y: 0.58 },
    { text: `idle ${fmt((idleUs * count) / boundaries)} µs`, color: kit.palette.pink, size: 0.13, y: 0.78 },
    { text: `${numbers.shortKernels} of 10 kernels run under 3 µs`, color: kit.palette.faint, size: 0.1, y: 0.95 },
  ]);
  stepCard.group.position.set(colX - 0.02, 0, colZ + 0.1);
  root.add(stepCard.group);

  let time = 0;
  let shown = -1;
  let shownPhase = -2;
  const run = 12;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? run : (time + dt) % (run + 2);
      blocks.forEach((b, i) => b.fill.set(smooth(i * 2.4, i * 2.4 + 1.2, time), b.a, 0.002, z));
      gapSlabs.forEach((s, i) => {
        const part = smooth(i * 2.4 + 1.2, i * 2.4 + 2.4, time);
        s.scale.y = Math.max(0.01, part * 4);
        s.position.y = 0.024 * part;
        s.visible = part > 0;
      });
      const inGap = gapSlabs.findIndex((_, i) => time >= i * 2.4 + 1.2 && time < i * 2.4 + 2.4);
      const p = inGap < 0 ? -1 : Math.min(3, Math.floor(((time - (inGap * 2.4 + 1.2)) / 1.2) * 4));
      if (p !== shownPhase) {
        shownPhase = p;
        phase = p;
        stepsCard.redraw();
      }
      const level = smooth(0.5, run, time);
      const h = Math.max(0.0005, idleH * level);
      idleFill.scale.y = h;
      idleFill.position.set(colX, busyH + h / 2, colZ);
      const c = Math.round(boundaries * level);
      if (c !== shown) {
        shown = c;
        count = c;
        stepCard.redraw();
      }
    },
  };
}

function captureExhibit(kit: Kit, fonts: Fonts, numbers: CudaGraphsBookNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 1, 0.8, 0.12), 0.5, 0.002, -0.1));
  const code = [
    { text: "run_staged(b, b);", note: "eager, so cuBLAS picks its kernels", at: 0 },
    { text: "cudaStreamBeginCapture(stream, Global);", note: "", at: 2 },
    { text: "run_staged(b, b);", note: "recorded, not run", at: 2.6 },
    { text: "cudaStreamEndCapture(stream, &graph);", note: "", at: 6.4 },
    { text: "cudaGraphInstantiate(&exec, graph, 0);", note: "once, at startup", at: 7.2 },
    { text: "cudaGraphLaunch(exec, stream);", note: "every decode step", at: 9.2 },
  ];
  let line = 0;
  const codeCard = textCard(kit, fonts, 0.84, 0.2, () =>
    code.flatMap((c, i) => [
      { text: c.text, color: i === line ? kit.palette.pink : kit.palette.ink, size: 0.1, y: 0.16 + i * 0.15 },
      { text: c.note, color: kit.palette.faint, size: 0.09, y: 0.16 + i * 0.15, x: 0.6 },
    ]),
  );
  codeCard.group.position.set(0.5, 0, -0.56);
  root.add(codeCard.group);

  const streamZ = -0.3;
  const rail = kit.solid(new THREE.BoxGeometry(0.84, 0.01, 0.03));
  rail.position.set(0.5, 0.005, streamZ);
  root.add(rail);
  const streamCard = textCard(kit, fonts, 0.2, 0.03, () => [{ text: "the stream", color: kit.palette.faint, size: 0.6, y: 0.75 }], -0.9);
  streamCard.group.position.set(0.18, 0, streamZ + 0.025);
  root.add(streamCard.group);
  const pulse = new THREE.Mesh(new THREE.SphereGeometry(0.016, 18, 14), kit.blue);
  pulse.castShadow = true;
  root.add(pulse);

  const nodeW = 0.13;
  const nodeH = 0.04;
  const nx = (i: number) => 0.14 + (i % 5) * 0.18;
  const nz = (i: number) => -0.12 + Math.floor(i / 5) * 0.13;
  const trayGeometry = new THREE.BoxGeometry(0.9, 0.008, 0.3);
  const tray = kit.solid(trayGeometry);
  tray.position.set(0.5, 0.004, -0.055);
  root.add(tray);
  const graphNodes = nodes.map((n, i) => {
    const big = ["QKV", "O", "gate/up", "down"].includes(n);
    const node = kit.solid(new THREE.BoxGeometry(nodeW, nodeH, 0.05), { threshold: 40 });
    const fill = new THREE.Mesh(new THREE.BoxGeometry(nodeW - 0.006, nodeH - 0.004, 0.044), kit.blue);
    fill.castShadow = true;
    root.add(node, fill);
    const card = textCard(kit, fonts, nodeW, 0.032, () => [{ text: n, color: big ? kit.palette.ink : kit.palette.blue, size: 0.62, y: 0.74, x: 0.5, align: "center" }], -1.0);
    card.group.position.set(nx(i), 0, nz(i) + 0.035);
    root.add(card.group);
    return { node, fill, card: card.group };
  });
  const edges: number[] = [];
  for (let i = 0; i < nodes.length - 1; i++) {
    const a = new THREE.Vector3(nx(i) + nodeW / 2, nodeH / 2 + 0.008, nz(i));
    const b = new THREE.Vector3(nx(i + 1) - nodeW / 2, nodeH / 2 + 0.008, nz(i + 1));
    if (i === 4) {
      a.set(nx(4), nodeH + 0.008, nz(4) + 0.025);
      b.set(nx(5), nodeH + 0.008, nz(5) - 0.025);
    }
    edges.push(a.x, a.y, a.z, b.x, b.y, b.z);
  }
  const edgeGeometry = new THREE.BufferGeometry();
  edgeGeometry.setAttribute("position", new THREE.Float32BufferAttribute(edges, 3));
  const edgeLines = new THREE.LineSegments(edgeGeometry, new THREE.LineBasicMaterial({ color: kit.palette.ink, transparent: true }));
  root.add(edgeLines);

  const boxGeometry = new THREE.BoxGeometry(0.92, 0.1, 0.32);
  const lid = new THREE.Group();
  lid.add(new THREE.Mesh(boxGeometry, kit.glass), inkEdges(boxGeometry, kit.line, 30));
  const lidPivot = new THREE.Group();
  lidPivot.position.set(0.5, 0, -0.055);
  lid.position.y = 0.05;
  lidPivot.add(lid);
  root.add(lidPivot);

  const chip = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.02, 0.03), kit.pink);
  chip.castShadow = true;
  root.add(chip);
  const sumCard = textCard(kit, fonts, 0.84, 0.09, () => [
    { text: `instantiated once: ${fmt(numbers.kernels)} kernels, launched with one cudaGraphLaunch`, color: kit.palette.ink, size: 0.27, y: 0.4 },
    { text: "one layer of the graph drawn; it holds all 22 and the LM head", color: kit.palette.faint, size: 0.22, y: 0.82 },
  ]);
  sumCard.group.position.set(0.5, 0, 0.2);
  root.add(sumCard.group);

  let time = 0;
  let shownLine = -1;
  const loop = 13;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 9.15 : (time + dt) % loop;
      const eager = smooth(0, 1.8, time);
      pulse.visible = time < 1.9 && !reduced;
      pulse.position.set(0.1 + 0.8 * eager, 0.026, streamZ);
      graphNodes.forEach((g, i) => {
        const s = 2.6 + i * 0.36;
        const drop = smooth(s, s + 0.7, time);
        const x = 0.1 + 0.8 * (i / 9);
        const px = x + (nx(i) - x) * drop;
        const pz = streamZ + (nz(i) - streamZ) * drop;
        const lift = Math.sin(Math.PI * drop) * 0.06;
        const visible = time >= s || reduced;
        g.node.visible = visible;
        g.card.visible = drop > 0.98 || reduced;
        g.node.position.set(px, nodeH / 2 + 0.008 + lift, pz);
        const run = smooth(10 + i * 0.2, 10.3 + i * 0.2, time);
        g.fill.visible = run > 0.01 && !reduced;
        g.fill.scale.set(Math.max(0.01, run), 1, 1);
        g.fill.position.set(px - ((nodeW - 0.006) * (1 - run)) / 2, nodeH / 2 + 0.008, pz);
      });
      (edgeLines.material as THREE.LineBasicMaterial).opacity = reduced ? 1 : smooth(6.4, 7, time);
      const close = reduced ? 1 : smooth(7.2, 8.6, time);
      lidPivot.scale.y = Math.max(0.001, close);
      lidPivot.visible = close > 0.01;
      const fly = smooth(9.2, 10, time);
      chip.visible = time >= 9.2 && time < 10.1 && !reduced;
      chip.position.set(0.5, 0.2 - 0.12 * fly, -0.3 + 0.2 * fly);
      let l = 0;
      code.forEach((c, i) => {
        if (time >= c.at) l = i;
      });
      if (reduced) l = 4;
      if (l !== shownLine) {
        shownLine = l;
        line = l;
        codeCard.redraw();
      }
    },
  };
}

function stageExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 1, 0.8, 0.12), 0.5, 0.002, -0.1));
  const cols = [0.16, 0.3, 0.44, 0.58, 0.75];
  const hostCard = textCard(kit, fonts, 0.84, 0.17, () => [
    { text: "the step's batch on the host: 3 requests padded to the 4-row bucket", color: kit.palette.faint, size: 0.11, y: 0.14 },
    ...fields.map((f, j) => ({ text: f, color: kit.palette.faint, size: 0.11, y: 0.32, x: (cols[j] - 0.08) / 0.84 + 0.06 })),
    ...stageRows.flatMap((r, i) => [
      { text: r.name, color: r.pad ? kit.palette.pink : kit.palette.ink, size: 0.11, y: 0.5 + i * 0.15 },
      ...r.v.map((v, j) => ({ text: v, color: r.pad ? kit.palette.pink : kit.palette.ink, size: 0.11, y: 0.5 + i * 0.15, x: (cols[j] - 0.08) / 0.84 + 0.06 })),
    ]),
  ]);
  hostCard.group.position.set(0.5, 0, -0.56);
  root.add(hostCard.group);

  const bufZ = -0.18;
  const cellW = 0.024;
  const segW = [0.11, 0.11, 0.11, 0.11, 0.2];
  const segX = [0.12, 0.25, 0.38, 0.51, 0.64];
  const plate = kit.solid(new THREE.BoxGeometry(0.8, 0.012, 0.09));
  plate.position.set(0.5, 0.006, bufZ);
  root.add(plate);
  segX.forEach((x, j) => {
    const seg = kit.solid(new THREE.BoxGeometry(segW[j], 0.01, 0.06), { material: kit.glass });
    seg.position.set(x + segW[j] / 2, 0.017, bufZ);
    root.add(seg);
  });
  const offsets = ["0: token", "t: position", "2t: slot", "3t: context", "4t: block tables"];
  const bufCard = textCard(
    kit,
    fonts,
    0.84,
    0.07,
    () => [
      ...offsets.map((o, j) => ({ text: o, color: kit.palette.ink, size: 0.3, y: 0.36, x: (segX[j] - 0.08) / 0.84 })),
      { text: "one device buffer of int32, t = 4 rows, at the same address on every step", color: kit.palette.faint, size: 0.28, y: 0.84 },
    ],
    -0.8,
  );
  bufCard.group.position.set(0.5, 0, bufZ + 0.07);
  root.add(bufCard.group);

  const cubes: { mesh: THREE.Mesh; field: number; row: number }[] = [];
  const cubeGeometry = new THREE.BoxGeometry(cellW - 0.004, 0.018, 0.03);
  for (let row = 0; row < 4; row++) {
    for (let field = 0; field < 5; field++) {
      const mesh = new THREE.Mesh(cubeGeometry, stageRows[row].pad ? kit.pink : kit.blue);
      mesh.castShadow = true;
      cubes.push({ mesh, field, row });
      root.add(mesh);
    }
  }
  const cellX = (field: number, row: number) => segX[field] + 0.008 + row * (field === 4 ? 0.048 : cellW) + (cellW - 0.004) / 2;

  const kz = 0.12;
  const reads = [
    { name: "embedding", field: 0 },
    { name: "rope", field: 1 },
    { name: "write_kv", field: 2 },
    { name: "attention", field: 3 },
  ];
  const kernelX = [0.17, 0.37, 0.57, 0.77];
  const graph = kit.solid(new THREE.BoxGeometry(0.84, 0.008, 0.11));
  graph.position.set(0.5, 0.004, kz);
  root.add(graph);
  const rods = reads.map((r, i) => {
    const k = kit.solid(new THREE.BoxGeometry(0.15, 0.035, 0.05), { threshold: 40 });
    k.position.set(kernelX[i], 0.026, kz);
    root.add(k);
    const from = new THREE.Vector3(segX[r.field] + segW[r.field] / 2, 0.03, bufZ + 0.03);
    const to = new THREE.Vector3(kernelX[i], 0.045, kz - 0.025);
    const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 1, 8), kit.pink);
    rod.position.copy(from).add(to).multiplyScalar(0.5);
    rod.scale.y = from.distanceTo(to);
    rod.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), to.clone().sub(from).normalize());
    root.add(rod);
    if (r.field === 3) {
      const from2 = new THREE.Vector3(segX[4] + segW[4] / 2, 0.03, bufZ + 0.03);
      const rod2 = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 1, 8), kit.pink);
      rod2.position.copy(from2).add(to).multiplyScalar(0.5);
      rod2.scale.y = from2.distanceTo(to);
      rod2.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), to.clone().sub(from2).normalize());
      root.add(rod2);
      return [rod, rod2];
    }
    return [rod];
  });
  const kernelCard = textCard(
    kit,
    fonts,
    0.84,
    0.07,
    () => [
      ...reads.map((r, i) => ({ text: r.name, color: kit.palette.ink, size: 0.3, y: 0.36, x: (kernelX[i] - 0.08) / 0.84, align: "center" as CanvasTextAlign })),
      { text: "the captured kernels: every pointer and grid fixed when the graph was recorded", color: kit.palette.faint, size: 0.28, y: 0.84 },
    ],
    -0.8,
  );
  kernelCard.group.position.set(0.5, 0, kz + 0.05);
  const noteCard = textCard(kit, fonts, 0.84, 0.06, () => [
    { text: "a padded row writes nowhere (slot -1) and reads nothing (context 0)", color: kit.palette.pink, size: 0.3, y: 0.42 },
    { text: "a replay gets new values at the same addresses, never new addresses", color: kit.palette.ink, size: 0.3, y: 0.86 },
  ]);
  noteCard.group.position.set(0.5, 0, 0.3);
  root.add(kernelCard.group, noteCard.group);

  let time = 0;
  const loop = 9;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 6 : (time + dt) % loop;
      cubes.forEach((c) => {
        const s = 0.4 + c.row * 0.5 + c.field * 0.08;
        const drop = reduced ? 1 : smooth(s, s + 1, time);
        const leave = reduced ? 0 : smooth(8.2, 8.9, time);
        const startX = cols[c.field] + 0.02;
        const startZ = -0.47 + c.row * 0.03;
        const x = startX + (cellX(c.field, c.row) - startX) * drop;
        const z = startZ + (bufZ - startZ) * drop;
        c.mesh.position.set(x, 0.031 + Math.sin(Math.PI * drop) * 0.08 - leave * 0.03, z);
        c.mesh.visible = (drop > 0 || reduced) && leave < 0.95;
      });
      const read = reduced ? 1 : smooth(4.2, 4.8, time) * (1 - smooth(7.4, 8, time));
      rods.flat().forEach((r) => {
        r.visible = read > 0.02;
        r.scale.x = read;
        r.scale.z = read;
      });
    },
  };
}

function splitsExhibit(kit: Kit, fonts: Fonts, numbers: CudaGraphsBookNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 1, 0.8, 0.12), 0.5, 0.002, -0.15));
  const gx = (j: number) => 0.34 + j * 0.16;
  const gz = (i: number) => -0.46 + i * 0.066;
  const count = Object.values(captured).reduce((s, v) => s + v.length, 0);
  const head = textCard(
    kit,
    fonts,
    0.86,
    0.045,
    () => [
      { text: "bucket", color: kit.palette.faint, size: 0.46, y: 0.7 },
      ...splitCounts.map((s, j) => ({ text: `${s} split${s > 1 ? "s" : ""}`, color: kit.palette.faint, size: 0.46, y: 0.7, x: (gx(j) - 0.07) / 0.86, align: "center" as CanvasTextAlign })),
    ],
    -0.6,
  );
  head.group.position.set(0.5, 0, gz(0) - 0.09);
  root.add(head.group);
  const cellGeometry = new THREE.BoxGeometry(0.12, 0.03, 0.044);
  const cells: Record<string, THREE.Group> = {};
  buckets.forEach((b, i) => {
    const label = textCard(kit, fonts, 0.16, 0.03, () => [{ text: `${b} row${b > 1 ? "s" : ""}`, color: kit.palette.ink, size: 0.66, y: 0.76 }], -1.1);
    label.group.position.set(0.15, 0, gz(i) + 0.005);
    root.add(label.group);
    splitCounts.forEach((s, j) => {
      if (captured[b].includes(s)) {
        const cell = kit.solid(cellGeometry, { threshold: 40 });
        cell.position.set(gx(j), 0.015, gz(i));
        root.add(cell);
        cells[`${b}-${s}`] = cell;
      } else {
        const ghost = inkEdges(cellGeometry, kit.thin, 30, 0.002);
        ghost.position.set(gx(j), 0.015, gz(i));
        root.add(ghost);
      }
    });
  });
  const live = new THREE.Mesh(new THREE.BoxGeometry(0.114, 0.026, 0.038), kit.blue);
  live.castShadow = true;
  root.add(live);

  const railZ = 0.0;
  const railA = 0.12;
  const railB = 0.88;
  const maxContext = thresholds[thresholds.length - 1].context;
  const cx = (c: number) => railA + ((railB - railA) * c) / maxContext;
  const rail = kit.solid(new THREE.BoxGeometry(railB - railA, 0.016, 0.02));
  rail.position.set(0.5, 0.008, railZ);
  root.add(rail);
  const ticks: number[] = [];
  for (const t of thresholds) ticks.push(cx(t.context), 0.017, railZ - 0.02, cx(t.context), 0.017, railZ + 0.02);
  const tickGeometry = new THREE.BufferGeometry();
  tickGeometry.setAttribute("position", new THREE.Float32BufferAttribute(ticks, 3));
  root.add(new THREE.LineSegments(tickGeometry, new THREE.LineBasicMaterial({ color: kit.palette.ink })));
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.026, 28, 22), kit.pink);
  knob.castShadow = true;
  root.add(knob);

  let context = 300;
  const splitsAt = (c: number) => thresholds.filter((t) => t.context <= c).pop()?.splits ?? 1;
  const readout = textCard(kit, fonts, 0.84, 0.075, () => [
    { text: "one request's longest context", color: kit.palette.faint, size: 0.26, y: 0.32 },
    { text: `${fmt(Math.floor(context))} tokens: the 1-row graph with ${splitsAt(context)} split${splitsAt(context) > 1 ? "s" : ""}`, color: kit.palette.ink, size: 0.4, y: 0.82, serif: true },
    ...thresholds.map((t) => ({ text: fmt(t.context), color: kit.palette.faint, size: 0.24, y: 0.32, x: (cx(t.context) - 0.08) / 0.84, align: "center" as CanvasTextAlign })).slice(1),
  ]);
  readout.group.position.set(0.5, 0, railZ + 0.04);
  root.add(readout.group);

  const max = Math.max(numbers.longRun.before, numbers.longRun.after, numbers.longRun.eager);
  const bars = [
    { label: "5 graphs, 1 split", v: numbers.longRun.before, color: kit.palette.faint },
    { label: "eager, split", v: numbers.longRun.eager, color: kit.palette.ink },
    { label: `${count} graphs`, v: numbers.longRun.after, color: kit.palette.blue },
  ];
  const barCard = createCard(kit, 0.84, 0.14, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.13}px ${fonts.sans}`;
    ctx.fillText("ms per token, one request with a 1,984-token prompt", w * 0.03, h * 0.17);
    bars.forEach((b, i) => {
      const y = h * (0.42 + i * 0.22);
      ctx.fillStyle = kit.palette.faint;
      ctx.font = `${h * 0.12}px ${fonts.sans}`;
      ctx.fillText(b.label, w * 0.03, y);
      ctx.fillStyle = b.color;
      ctx.fillRect(w * 0.3, y - h * 0.1, w * 0.5 * (b.v / max), h * 0.11);
      ctx.fillStyle = b.color === kit.palette.faint ? kit.palette.ink : b.color;
      ctx.fillText(`${b.v.toFixed(1)} ms`, w * 0.82, y);
    });
  });
  barCard.group.position.set(0.5, 0, 0.2);
  root.add(barCard.group);

  let held = false;
  let shown = -1;
  const apply = () => {
    knob.position.set(cx(context), 0.042, railZ);
    const s = splitsAt(context);
    const cell = cells[`1-${s}`];
    live.position.set(cell.position.x, 0.034, cell.position.z);
    const whole = Math.floor(context);
    if (whole !== shown) {
      shown = whole;
      readout.redraw();
    }
  };
  apply();
  return {
    root,
    knob,
    setHeld: (v: boolean) => {
      held = v;
    },
    drag: (x: number) => {
      context = Math.min(maxContext, Math.max(0, ((x - railA) / (railB - railA)) * maxContext));
      apply();
    },
    update: (dt: number, reduced: boolean) => {
      if (!held && !reduced) context = context + dt * 150 > maxContext ? 0 : context + dt * 150;
      apply();
    },
  };
}

export function createCudaGraphsModel(kit: Kit, fonts: Fonts, numbers: CudaGraphsBookNumbers): BookModel {
  const launches = launchesExhibit(kit, fonts, numbers);
  const gaps = gapsExhibit(kit, fonts, numbers);
  const capture = captureExhibit(kit, fonts, numbers);
  const stage = stageExhibit(kit, fonts);
  const splits = splitsExhibit(kit, fonts, numbers);
  return {
    exhibits: [launches.root, gaps.root, capture.root, stage.root, splits.root],
    handles: [splits.knob],
    dragStart: () => splits.setHeld(true),
    drag: (x) => splits.drag(x),
    dragEnd: () => splits.setHeld(false),
    update: (dt, exhibit, _stop, reduced) => {
      if (exhibit === 0) launches.update(dt, reduced);
      if (exhibit === 1) gaps.update(dt, reduced);
      if (exhibit === 2) capture.update(dt, reduced);
      if (exhibit === 3) stage.update(dt, reduced);
      if (exhibit === 4) splits.update(dt, reduced);
    },
  };
}
