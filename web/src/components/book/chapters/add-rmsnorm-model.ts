import * as THREE from "three";
import { createCard, type Kit, ribbonTube, washPlane } from "../ink";
import type { BookModel } from "../book-scene";
import type { Fonts } from "../typeset";

export type AddRmsnormNumbers = { pair: number; first: number; firstFlat: string; naiveKb: number; fusedKb: number };

type Line = { text: string; color: string; size: number; serif?: boolean };

const placed = <T extends THREE.Object3D>(object: T, x: number, y: number, z: number): T => {
  object.position.set(x, y, z);
  return object;
};
const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

function note(kit: Kit, fonts: Fonts, w: number, h: number, lines: () => Line[], tilt = -0.42) {
  return createCard(
    kit,
    w,
    h,
    (ctx, W, H) => {
      const rows = lines();
      const total = rows.reduce((a, l) => a + l.size * H * 1.3, 0) - rows[rows.length - 1].size * H * 0.3;
      let y = (H - total) / 2;
      for (const l of rows) {
        y += l.size * H;
        ctx.fillStyle = l.color;
        ctx.font = `${l.size * H}px ${l.serif ? fonts.serif : fonts.sans}`;
        ctx.fillText(l.text, W * 0.04, y);
        y += l.size * H * 0.3;
      }
    },
    tilt,
  );
}

function arcLine(kit: Kit, from: THREE.Vector3, to: THREE.Vector3, lift: number) {
  const points: number[] = [];
  const at = (t: number) => new THREE.Vector3().lerpVectors(from, to, t).setY(from.y + (to.y - from.y) * t + Math.sin(Math.PI * t) * lift);
  for (let i = 0; i < 24; i++) points.push(...at(i / 24).toArray(), ...at((i + 1) / 24).toArray());
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
  const line = new THREE.LineSegments(geometry, new THREE.LineDashedMaterial({ color: kit.palette.faint, dashSize: 0.01, gapSize: 0.008 }));
  line.computeLineDistances();
  return { line, at };
}

const XS = [0.3, 0.5, 0.2, 0.7, 0.4, 0.6, 0.3, 0.5, 0.2, 0.4, 0.6, 0.3, 0.5, 0.4, 0.2, 0.6];
const RS = [0.3, 0.4, 0.1, 0.5, 0.1, 0.2, 0.8, 0.1, 0.1, 0.3, 0.4, 0.1, 0.4, 0.2, 0.3, 0.4];
const SUMS = XS.map((x, i) => x + RS[i]);
const WS = [1.1, 0.9, 1.0, 1.2, 0.8, 1.0, 0.9, 1.1, 1.0, 1.2, 0.9, 0.8, 1.1, 1.0, 0.9, 1.0];

function jobExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 1, 1.0, 0.14), 0.55, 0.002, -0.18));
  const k = 0.09;
  const k2 = 0.035;
  const cw = 0.04;
  const x0 = 0.28;
  const rows = { x: -0.55, r: -0.4, sum: -0.25, sq: -0.08, norm: 0.08 };
  const colX = (i: number) => x0 + (i + 0.5) * cw;
  const unit = new THREE.BoxGeometry(cw * 0.72, 1, 0.04);
  const barRow = (z: number, material: THREE.Material) => {
    const base = kit.solid(new THREE.BoxGeometry(cw * 16 + 0.012, 0.006, 0.054));
    base.position.set(x0 + cw * 8, 0.003, z);
    root.add(base);
    return SUMS.map((_, i) => {
      const bar = kit.solid(unit, { material, threshold: 40 });
      bar.position.set(colX(i), 0, z);
      root.add(bar);
      return bar;
    });
  };
  const setHeight = (bar: THREE.Group, value: number) => {
    const h = Math.max(0.0005, value);
    bar.scale.y = h;
    bar.position.y = 0.006 + h / 2;
  };
  const xBars = barRow(rows.x, kit.blue);
  const rBars = barRow(rows.r, kit.blue);
  const sumBars = barRow(rows.sum, kit.pink);
  const sqBars = barRow(rows.sq, kit.blue);
  const normBars = barRow(rows.norm, kit.pink);

  const label = (z: number, name: string, sub: string) => {
    const card = note(kit, fonts, 0.2, 0.08, () => [
      { text: name, color: kit.palette.ink, size: 0.44, serif: true },
      { text: sub, color: kit.palette.faint, size: 0.26 },
    ]);
    card.group.position.set(0.165, 0, z - 0.02);
    root.add(card.group);
  };
  label(rows.x, "x", "the layer's output");
  label(rows.r, "r", "the residual");
  label(rows.sum, "sum = x + r", "the next residual");
  label(rows.sq, "sum²", "one per value");
  label(rows.norm, "norm", "w · sum · rstd");

  let scale = 1;
  const rstdOf = (s: number) => 1 / Math.sqrt(SUMS.reduce((a, v) => a + (v * s) ** 2, 0) / SUMS.length + 1e-5);
  const rstdCard = note(kit, fonts, 0.5, 0.075, () => [
    { text: `rstd = 1 / √(mean of sum² + ε) = ${rstdOf(scale).toFixed(2)}`, color: kit.palette.ink, size: 0.3, serif: true },
    { text: "one number per row, shared by all 2,048 values", color: kit.palette.faint, size: 0.2 },
  ]);
  rstdCard.group.position.set(0.55, 0, 0.17);
  root.add(rstdCard.group);

  const railZ = 0.34;
  const knobFrom = 0.24;
  const knobSpan = 0.52;
  const slider = kit.solid(new THREE.BoxGeometry(knobSpan + 0.04, 0.014, 0.02));
  slider.position.set(knobFrom + knobSpan / 2, 0.007, railZ);
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.026, 26, 20), kit.pink);
  knob.castShadow = true;
  const scaleCard = note(kit, fonts, 0.17, 0.065, () => [
    { text: "scale the row", color: kit.palette.faint, size: 0.24 },
    { text: `x and r × ${scale.toFixed(2)}`, color: kit.palette.ink, size: 0.36, serif: true },
  ]);
  scaleCard.group.position.set(0.875, 0, railZ - 0.03);
  root.add(slider, knob, scaleCard.group);

  let time = 0;
  let held = false;
  let shown = "";
  const apply = () => {
    knob.position.set(knobFrom + ((scale - 0.4) / 1.6) * knobSpan, 0.04, railZ);
    const sumIn = smooth(1.2, 2.6, time);
    const sqIn = smooth(3.2, 4.6, time);
    const normIn = smooth(5.8, 7.2, time);
    const rstd = rstdOf(scale);
    SUMS.forEach((s, i) => {
      setHeight(xBars[i], XS[i] * scale * k);
      setHeight(rBars[i], RS[i] * scale * k);
      setHeight(sumBars[i], s * scale * k * sumIn);
      setHeight(sqBars[i], (s * scale) ** 2 * k2 * sqIn);
      setHeight(normBars[i], s * scale * rstd * WS[i] * k * 0.6 * normIn);
    });
    rstdCard.group.visible = smooth(4.6, 5.4, time) > 0.5;
    const key = scale.toFixed(2);
    if (key !== shown) {
      shown = key;
      rstdCard.redraw();
      scaleCard.redraw();
    }
  };
  apply();
  return {
    root,
    knob,
    setHeld: (value: boolean) => {
      held = value;
    },
    drag: (x: number) => {
      scale = Math.min(2, Math.max(0.4, 0.4 + ((x - knobFrom) / knobSpan) * 1.6));
      time = Math.max(time, 7.2);
      apply();
    },
    update: (dt: number, reduced: boolean) => {
      if (reduced || held) time = Math.max(time, 7.2);
      else time = (time + dt) % 12;
      apply();
    },
  };
}

function roundTripExhibit(kit: Kit, fonts: Fonts, numbers: AddRmsnormNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.pink, 1, 1.0, 0.12), 0.5, 0.002, -0.16));
  const dram = kit.solid(new THREE.BoxGeometry(0.12, 0.08, 0.84), { hull: true });
  dram.position.set(0.16, 0.04, -0.15);
  root.add(dram);
  const dramCard = note(kit, fonts, 0.14, 0.05, () => [
    { text: "DRAM", color: kit.palette.ink, size: 0.5, serif: true },
  ]);
  dramCard.group.position.set(0.16, 0, 0.3);
  root.add(dramCard.group);

  const kernel = (z: number, w: number, title: string, sub: string) => {
    const box = kit.solid(new THREE.BoxGeometry(w, 0.05, 0.08), { hull: true });
    box.position.set(0.42 + w / 2, 0.025, z);
    const card = note(kit, fonts, 0.3, 0.06, () => [
      { text: title, color: kit.palette.ink, size: 0.4, serif: true },
      { text: sub, color: kit.palette.faint, size: 0.22 },
    ]);
    card.group.position.set(0.42 + 0.15, 0, z + 0.05);
    root.add(box, card.group);
  };
  const zAdd = -0.54;
  const zNorm = -0.33;
  const zFused = -0.02;
  kernel(zAdd, 0.22, "add", "kernel 1: writes the sum out");
  kernel(zNorm, 0.22, "rmsnorm", "kernel 2: reads the sum back, twice");
  kernel(zFused, 0.26, "add_rmsnorm", "one kernel: the sum stays on the chip");

  const edgeX = 0.23;
  const kernelX = 0.41;
  const packets: { mesh: THREE.Group; at: (t: number) => THREE.Vector3; start: number }[] = [];
  const packet = (z: number, inbound: boolean, material: THREE.Material, start: number, lift = 0.07) => {
    const from = new THREE.Vector3(inbound ? edgeX : kernelX, 0.05, z);
    const to = new THREE.Vector3(inbound ? kernelX : edgeX, 0.05, z);
    const { line, at } = arcLine(kit, from, to, lift);
    const mesh = kit.solid(new THREE.BoxGeometry(0.036, 0.022, 0.022), { material });
    root.add(line, mesh);
    packets.push({ mesh, at, start });
  };
  packet(zAdd - 0.02, true, kit.blue, 0.3);
  packet(zAdd + 0.02, false, kit.pink, 1.8);
  packet(zNorm - 0.025, true, kit.pink, 3.3);
  packet(zNorm, true, kit.pink, 4.6, 0.1);
  packet(zNorm + 0.025, true, kit.blue, 3.3, 0.04);
  packet(zNorm + 0.025, false, kit.pink, 6.1, 0.1);
  packet(zFused - 0.02, true, kit.blue, 0.3);
  packet(zFused + 0.02, false, kit.pink, 1.8);

  const barX = 0.86;
  const tall = 0.2;
  const column = (z: number, kb: number, material: THREE.Material, text: string, sub: string) => {
    const bar = kit.solid(new THREE.BoxGeometry(0.07, 1, 0.07), { material });
    bar.position.set(barX, 0, z);
    const card = note(kit, fonts, 0.2, 0.06, () => [
      { text, color: kit.palette.ink, size: 0.38, serif: true },
      { text: sub, color: kit.palette.faint, size: 0.22 },
    ]);
    card.group.position.set(barX - 0.02, 0, z + 0.07);
    root.add(bar, card.group);
    return { bar, height: (tall * kb) / numbers.naiveKb };
  };
  const columns = [
    column(-0.44, numbers.naiveKb, kit.blue, `${numbers.naiveKb} KB`, "2 launches"),
    column(zFused - 0.01, numbers.fusedKb, kit.pink, `${numbers.fusedKb} KB`, "1 launch"),
  ];
  const headCard = note(kit, fonts, 0.3, 0.05, () => [
    { text: "DRAM bytes per launch, 64 tokens", color: kit.palette.faint, size: 0.34 },
  ]);
  headCard.group.position.set(0.78, 0, 0.16);
  root.add(headCard.group);

  let time = 0;
  const travel = 1.1;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 0 : (time + dt) % 10;
      for (const p of packets) {
        const t = reduced ? 0.5 : (time - p.start) / travel;
        p.mesh.visible = t >= 0 && t <= 1;
        if (p.mesh.visible) p.mesh.position.copy(p.at(smooth(0, 1, t)));
      }
      const grow = reduced ? 1 : smooth(0.3, 7.2, time);
      for (const c of columns) {
        const h = Math.max(0.0005, c.height * grow);
        c.bar.scale.y = h;
        c.bar.position.y = h / 2;
      }
    },
  };
}

const PHASES = [
  { text: "load x, r: 8 scalar loads of 2 bytes", w: 0.22 },
  { text: "add, write the sum, square it", w: 0.12 },
  { text: "reduce: shuffles, shared memory", w: 0.14, barrier: true },
  { text: "thread 0 writes rstd to shared", w: 0.06, barrier: true },
  { text: "read every sum back from memory", w: 0.2, back: true },
  { text: "scale by w, write the norm", w: 0.14 },
];

function firstFusedExhibit(kit: Kit, fonts: Fonts, numbers: AddRmsnormNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 1, 0.9, 0.14), 0.5, 0.002, -0.15));
  const x0 = 0.1;
  const span = 0.82;
  const total = PHASES.reduce((a, p) => a + p.w, 0);
  const z = -0.47;
  let start = 0;
  const phases = PHASES.map((p, i) => {
    const from = x0 + (start / total) * span;
    const w = (p.w / total) * span;
    start += p.w;
    const block = kit.solid(new THREE.BoxGeometry(w - 0.006, 0.03, 0.08));
    block.position.set(from + w / 2, 0.015, z);
    const fill = new THREE.Mesh(new THREE.BoxGeometry(1, 0.012, 0.064), p.back ? kit.pink : kit.blue);
    fill.castShadow = true;
    fill.position.set(from + 0.003, 0.036, z);
    const tag = note(kit, fonts, 0.04, 0.04, () => [{ text: String(i + 1), color: kit.palette.ink, size: 0.62, serif: true }]);
    tag.group.position.set(from + w / 2, 0, z + 0.06);
    root.add(block, fill, tag.group);
    if (p.barrier) {
      const wall = kit.solid(new THREE.BoxGeometry(0.006, 0.13, 0.13));
      wall.position.set(from + w, 0.065, z);
      root.add(wall);
    }
    return { from, w, fill };
  });
  const legend = note(kit, fonts, 0.86, 0.2, () => [
    ...PHASES.map((p, i) => ({ text: `${i + 1}  ${p.text}${p.barrier ? ", then a barrier" : ""}`, color: p.back ? kit.palette.pink : kit.palette.ink, size: 0.125 })),
  ], -1.0);
  legend.group.position.set(0.5, 0, -0.38);
  root.add(legend.group);

  const rodX = 0.3;
  const rodSpan = 0.6;
  const rod = (zz: number, us: number, material: THREE.Material, name: string) => {
    const bar = kit.solid(new THREE.BoxGeometry(1, 0.03, 0.05), { material });
    bar.position.set(rodX, 0.015, zz);
    const card = note(kit, fonts, 0.22, 0.075, () => [
      { text: name, color: kit.palette.faint, size: 0.26 },
      { text: `${us.toFixed(1)} µs`, color: kit.palette.ink, size: 0.42, serif: true },
    ]);
    card.group.position.set(0.17, 0, zz + 0.01);
    root.add(bar, card.group);
    return { bar, length: (rodSpan * us) / Math.max(numbers.pair, numbers.first) };
  };
  const rods = [rod(-0.08, numbers.pair, kit.blue, "naive pair, 2 launches"), rod(0.06, numbers.first, kit.pink, "first fused, 1 launch")];
  const footer = note(kit, fonts, 0.8, 0.08, () => [
    { text: `one token, µs per launch; at 1, 16 and 64 tokens: ${numbers.firstFlat}`, color: kit.palette.ink, size: 0.26 },
    { text: "the time does not grow with the rows: latency bound, not bandwidth", color: kit.palette.faint, size: 0.22 },
  ]);
  footer.group.position.set(0.5, 0, 0.2);
  root.add(footer.group);

  let time = 0;
  const run = 7;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? run : (time + dt) % (run + 2.5);
      const t = Math.min(1, time / run);
      let at = 0;
      for (const [i, p] of phases.entries()) {
        const a = at;
        at += PHASES[i].w / total;
        const done = smooth(a, at, t);
        const w = Math.max(0.0005, (p.w - 0.014) * done);
        p.fill.scale.x = w;
        p.fill.position.x = p.from + 0.004 + w / 2;
      }
      for (const r of rods) {
        const l = Math.max(0.0005, r.length * t);
        r.bar.scale.x = l;
        r.bar.position.x = rodX + l / 2;
      }
    },
  };
}

const STAGES = [
  { text: "x, r", sub: "read from memory", bf16: true },
  { text: "x + r", sub: "added in float", bf16: false },
  { text: "sum = round(x + r)", sub: "written out: the same bits as the naive add", bf16: true, round: true },
  { text: "sum², from the rounded sum", sub: "review fix: the first draft squared the float sum", bf16: false, fix: true },
  { text: "Σ sum² over the row, then rstd", sub: "the only step whose order may differ", bf16: false },
  { text: "round(sum · rstd)", sub: "review fix: it rounded once where the naive kernel rounds twice", bf16: true, round: true, fix: true },
  { text: "norm = round(w · that)", sub: "written out", bf16: true, round: true },
];

function roundingExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.pink, 0.9, 1.0, 0.1), 0.5, 0.002, -0.12));
  const z0 = -0.56;
  const gap = 0.145;
  const px = 0.13;
  const zOf = (i: number) => z0 + i * gap;
  const spine = kit.solid(new THREE.BoxGeometry(0.012, 0.01, gap * (STAGES.length - 1)), { edges: false });
  spine.position.set(px, 0.005, zOf(3));
  root.add(spine);
  STAGES.forEach((s, i) => {
    const plate = kit.solid(new THREE.CylinderGeometry(0.035, 0.04, 0.016, 32), { threshold: 35 });
    plate.position.set(px, 0.008, zOf(i));
    root.add(plate);
    if (s.round) {
      const ring = kit.solid(new THREE.TorusGeometry(0.03, 0.005, 10, 36), { edges: false });
      ring.position.set(px, 0.075, zOf(i) - gap / 2);
      root.add(ring);
    }
    const card = note(
      kit,
      fonts,
      0.58,
      0.085,
      () => [
        { text: s.text, color: kit.palette.ink, size: 0.36, serif: true },
        { text: s.sub, color: s.fix ? kit.palette.ink : kit.palette.faint, size: 0.2 },
      ],
      -1.1,
    );
    card.group.position.set(0.5, 0, zOf(i) - 0.04);
    const type = note(kit, fonts, 0.11, 0.05, () => [{ text: s.bf16 ? "bf16" : "float", color: s.bf16 ? kit.palette.pink : kit.palette.blue, size: 0.5 }], -1.1);
    type.group.position.set(0.875, 0, zOf(i) - 0.025);
    root.add(card.group, type.group);
  });
  const cube = kit.solid(new THREE.BoxGeometry(0.036, 0.036, 0.036), { material: kit.pink });
  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.022, 22, 16), kit.blue);
  ball.castShadow = true;
  root.add(cube, ball);

  let time = 0;
  const hop = 1.3;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? hop * (STAGES.length - 1) : (time + dt) % (hop * STAGES.length + 1.5);
      const stage = Math.min(STAGES.length - 1, Math.floor(time / hop));
      const local = Math.min(1, (time - stage * hop) / hop);
      const move = stage < STAGES.length - 1 ? smooth(0.35, 1, local) : 0;
      const next = Math.min(STAGES.length - 1, stage + 1);
      const z = zOf(stage) + (zOf(next) - zOf(stage)) * move;
      const y = 0.04 + Math.sin(Math.PI * move) * 0.04;
      const bf16 = move < 0.5 ? STAGES[stage].bf16 : STAGES[next].bf16;
      cube.visible = bf16;
      ball.visible = !bf16;
      cube.position.set(px, y, z);
      ball.position.set(px, y, z);
    },
  };
}

function vectorsExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 1, 1.0, 0.14), 0.55, 0.002, -0.14));
  const cw = 0.0105;
  const x0 = 0.22;
  const group = 8 * cw + 0.008;
  const cellX = (g: number, e: number) => x0 + g * group + (e + 0.5) * cw;
  const geometry = new THREE.BoxGeometry(cw - 0.0018, 0.016, 0.045);
  const row = (z: number) => {
    const cells: THREE.Mesh[] = [];
    for (let g = 0; g < 8; g++)
      for (let e = 0; e < 8; e++) {
        const cell = new THREE.Mesh(geometry, kit.ink);
        cell.castShadow = true;
        cell.receiveShadow = true;
        cell.position.set(cellX(g, e), 0.008, z);
        root.add(cell);
        cells.push(cell);
      }
    return cells;
  };
  const zFirst = -0.52;
  const zTuned = -0.3;
  const zRegs = -0.17;
  const first = row(zFirst);
  const tuned = row(zTuned);
  const regs: THREE.Group[] = [];
  for (let g = 0; g < 8; g++) {
    const tray = kit.solid(new THREE.BoxGeometry(8 * cw + 0.004, 0.006, 0.05), { edges: true });
    tray.position.set(cellX(g, 3.5), 0.003, zRegs);
    const load = kit.solid(new THREE.BoxGeometry(8 * cw - 0.002, 0.016, 0.04), { material: kit.blue });
    root.add(tray, load);
    regs.push(load);
  }
  const label = (z: number, lines: () => Line[]) => {
    const card = note(kit, fonts, 0.15, 0.07, lines);
    card.group.position.set(0.13, 0, z + 0.02);
    root.add(card.group);
  };
  label(zFirst, () => [
    { text: "first fused", color: kit.palette.ink, size: 0.3, serif: true },
    { text: "thread t reads t,", color: kit.palette.faint, size: 0.17 },
    { text: "t + 256, t + 512 …", color: kit.palette.faint, size: 0.17 },
  ]);
  label(zTuned, () => [
    { text: "tuned", color: kit.palette.ink, size: 0.3, serif: true },
    { text: "thread t reads 8t …", color: kit.palette.faint, size: 0.17 },
    { text: "8t + 7 in one load", color: kit.palette.faint, size: 0.17 },
  ]);
  label(zRegs, () => [
    { text: "registers", color: kit.palette.ink, size: 0.3, serif: true },
    { text: "the 8 rounded sums;", color: kit.palette.faint, size: 0.17 },
    { text: "pass 2 reads these", color: kit.palette.faint, size: 0.17 },
  ]);
  const axis = (z: number, text: (g: number) => string) => {
    const card = createCard(
      kit,
      8 * group,
      0.035,
      (ctx, W, H) => {
        ctx.fillStyle = kit.palette.faint;
        ctx.font = `${H * 0.55}px ${fonts.sans}`;
        ctx.textAlign = "center";
        for (let g = 0; g < 8; g++) ctx.fillText(text(g), ((g + 0.5) / 8) * W, H * 0.7);
        ctx.textAlign = "left";
      },
      -0.7,
    );
    card.group.position.set(x0 + 4 * group, 0, z + 0.035);
    root.add(card.group);
  };
  axis(zFirst, (g) => `round ${g}`);
  axis(zTuned, (g) => `t${g}`);

  const counters = [
    { z: 0.04, n: 32, len: 0.016, material: kit.ink, name: "first: 32 loads", sub: "× 2 bytes per thread" },
    { z: 0.17, n: 3, len: 0.016 * 8 + 0.0035 * 7, material: kit.blue, name: "tuned: 3 loads", sub: "× 16 bytes per thread" },
  ].map((c) => {
    const blocks: THREE.Group[] = [];
    for (let i = 0; i < c.n; i++) {
      const b = kit.solid(new THREE.BoxGeometry(c.len, 0.014, 0.04), { material: c.material, threshold: 40 });
      b.position.set(0.26 + i * (c.len + 0.0035) + c.len / 2, 0.007, c.z);
      root.add(b);
      blocks.push(b);
    }
    const card = note(kit, fonts, 0.17, 0.065, () => [
      { text: c.name, color: kit.palette.ink, size: 0.3, serif: true },
      { text: c.sub, color: kit.palette.faint, size: 0.2 },
    ]);
    card.group.position.set(0.13, 0, c.z + 0.02);
    root.add(card.group);
    return blocks;
  });
  const footer = note(kit, fonts, 0.78, 0.06, () => [
    { text: "global loads of x, r, sum and w per thread at hidden 2048; 2,048 values = 256 threads × 8", color: kit.palette.faint, size: 0.24 },
  ]);
  footer.group.position.set(0.53, 0, 0.27);
  root.add(footer.group);

  const blue = kit.blue;
  let time = 0;
  const round = 0.6;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 9 : (time + dt) % 10;
      first.forEach((cell, i) => {
        const g = Math.floor(i / 8);
        const e = i % 8;
        const t0 = 0.3 + g * round + e * (round / 8);
        const on = smooth(t0, t0 + 0.25, time);
        cell.material = on > 0.5 ? blue : kit.ink;
        cell.position.y = 0.008 + on * 0.008;
      });
      const lift = smooth(0.3, 1.0, time);
      const carry = smooth(1.2, 2.4, time);
      tuned.forEach((cell) => {
        cell.material = lift > 0.5 ? blue : kit.ink;
        cell.position.y = 0.008 + lift * 0.008;
      });
      regs.forEach((load, g) => {
        load.visible = lift > 0.5;
        load.position.set(cellX(g, 3.5), 0.03 + Math.sin(Math.PI * carry) * 0.05 - carry * 0.016, zTuned + (zRegs - zTuned) * carry);
      });
      counters.forEach((blocks, k) =>
        blocks.forEach((b, i) => {
          b.visible = (k === 0 ? 0.3 + (i / 32) * 8 * round : 0.3 + i * 0.8) <= time;
        }),
      );
    },
  };
}

const LANES = [3, 1, 4, 1, 5, 9, 2, 6];
const MASKS = [4, 2, 1];

function shufflesExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 1, 1.0, 0.14), 0.55, 0.002, -0.18));
  const levels = [LANES];
  for (const m of MASKS) {
    const prev = levels[levels.length - 1];
    levels.push(prev.map((v, j) => v + prev[j ^ m]));
  }
  const lx = (j: number) => 0.26 + j * 0.09;
  const lz = (k: number) => -0.54 + k * 0.19;
  const k = 0.0045;
  const pillars: { group: THREE.Group; height: number; level: number }[] = [];
  const ribbons: { mesh: THREE.Mesh; level: number }[] = [];
  levels.forEach((vals, level) => {
    const last = level === levels.length - 1;
    vals.forEach((v, j) => {
      const height = v * k;
      const pillar = kit.solid(new THREE.CylinderGeometry(0.018, 0.018, 1, 24), { material: last ? kit.pink : kit.blue, threshold: 40 });
      pillar.position.set(lx(j), 0, lz(level));
      root.add(pillar);
      pillars.push({ group: pillar, height, level });
      const card = note(kit, fonts, 0.05, 0.035, () => [{ text: String(v), color: kit.palette.ink, size: 0.62, serif: true }]);
      card.group.position.set(lx(j), 0, lz(level) + 0.035);
      root.add(card.group);
      if (level > 0) {
        const src = j ^ MASKS[level - 1];
        const from = new THREE.Vector3(lx(src), 0.004, lz(level - 1) + 0.065);
        const to = new THREE.Vector3(lx(j), 0.004, lz(level) - 0.022);
        const mid = new THREE.Vector3().lerpVectors(from, to, 0.5).setY(0.026);
        const mesh = ribbonTube([from, mid, to], 0.0024, kit.blue);
        root.add(mesh);
        ribbons.push({ mesh, level });
      }
    });
    const tag = note(kit, fonts, 0.12, 0.05, () =>
      level === 0
        ? [
            { text: "lane values", color: kit.palette.ink, size: 0.34, serif: true },
            { text: "partial sums", color: kit.palette.faint, size: 0.24 },
          ]
        : [
            { text: `xor ${MASKS[level - 1]}`, color: kit.palette.ink, size: 0.4, serif: true },
            { text: `lane j adds j ^ ${MASKS[level - 1]}`, color: kit.palette.faint, size: 0.22 },
          ],
    );
    tag.group.position.set(0.13, 0, lz(level) + 0.01);
    root.add(tag.group);
  });
  const footer = note(kit, fonts, 0.84, 0.1, () => [
    { text: "a real warp: 32 lanes, xor 16, 8, 4, 2, 1; then 8 warp totals, one barrier, xor again", color: kit.palette.ink, size: 0.19 },
    { text: "every thread now holds the total and computes rstd itself:", color: kit.palette.pink, size: 0.19 },
    { text: "no thread 0, no shared rstd, no second barrier", color: kit.palette.faint, size: 0.19 },
  ]);
  footer.group.position.set(0.52, 0, 0.27);
  root.add(footer.group);

  let time = 0;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 9 : (time + dt) % 11;
      for (const p of pillars) {
        const grow = p.level === 0 ? 1 : smooth(0.6 + p.level * 2, 1.6 + p.level * 2, time);
        const h = Math.max(0.0005, p.height * grow);
        p.group.scale.y = h;
        p.group.position.y = h / 2;
      }
      for (const r of ribbons) r.mesh.visible = time >= 0.2 + r.level * 2;
    },
  };
}

export function createAddRmsnormModel(kit: Kit, fonts: Fonts, numbers: AddRmsnormNumbers): BookModel {
  const job = jobExhibit(kit, fonts);
  const trip = roundTripExhibit(kit, fonts, numbers);
  const first = firstFusedExhibit(kit, fonts, numbers);
  const rounding = roundingExhibit(kit, fonts);
  const vectors = vectorsExhibit(kit, fonts);
  const shuffles = shufflesExhibit(kit, fonts);
  const parts = [job, trip, first, rounding, vectors, shuffles];
  return {
    exhibits: parts.map((p) => p.root),
    handles: [job.knob],
    dragStart: () => job.setHeld(true),
    drag: (x) => job.drag(x),
    dragEnd: () => job.setHeld(false),
    update: (dt, exhibit, _stop, reduced) => parts[exhibit]?.update(dt, reduced),
  };
}
