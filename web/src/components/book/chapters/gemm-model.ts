import * as THREE from "three";
import { bytesOf } from "@/components/kernels/gemm/figures";
import type { GemmNumbers } from "@/components/kernels/gemm/story";
import { createCard, type Kit, inkSegments, washPlane } from "../ink";
import type { BookModel } from "../book-scene";
import type { Fonts } from "../typeset";

export const GEMM_STOP = { job: 0, gemv: 1, roof: 2, batch: 3, trick: 6 };

const placed = <T extends THREE.Object3D>(object: T, x: number, y: number, z: number): T => {
  object.position.set(x, y, z);
  return object;
};
const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const approach = (value: number, goal: number, dt: number, reduced: boolean) => (reduced ? goal : value + (goal - value) * Math.min(1, dt * 4));
const fmt = (v: number) => Math.round(v).toLocaleString("en-US");

type Pen = (text: string, x: number, y: number, size: number, color: string, serif?: boolean) => void;
const pen =
  (ctx: CanvasRenderingContext2D, fonts: Fonts, w: number, h: number): Pen =>
  (text, x, y, size, color, serif = false) => {
    ctx.fillStyle = color;
    ctx.font = `${h * size}px ${serif ? fonts.serif : fonts.sans}`;
    ctx.fillText(text, w * x, h * y);
  };

function riser(kit: Kit, width: number, height: number, depth: number, material?: THREE.Material) {
  const pivot = new THREE.Group();
  const body = kit.solid(new THREE.BoxGeometry(width, height, depth), { material });
  body.position.y = height / 2;
  pivot.add(body);
  pivot.scale.y = 0.0001;
  return pivot;
}

function jobExhibit(kit: Kit, fonts: Fonts, n: GemmNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 1, 0.8, 0.14), 0.5, 0.002, -0.22));
  const maxBytes = Math.max(...n.projections.map(bytesOf));
  const maxUs = Math.max(...n.projections.map((p) => p.us));
  const towers: THREE.Group[] = [];
  n.projections.forEach((p, i) => {
    const x = 0.17 + i * 0.168;
    const z = -0.42;
    const weights = riser(kit, 0.058, 0.32 * (bytesOf(p) / maxBytes), 0.07);
    const time = riser(kit, 0.058, 0.32 * (p.us / maxUs), 0.07, kit.blue);
    weights.position.set(x - 0.033, 0, z);
    time.position.set(x + 0.033, 0, z);
    towers.push(weights, time);
    const card = createCard(kit, 0.16, 0.13, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      write(p.label, 0.05, 0.26, 0.24, kit.palette.ink, true);
      write(`[${p.out.toLocaleString("en-US")} × ${p.in.toLocaleString("en-US")}]${p.per > 1 ? ` × ${p.per}` : ""}`, 0.05, 0.46, 0.115, kit.palette.faint);
      write(`${fmt(bytesOf(p) / 1e6)} MB`, 0.05, 0.69, 0.19, kit.palette.ink);
      write(`${fmt(p.us)} µs`, 0.05, 0.92, 0.19, kit.palette.blue);
    });
    card.group.position.set(x, 0, z + 0.13);
    root.add(weights, time, card.group);
  });
  const summary = createCard(kit, 0.86, 0.22, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("ink: weights read per decode step", 0.03, 0.17, 0.1, kit.palette.ink);
    write("blue: GPU time per decode step, batch 1", 0.5, 0.17, 0.1, kit.palette.blue);
    write(`${(n.weightBytes / 1e9).toFixed(2)} GB of weights per token`, 0.03, 0.4, 0.14, kit.palette.ink, true);
    write(`these five own ${n.share.toFixed(1)}% of the GPU time`, 0.03, 0.6, 0.14, kit.palette.ink, true);
    write("the towers have the same shape: each projection takes as long as", 0.03, 0.79, 0.1, kit.palette.faint);
    write("its weights take to come out of DRAM", 0.03, 0.94, 0.1, kit.palette.faint);
  });
  summary.group.position.set(0.5, 0, 0.1);
  root.add(summary.group);
  let time = 0;
  return {
    root,
    enter: () => {
      time = 0;
    },
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 9 : Math.min(9, time + dt);
      towers.forEach((t, i) => {
        const start = 0.3 + Math.floor(i / 2) * 0.25 + (i % 2) * 0.12;
        t.scale.y = Math.max(0.0001, smooth(start, start + 1.2, time));
      });
    },
  };
}

function cellCard(kit: Kit, fonts: Fonts, size: number, label: string, lit: () => boolean) {
  const cell = new THREE.Group();
  const block = kit.solid(new THREE.BoxGeometry(size, 0.016, size));
  block.position.y = 0.008;
  const top = createCard(kit, size * 0.92, size * 0.92, (ctx, w, h) => {
    if (lit()) {
      ctx.fillStyle = kit.palette.blue;
      ctx.fillRect(0, 0, w, h);
    }
    ctx.textAlign = "center";
    pen(ctx, fonts, w, h)(label, 0.5, 0.64, 0.42, lit() ? kit.palette.paper : kit.palette.ink, true);
  }, -Math.PI / 2);
  top.group.position.set(0, 0.017, (-size * 0.92) / 2);
  cell.add(block, top.group);
  return { cell, redraw: top.redraw };
}

function trickExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 1, 0.9, 0.12), 0.5, 0.002, -0.2));
  const out = 3;
  const inn = 4;
  const size = 0.064;
  let active = 0;
  const cells: { group: THREE.Group; row: number; redraw: () => void }[] = [];
  const add = (r: number, c: number, x: number, z: number) => {
    const { cell, redraw } = cellCard(kit, fonts, size, `${r},${c}`, () => r === active);
    cell.position.set(x, 0, z);
    cells.push({ group: cell, row: r, redraw });
    root.add(cell);
  };
  for (let k = 0; k < out * inn; k++) add(Math.floor(k / inn), k % inn, 0.12 + k * 0.071, -0.54);
  for (let k = 0; k < out * inn; k++) add(Math.floor(k / inn), k % inn, 0.14 + (k % inn) * 0.078, -0.3 + Math.floor(k / inn) * 0.078);
  for (let k = 0; k < out * inn; k++) add(Math.floor(k / inn), k % inn, 0.64 + Math.floor(k / inn) * 0.078, -0.3 + (k % inn) * 0.078);
  const arrow = inkSegments([0.09, 0.004, -0.47, 0.94, 0.004, -0.47, 0.94, 0.004, -0.47, 0.92, 0.004, -0.48, 0.94, 0.004, -0.47, 0.92, 0.004, -0.46], kit.thin);
  const memory = createCard(kit, 0.84, 0.045, (ctx, w, h) => {
    pen(ctx, fonts, w, h)(`W in memory: [out = ${out}, in = ${inn}], one row after the other; address →`, 0.01, 0.72, 0.58, kit.palette.faint);
  }, -0.6);
  memory.group.position.set(0.5, 0, -0.45);
  const caption = (x: number, title: string, line: string) => {
    const card = createCard(kit, 0.36, 0.09, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      write(title, 0.03, 0.42, 0.34, kit.palette.ink, true);
      write(line, 0.03, 0.82, 0.24, kit.palette.faint);
    });
    card.group.position.set(x, 0, 0.08);
    root.add(card.group);
  };
  caption(0.29, "row-major [out, in]", "what the model stores");
  caption(0.73, "column-major, ld = in", "what cuBLAS sees: [in, out] = Wᵀ");
  const call = createCard(kit, 0.86, 0.19, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("cublasGemmEx(OP_T, OP_N, m = out, n = B, k = in,", 0.03, 0.2, 0.12, kit.palette.ink);
    write("W, lda = in, x, ldb = in, y, ldc = out)", 0.1, 0.38, 0.12, kit.palette.ink);
    write("C = op(A) · B = (Wᵀ)ᵀ · Xᵀ = W · Xᵀ = Yᵀ, written column-major", 0.03, 0.64, 0.13, kit.palette.ink, true);
    write("Yᵀ column-major is Y row-major: no copy, no transpose kernel", 0.03, 0.88, 0.13, kit.palette.blue);
  });
  call.group.position.set(0.5, 0, 0.24);
  root.add(arrow, memory.group, call.group);
  let time = 0;
  let shown = -1;
  const lifts = cells.map(() => 0);
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = (time + dt) % 9;
      const next = reduced ? 0 : Math.floor(time / 3);
      if (next !== shown) {
        shown = active = next;
        cells.forEach((c) => c.redraw());
      }
      cells.forEach((c, i) => {
        lifts[i] = approach(lifts[i], c.row === active ? 1 : 0, dt * 0.6, reduced);
        c.group.position.y = lifts[i] * 0.03;
      });
    },
  };
}

function gemvExhibit(kit: Kit, fonts: Fonts, n: GemmNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 1, 0.9, 0.12), 0.5, 0.002, -0.18));
  const drawn = 12;
  const x0 = 0.13;
  const length = 0.56;
  for (let k = 0; k < 16; k++) {
    const v = kit.solid(new THREE.BoxGeometry(0.03, 0.018, 0.034), { material: kit.pink });
    v.position.set(x0 + 0.017 + k * 0.035, 0.009, -0.5);
    root.add(v);
  }
  const xCard = createCard(kit, 0.25, 0.07, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("x: one row of 2,048 values", 0.03, 0.42, 0.3, kit.palette.pink, true);
    write("4 KB, stays in cache", 0.03, 0.82, 0.24, kit.palette.faint);
  });
  xCard.group.position.set(0.85, 0, -0.5);
  const rows: { fill: THREE.Mesh; y: THREE.Group }[] = [];
  for (let r = 0; r < drawn; r++) {
    const z = -0.4 + r * 0.036;
    const slab = kit.solid(new THREE.BoxGeometry(length, 0.012, 0.026));
    slab.position.set(x0 + length / 2, 0.006, z);
    const fill = new THREE.Mesh(new THREE.BoxGeometry(length, 0.014, 0.028), kit.blue);
    fill.geometry.translate(length / 2, 0.007, 0);
    fill.position.set(x0, 0, z);
    fill.scale.x = 0.0001;
    const y = riser(kit, 0.026, 0.04, 0.026, kit.blue);
    y.position.set(x0 + length + 0.05, 0, z);
    rows.push({ fill, y });
    root.add(slab, fill, y);
  }
  const wCard = createCard(kit, 0.19, 0.1, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("y: one value per row,", 0.04, 0.3, 0.2, kit.palette.blue, true);
    write("a dot product of", 0.04, 0.55, 0.17, kit.palette.faint);
    write("2,048 pairs", 0.04, 0.78, 0.17, kit.palette.faint);
  });
  wCard.group.position.set(0.885, 0, -0.26);
  const floorMs = n.weightBytes / (n.peak.gbs * 1e9) / 1e-3;
  const summary = createCard(kit, 0.84, 0.17, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write(`W, streamed from DRAM: ${drawn} of 2,560 QKV rows drawn`, 0.03, 0.17, 0.11, kit.palette.faint);
    write("each weight: 2 bytes read once, 1 multiply-add = 2 FLOPs", 0.03, 0.38, 0.13, kit.palette.ink, true);
    write("1 FLOP per byte: the speed of DRAM is the speed of the GEMV", 0.03, 0.58, 0.13, kit.palette.blue, true);
    write(`${(n.weightBytes / 1e9).toFixed(2)} GB per token at the measured ${n.peak.gbs} GB/s: ${floorMs.toFixed(2)} ms at best;`, 0.03, 0.78, 0.11, kit.palette.ink);
    write(`the engine with CUDA Graphs takes ${(n.tpotUs / 1000).toFixed(2)} ms`, 0.03, 0.94, 0.11, kit.palette.ink);
  });
  summary.group.position.set(0.5, 0, 0.13);
  root.add(xCard.group, wCard.group, summary.group);
  let time = 0;
  const slot = 0.65;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 0 : (time + dt) % (drawn * slot + 3.5);
      rows.forEach(({ fill, y }, r) => {
        const start = 0.4 + r * slot;
        const read = reduced ? 1 : smooth(start, start + slot * 1.1, time);
        fill.scale.x = Math.max(0.0001, read);
        fill.visible = read > 0 && read < 1;
        y.scale.y = Math.max(0.0001, reduced ? 1 : smooth(start + slot * 0.9, start + slot * 1.9, time));
      });
    },
  };
}

function roofExhibit(kit: Kit, fonts: Fonts, n: GemmNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 1, 0.8, 0.12), 0.5, 0.002, -0.22));
  const top = 0.26;
  const z = -0.32;
  const towers = n.roofRows.map((r, i) => {
    const x = 0.15 + i * 0.146;
    const tower = riser(kit, 0.056, top * (r.gbs / n.peak.gbs), 0.06, r.warn ? kit.pink : kit.blue);
    tower.position.set(x, 0, z);
    const card = createCard(kit, 0.136, 0.085, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      write(r.label, 0.05, 0.3, 0.27, r.warn ? kit.palette.pink : kit.palette.ink, true);
      write(`${r.gbs} GB/s`, 0.05, 0.6, 0.2, r.warn ? kit.palette.pink : kit.palette.blue);
      write(`${r.pct}% of roof`, 0.05, 0.88, 0.2, kit.palette.faint);
    });
    card.group.position.set(x, 0, z + 0.09);
    root.add(tower, card.group);
    return tower;
  });
  const roof = new THREE.Mesh(new THREE.PlaneGeometry(0.86, 0.14), kit.glass);
  roof.rotation.x = -Math.PI / 2;
  roof.position.set(0.517, top, z);
  const [a, b, c, d] = [0.087, 0.947, z - 0.07, z + 0.07];
  const edge = inkSegments([a, top, c, b, top, c, b, top, c, b, top, d, b, top, d, a, top, d, a, top, d, a, top, c], kit.line);
  const summary = createCard(kit, 0.84, 0.17, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write(`glass plane: the measured peak, ${n.peak.gbs} GB/s from a plain copy kernel`, 0.03, 0.18, 0.12, kit.palette.ink);
    write("the roof is a measurement, not a hard limit: 102% reads as \"at the roof\"", 0.03, 0.4, 0.12, kit.palette.faint);
    write("a faster GEMV would have to read fewer bytes:", 0.03, 0.64, 0.14, kit.palette.blue, true);
    write("smaller weights (8-bit, 4-bit) or more rows per read", 0.03, 0.86, 0.14, kit.palette.blue, true);
  });
  summary.group.position.set(0.5, 0, 0.06);
  root.add(roof, edge, summary.group);
  let time = 0;
  return {
    root,
    enter: () => {
      time = 0;
    },
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 9 : Math.min(9, time + dt);
      towers.forEach((t, i) => {
        const start = 0.3 + i * 0.18;
        t.scale.y = Math.max(0.0001, smooth(start, start + 1.4, time));
      });
    },
  };
}

function batchExhibit(kit: Kit, fonts: Fonts, n: GemmNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.pink, 1, 0.8, 0.1), 0.5, 0.002, -0.2));
  const x0 = 0.13;
  const length = 0.56;
  const most = 16;
  const wz = -0.52;
  const weight = kit.solid(new THREE.BoxGeometry(length, 0.016, 0.03));
  weight.position.set(x0 + length / 2, 0.008, wz);
  const pulse = new THREE.Mesh(new THREE.BoxGeometry(length, 0.018, 0.032), kit.blue);
  pulse.geometry.translate(length / 2, 0.009, 0);
  pulse.position.set(x0, 0, wz);
  const wCard = createCard(kit, 0.22, 0.07, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("one row of W", 0.04, 0.42, 0.3, kit.palette.ink, true);
    write("from DRAM, read once", 0.04, 0.82, 0.24, kit.palette.faint);
  });
  wCard.group.position.set(0.85, 0, wz - 0.01);
  root.add(weight, pulse, wCard.group);
  const rows = Array.from({ length: most }, (_, r) => {
    const z = -0.44 + r * 0.021;
    const x = new THREE.Group();
    const slab = kit.solid(new THREE.BoxGeometry(length, 0.008, 0.014), { material: kit.pink, edges: false });
    slab.position.set(x0 + length / 2, 0.004, z);
    x.add(slab);
    const y = riser(kit, 0.02, 0.03, 0.014, kit.blue);
    y.position.set(x0 + length + 0.05, 0, z);
    root.add(x, y);
    return { x, y, shown: r === 0 ? 1 : 0, out: 0 };
  });
  let batch = most;
  const readout = createCard(kit, 0.84, 0.075, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write(`${batch} row${batch > 1 ? "s" : ""}`, 0.02, 0.62, 0.5, kit.palette.pink, true);
    write(`each weight read once feeds ${batch} dot product${batch > 1 ? "s" : ""}: 2 × ${batch} FLOPs per 2 bytes = ${batch} FLOP per byte`, 0.19, 0.42, 0.22, kit.palette.ink);
    write(`still far left of the ridge at ${Math.round(n.ridge)} FLOP per byte, so memory still sets the time`, 0.19, 0.8, 0.22, kit.palette.faint);
  });
  readout.group.position.set(0.5, 0, -0.07);
  const railZ = 0.06;
  const railX = [0.15, 0.85];
  const rail = kit.solid(new THREE.BoxGeometry(0.72, 0.016, 0.02));
  rail.position.set(0.5, 0.008, railZ);
  const ticks: number[] = [];
  for (let b = 1; b <= most; b++) {
    const x = railX[0] + ((b - 1) / (most - 1)) * (railX[1] - railX[0]);
    ticks.push(x, 0.017, railZ - 0.016, x, 0.017, railZ + 0.016);
  }
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.028, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(railX[1], 0.045, railZ);
  const summary = createCard(kit, 0.84, 0.13, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("measured, five projections per decode step", 0.03, 0.22, 0.15, kit.palette.faint);
    write(`1 row: ${fmt(n.us1)} µs`, 0.03, 0.56, 0.24, kit.palette.ink, true);
    write(`16 rows: ${fmt(n.us16)} µs`, 0.5, 0.56, 0.24, kit.palette.blue, true);
    write(`gate/up: ${n.roof1.gateUp.toFixed(2)} TFLOP/s at 1 row, ${n.roof16.gateUp.toFixed(2)} at 16: the same time for 16 times the FLOPs`, 0.03, 0.88, 0.14, kit.palette.faint);
  });
  summary.group.position.set(0.5, 0, 0.17);
  root.add(readout.group, rail, inkSegments(ticks, kit.thin), knob, summary.group);
  const setBatch = (b: number) => {
    if (b === batch) return;
    batch = b;
    readout.redraw();
  };
  let held = false;
  let knobX = railX[1];
  let time = 0;
  return {
    root,
    knob,
    setHeld: (value: boolean) => {
      held = value;
    },
    drag: (x: number) => {
      knobX = Math.min(railX[1], Math.max(railX[0], x));
      setBatch(1 + Math.round(((knobX - railX[0]) / (railX[1] - railX[0])) * (most - 1)));
    },
    update: (dt: number, reduced: boolean) => {
      const goal = railX[0] + ((batch - 1) / (most - 1)) * (railX[1] - railX[0]);
      knob.position.x = held ? knobX : approach(knob.position.x, goal, dt * 0.7, reduced);
      time = reduced ? 0 : (time + dt) % 4.5;
      const read = reduced ? 1 : smooth(0.3, 1.6, time);
      pulse.scale.x = Math.max(0.0001, read);
      pulse.visible = read > 0 && read < 1;
      rows.forEach((r, i) => {
        r.shown = approach(r.shown, i < batch ? 1 : 0, dt * 0.5, reduced);
        r.x.scale.set(1, Math.max(0.0001, r.shown), Math.max(0.0001, r.shown));
        r.x.visible = r.shown > 0.01;
        const start = 1.5 + i * 0.05;
        const fill = reduced ? 1 : smooth(start, start + 0.8, time);
        r.y.scale.y = Math.max(0.0001, fill * r.shown);
      });
    },
  };
}

export function createGemmModel(kit: Kit, fonts: Fonts, numbers: GemmNumbers): BookModel {
  const job = jobExhibit(kit, fonts, numbers);
  const trick = trickExhibit(kit, fonts);
  const gemv = gemvExhibit(kit, fonts, numbers);
  const roof = roofExhibit(kit, fonts, numbers);
  const batch = batchExhibit(kit, fonts, numbers);
  return {
    exhibits: [job.root, trick.root, gemv.root, roof.root, batch.root],
    handles: [batch.knob],
    dragStart: () => batch.setHeld(true),
    drag: (x) => batch.drag(x),
    dragEnd: () => batch.setHeld(false),
    enter: (stop) => {
      if (stop === GEMM_STOP.job) job.enter();
      if (stop === GEMM_STOP.roof) roof.enter();
    },
    update: (dt, exhibit, _stop, reduced) => {
      if (exhibit === 0) job.update(dt, reduced);
      if (exhibit === 1) trick.update(dt, reduced);
      if (exhibit === 2) gemv.update(dt, reduced);
      if (exhibit === 3) roof.update(dt, reduced);
      if (exhibit === 4) batch.update(dt, reduced);
    },
  };
}
