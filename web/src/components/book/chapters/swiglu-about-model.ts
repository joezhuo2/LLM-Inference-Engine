import * as THREE from "three";
import { aboutGates, aboutPicked, aboutUps, FF, HIDDEN, PER_CELL, sigmoid } from "@/components/kernels/swiglu/about";
import { silu } from "@/components/kernels/swiglu/figures";
import type { BookModel } from "../book-scene";
import { createCard, inkSegments, type Kit, ribbonTube, washPlane } from "../ink";
import type { Fonts } from "../typeset";

const FF_CELLS = FF / PER_CELL;
const HIDDEN_CELLS = HIDDEN / PER_CELL;
const PITCH = 0.025;
const DEPTH = 0.019;
const WIDTH = 0.05;
const Z0 = -0.455;
const X_COL = 0.15;
const GATE_COL = 0.38;
const UP_COL = 0.455;
const OUT_COL = 0.66;
const RES_COL = 0.88;
const G_MIN = -4;
const G_MAX = 4;
const PLOT = [0.1, 0.56];
const CURVE_Z = 0.0;
const CURVE_Y = 0.028;
const BAR_UNIT = 0.03;
const CYCLE = 10;

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const fmt = (n: number) => n.toLocaleString("en-US");
const num = (n: number, digits = 2) => n.toFixed(digits).replace("-", "−");

type Pen = (text: string, x: number, y: number, size: number, color: string, serif?: boolean) => void;
const pen =
  (ctx: CanvasRenderingContext2D, fonts: Fonts, w: number, h: number): Pen =>
  (text, x, y, size, color, serif = false) => {
    ctx.fillStyle = color;
    ctx.font = `${h * size}px ${serif ? fonts.serif : fonts.sans}`;
    ctx.fillText(text, w * x, h * y);
  };

function bar(kit: Kit, material: THREE.Material | undefined, width = WIDTH, depth = DEPTH) {
  const pivot = new THREE.Group();
  const body = kit.solid(new THREE.BoxGeometry(width, 1, depth), { material, threshold: 40 });
  body.position.y = 0.5;
  pivot.add(body);
  return pivot;
}

const rowZ = (j: number) => Z0 + j * PITCH;
const hiddenZ = (k: number) => rowZ((FF_CELLS - HIDDEN_CELLS) / 2 + k);
const gx = (g: number) => PLOT[0] + ((g - G_MIN) / (G_MAX - G_MIN)) * (PLOT[1] - PLOT[0]);
const gateHeight = (g: number) => 0.004 + 0.012 * Math.max(0, silu(g));
const upHeight = (u: number) => 0.004 + 0.026 * u;
const outHeight = (v: number) => 0.004 + 0.012 * Math.max(0, v);

export function createSwigluAboutModel(kit: Kit, fonts: Fonts): BookModel {
  const root = new THREE.Group();
  const wash = washPlane(kit.palette.blue, 1, 0.55, 0.12);
  wash.position.set(0.5, 0.002, -0.3);
  root.add(wash);

  const title = createCard(kit, 0.86, 0.1, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("SwiGLU: Swish-Gated Linear Unit", 0.03, 0.42, 0.32, kit.palette.ink, true);
    write("a GLU, gated linear unit, whose gate goes through Swish, called SiLU in PyTorch", 0.03, 0.8, 0.15, kit.palette.faint);
  });
  title.group.position.set(0.5, 0, -0.6);
  root.add(title.group);

  const column = (x: number, count: number, z: (k: number) => number, height: number) =>
    Array.from({ length: count }, (_, k) => {
      const b = bar(kit, undefined);
      b.position.set(x, 0, z(k));
      b.scale.y = height;
      root.add(b);
      return b;
    });
  column(X_COL, HIDDEN_CELLS, hiddenZ, 0.012);
  const result = column(RES_COL, HIDDEN_CELLS, hiddenZ, 0.012);

  const gates = aboutGates.map((g, j) => {
    const b = bar(kit, kit.blue);
    b.position.set(GATE_COL, 0, rowZ(j));
    root.add(b);
    return b;
  });
  const ups = aboutUps.map((_, j) => {
    const b = bar(kit, kit.pink);
    b.position.set(UP_COL, 0, rowZ(j));
    root.add(b);
    return b;
  });
  const outs = aboutGates.map((_, j) => {
    const base = new THREE.Mesh(new THREE.BoxGeometry(WIDTH, 0.003, DEPTH), kit.ink);
    base.position.set(OUT_COL, 0.0015, rowZ(j));
    const b = bar(kit, kit.blue);
    b.position.set(OUT_COL, 0, rowZ(j));
    root.add(base, b);
    return b;
  });

  const midZ = rowZ((FF_CELLS - 1) / 2);
  const arrows: number[] = [];
  const arrow = (from: number, to: number) => {
    arrows.push(from, 0.003, midZ, to, 0.003, midZ, to - 0.016, 0.003, midZ - 0.01, to, 0.003, midZ, to - 0.016, 0.003, midZ + 0.01, to, 0.003, midZ);
  };
  arrow(X_COL + 0.04, GATE_COL - 0.04);
  arrow(UP_COL + 0.04, OUT_COL - 0.04);
  arrow(OUT_COL + 0.04, RES_COL - 0.04);
  const pz = rowZ(aboutPicked);
  const x0 = GATE_COL - 0.035;
  const x1 = OUT_COL + 0.035;
  const dz = PITCH / 2;
  arrows.push(x0, 0.002, pz - dz, x1, 0.002, pz - dz, x1, 0.002, pz - dz, x1, 0.002, pz + dz, x1, 0.002, pz + dz, x0, 0.002, pz + dz, x0, 0.002, pz + dz, x0, 0.002, pz - dz);
  root.add(inkSegments(arrows, kit.line));

  const labelZ = rowZ(FF_CELLS) + 0.012;
  const label = (x: number, width: number, head: string, sub: string, color: string) => {
    const card = createCard(kit, width, 0.07, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      write(head, 0.04, 0.46, 0.38, color, true);
      write(sub, 0.04, 0.84, 0.26, kit.palette.ink);
    });
    card.group.position.set(x, 0, labelZ);
    root.add(card.group);
  };
  label(X_COL, 0.17, `x: ${fmt(HIDDEN)}`, "hidden state", kit.palette.ink);
  label((GATE_COL + UP_COL) / 2, 0.24, "gate, up", `× W_gate, × W_up: ${fmt(FF)} each`, kit.palette.blue);
  label(OUT_COL, 0.17, "silu(g) × up", `${fmt(FF)}, one per pair`, kit.palette.blue);
  label(RES_COL, 0.17, `down: ${fmt(HIDDEN)}`, "× W_down", kit.palette.ink);

  const curvePoints = (f: (g: number) => number) => Array.from({ length: 41 }, (_, k) => {
    const g = G_MIN + (k / 40) * (G_MAX - G_MIN);
    return new THREE.Vector3(gx(g), 0.004 + CURVE_Y * f(g), CURVE_Z);
  });
  root.add(ribbonTube(curvePoints(silu), 0.0028, kit.blue));
  const sig: number[] = [];
  curvePoints(sigmoid).forEach((p, k, all) => {
    if (k > 0 && k % 2 === 1) sig.push(all[k - 1].x, all[k - 1].y, all[k - 1].z, p.x, p.y, p.z);
  });
  root.add(inkSegments(sig, kit.line));
  root.add(inkSegments([gx(G_MIN), 0.003, CURVE_Z, gx(G_MAX), 0.003, CURVE_Z, gx(0), 0.003, CURVE_Z, gx(0), 0.004 + CURVE_Y * 4, CURVE_Z], kit.thin));
  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.014, 22, 16), kit.blue);
  ball.castShadow = true;
  const dot = new THREE.Mesh(new THREE.SphereGeometry(0.009, 16, 12), kit.ink);
  root.add(ball, dot);
  const legend = createCard(kit, 0.48, 0.05, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("tube: silu(g) = g × sigmoid(g)", 0.03, 0.42, 0.34, kit.palette.blue);
    write("dashes: sigmoid(g), the first GLU's gate, 0 to 1", 0.03, 0.86, 0.3, kit.palette.ink);
  });
  legend.group.position.set(0.33, 0, CURVE_Z + 0.03);
  root.add(legend.group);

  const upBar = bar(kit, kit.pink, 0.06, 0.05);
  upBar.position.set(0.7, 0, CURVE_Z);
  upBar.scale.y = BAR_UNIT;
  const passBar = bar(kit, kit.blue, 0.06, 0.05);
  passBar.position.set(0.84, 0, CURVE_Z);
  root.add(upBar, passBar);
  const barCard = createCard(kit, 0.34, 0.05, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("up, outlined value", 0.04, 0.42, 0.3, kit.palette.pink);
    write("what passes", 0.48, 0.42, 0.3, kit.palette.blue);
    write("× up", 0.04, 0.86, 0.28, kit.palette.ink);
    write("silu(g) × up", 0.48, 0.86, 0.28, kit.palette.ink);
  });
  barCard.group.position.set(0.77, 0, CURVE_Z + 0.03);
  root.add(barCard.group);

  const railZ = 0.15;
  const rail = kit.solid(new THREE.BoxGeometry(PLOT[1] - PLOT[0] + 0.04, 0.016, 0.02));
  rail.position.set((PLOT[0] + PLOT[1]) / 2, 0.008, railZ);
  const ticks: number[] = [];
  for (let g = G_MIN; g <= G_MAX; g++) ticks.push(gx(g), 0.017, railZ - 0.018, gx(g), 0.017, railZ + 0.018);
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.028, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(gx(aboutGates[aboutPicked]), 0.045, railZ);
  const railCard = createCard(kit, 0.34, 0.05, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("drag: the outlined value's gate", 0.04, 0.42, 0.3, kit.palette.pink);
    write(`g from ${num(G_MIN, 0)} to ${G_MAX}, under the curve`, 0.04, 0.86, 0.28, kit.palette.faint);
  });
  railCard.group.position.set(0.77, 0, railZ - 0.02);
  root.add(rail, inkSegments(ticks, kit.thin), knob, railCard.group);

  const u = aboutUps[aboutPicked];
  let g = aboutGates[aboutPicked];
  let shown = Number.NaN;
  const readout = createCard(kit, 0.86, 0.1, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    const q = Math.round(g * 10) / 10;
    const s = sigmoid(q);
    const f = silu(q);
    write(`gate g = ${num(q, 1)}:  sigmoid(g) = ${num(s)},  silu(g) = ${num(q, 1)} × ${num(s)} = ${num(f)}`, 0.03, 0.34, 0.24, kit.palette.blue, true);
    write(`out = silu(g) × up = ${num(f)} × ${num(u, 0)} = ${num(f * u)}`, 0.03, 0.62, 0.21, kit.palette.ink);
    write(q < -2 ? "a negative gate: silu is about 0, this up value is shut off" : q > 2 ? "a large gate: sigmoid is near 1, up passes, scaled by about g" : "near 0: the gate lets part of up through", 0.03, 0.88, 0.2, kit.palette.faint);
  });
  readout.group.position.set(0.5, 0, 0.215);
  const note = createCard(kit, 0.86, 0.04, (ctx, w, h) => {
    pen(ctx, fonts, w, h)(`illustrative: each cell stands for ${PER_CELL} values; gate and up values made up, curves exact`, 0.03, 0.7, 0.5, kit.palette.faint);
  });
  note.group.position.set(0.5, 0, 0.37);
  root.add(readout.group, note.group);

  let held = false;
  let sweep = Math.asin(Math.min(1, g / 3.6));
  let time = 0;
  const apply = () => {
    const f = silu(g);
    ball.position.set(gx(g), 0.004 + CURVE_Y * f + 0.012, CURVE_Z);
    dot.position.set(gx(g), 0.004 + CURVE_Y * sigmoid(g) + 0.008, CURVE_Z);
    passBar.scale.y = Math.max(0.002, BAR_UNIT * f * u);
    upBar.scale.y = BAR_UNIT * u;
    const q = Math.round(g * 10) / 10;
    if (q !== shown) {
      shown = q;
      readout.redraw();
    }
  };
  apply();

  return {
    exhibits: [root],
    handles: [knob],
    dragStart: () => {
      held = true;
    },
    drag: (x) => {
      const k = Math.min(PLOT[1], Math.max(PLOT[0], x));
      g = G_MIN + ((k - PLOT[0]) / (PLOT[1] - PLOT[0])) * (G_MAX - G_MIN);
    },
    dragEnd: () => {
      held = false;
      sweep = Math.asin(Math.max(-1, Math.min(1, g / 3.6)));
    },
    enter: () => {
      time = 0;
    },
    update: (dt, _exhibit, _stop, reduced) => {
      time = reduced ? CYCLE - 1.5 : (time + dt) % CYCLE;
      if (!held && !reduced) {
        sweep += dt * 0.3;
        g = 3.6 * Math.sin(sweep);
      }
      knob.position.x = gx(g);
      apply();
      const fade = reduced ? 1 : 1 - smooth(CYCLE - 0.9, CYCLE, time);
      const rise = reduced ? 1 : smooth(0.3, 1.5, time) * fade;
      aboutGates.forEach((g0, j) => {
        const gj = j === aboutPicked ? g : g0;
        gates[j].scale.y = Math.max(0.0001, gateHeight(gj) * Math.max(rise, j === aboutPicked ? 1 : 0));
        ups[j].scale.y = Math.max(0.0001, upHeight(aboutUps[j]) * Math.max(rise, j === aboutPicked ? 1 : 0));
        const grown = j === aboutPicked ? 1 : reduced ? 1 : smooth(2 + j * 0.35, 2.9 + j * 0.35, time) * fade;
        const h = outHeight(silu(gj) * aboutUps[j]) * grown;
        outs[j].scale.y = Math.max(0.0001, h);
        outs[j].visible = h > 0.001;
      });
      const done = reduced ? 1 : smooth(6.6, 7.6, time) * fade;
      result.forEach((r) => {
        r.scale.y = Math.max(0.0001, 0.012 * done);
        r.visible = done > 0.02;
      });
    },
  };
}
