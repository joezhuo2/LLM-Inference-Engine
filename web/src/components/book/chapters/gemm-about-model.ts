import * as THREE from "three";
import type { BookModel } from "../book-scene";
import { createCard, inkSegments, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";

const K = 4;
const N = 3;
const MOST = 4;
const CELL = 0.05;
const PITCH = 0.062;
const H = 0.026;
const LIFT = 0.035;
const ROW_Z = -0.33;
const X_LEFT = 0.07;
const W_LEFT = 0.41;
const Y_LEFT = 0.73;
const W_TOP = ROW_Z - 0.04;
const CYCLE = 6.4;
const SLOT = 1.5;

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const bump = (a: number, b: number, x: number) => smooth(a, a + 0.35, x) * (1 - smooth(b - 0.35, b, x));
const approach = (value: number, goal: number, dt: number, reduced: boolean) => (reduced ? goal : value + (goal - value) * Math.min(1, dt * 4));

type Pen = (text: string, x: number, y: number, size: number, color: string, serif?: boolean) => void;
const pen =
  (ctx: CanvasRenderingContext2D, fonts: Fonts, w: number, h: number): Pen =>
  (text, x, y, size, color, serif = false) => {
    ctx.fillStyle = color;
    ctx.font = `${h * size}px ${serif ? fonts.serif : fonts.sans}`;
    ctx.fillText(text, w * x, h * y);
  };

function cube(kit: Kit, material?: THREE.Material, height = H) {
  const pivot = new THREE.Group();
  const body = kit.solid(new THREE.BoxGeometry(CELL, height, CELL), { material });
  body.position.y = height / 2;
  pivot.add(body);
  return pivot;
}

export function createGemmAboutModel(kit: Kit, fonts: Fonts): BookModel {
  const root = new THREE.Group();
  const wash = washPlane(kit.palette.blue, 1, 0.5, 0.12);
  wash.position.set(0.5, 0.002, -0.24);
  root.add(wash);

  const title = createCard(kit, 0.86, 0.11, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("GEMM: GEneral Matrix-Matrix multiply", 0.03, 0.34, 0.24, kit.palette.ink, true);
    write("GEMV: GEneral Matrix-Vector multiply", 0.03, 0.66, 0.24, kit.palette.ink, true);
    write("routine names from BLAS, the Basic Linear Algebra Subprograms (1979 on)", 0.03, 0.9, 0.13, kit.palette.faint);
  });
  title.group.position.set(0.5, 0, -0.56);
  root.add(title.group);

  const xRows = Array.from({ length: MOST }, (_, r) => {
    const row = new THREE.Group();
    for (let k = 0; k < K; k++) {
      const c = cube(kit, kit.pink);
      c.position.set(X_LEFT + CELL / 2 + k * PITCH, 0, ROW_Z + r * PITCH);
      row.add(c);
    }
    root.add(row);
    return { row, shown: r === 0 ? 1 : 0 };
  });

  const wColumns = Array.from({ length: N }, (_, j) => {
    const col = new THREE.Group();
    for (let k = 0; k < K; k++) {
      const c = cube(kit);
      c.position.set(W_LEFT + CELL / 2 + j * PITCH, 0, W_TOP + k * PITCH);
      col.add(c);
    }
    const glow = new THREE.Mesh(new THREE.BoxGeometry(CELL + 0.012, 0.004, K * PITCH), kit.blue);
    glow.position.set(W_LEFT + CELL / 2 + j * PITCH, 0.002, W_TOP + ((K - 1) * PITCH) / 2);
    glow.visible = false;
    root.add(col, glow);
    return { col, glow };
  });

  const outputs = Array.from({ length: MOST }, (_, r) =>
    Array.from({ length: N }, (_, j) => {
      const base = new THREE.Mesh(new THREE.BoxGeometry(CELL, 0.003, CELL), kit.ink);
      base.position.set(Y_LEFT + CELL / 2 + j * PITCH, 0.0015, ROW_Z + r * PITCH);
      const fill = cube(kit, kit.blue, 0.05);
      fill.position.set(Y_LEFT + CELL / 2 + j * PITCH, 0, ROW_Z + r * PITCH);
      fill.scale.y = 0.0001;
      root.add(base, fill);
      return { base, fill };
    }),
  );

  const signs: number[] = [];
  const dot = new THREE.Mesh(new THREE.SphereGeometry(0.009, 16, 12), kit.ink);
  dot.position.set(W_LEFT - 0.055, 0.009, ROW_Z);
  root.add(dot);
  const eq = Y_LEFT - 0.07;
  signs.push(eq - 0.016, 0.003, ROW_Z - 0.007, eq + 0.016, 0.003, ROW_Z - 0.007, eq - 0.016, 0.003, ROW_Z + 0.007, eq + 0.016, 0.003, ROW_Z + 0.007);
  root.add(inkSegments(signs, kit.line));

  const labelZ = ROW_Z + MOST * PITCH + 0.01;
  const label = (x: number, width: number, head: string, sub: string, color: string) => {
    const card = createCard(kit, width, 0.08, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      write(head, 0.04, 0.5, 0.46, color, true);
      write(sub, 0.04, 0.86, 0.24, kit.palette.ink);
    });
    card.group.position.set(x + width / 2, 0, labelZ);
    root.add(card.group);
  };
  label(X_LEFT - 0.01, 0.29, "x", "activations, one row per token", kit.palette.pink);
  label(W_LEFT - 0.01, 0.25, "Wᵀ", "weights, learned", kit.palette.ink);
  label(Y_LEFT - 0.03, 0.25, "y", "outputs", kit.palette.blue);

  let batch = 1;
  const readout = createCard(kit, 0.86, 0.1, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    const kind = batch === 1 ? "GEMV" : "GEMM";
    write(`${batch} row${batch > 1 ? "s" : ""} of x: a ${kind}`, 0.03, 0.36, 0.3, batch === 1 ? kit.palette.pink : kit.palette.blue, true);
    write(`${batch} × ${N} outputs × ${K} pairs = ${batch * N * K} multiply-adds`, 0.03, 0.64, 0.21, kit.palette.ink);
    write(`each weight is read once and used ${batch === 1 ? "once" : `${batch} times`}`, 0.03, 0.9, 0.21, kit.palette.ink);
  });
  readout.group.position.set(0.5, 0, 0.06);
  root.add(readout.group);

  const railZ = 0.22;
  const railX = [0.2, 0.8];
  const rail = kit.solid(new THREE.BoxGeometry(railX[1] - railX[0] + 0.04, 0.016, 0.02));
  rail.position.set(0.5, 0.008, railZ);
  const ticks: number[] = [];
  for (let b = 1; b <= MOST; b++) {
    const x = railX[0] + ((b - 1) / (MOST - 1)) * (railX[1] - railX[0]);
    ticks.push(x, 0.017, railZ - 0.018, x, 0.017, railZ + 0.018);
  }
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.028, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(railX[0], 0.045, railZ);
  const railCard = createCard(kit, 0.6, 0.045, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("rows of x", 0.02, 0.7, 0.5, kit.palette.faint);
    write("1: GEMV", 0.3, 0.7, 0.5, kit.palette.pink);
    write("2 to 4: GEMM", 0.58, 0.7, 0.5, kit.palette.blue);
  });
  railCard.group.position.set(0.5, 0, railZ + 0.06);
  const note = createCard(kit, 0.86, 0.04, (ctx, w, h) => {
    pen(ctx, fonts, w, h)("illustrative sizes: TinyLlama's W has 2,048 to 32,000 rows of 2,048 or 5,632 values", 0.03, 0.7, 0.5, kit.palette.faint);
  });
  note.group.position.set(0.5, 0, 0.38);
  root.add(rail, inkSegments(ticks, kit.thin), knob, railCard.group, note.group);

  const setBatch = (b: number) => {
    if (b === batch) return;
    batch = b;
    readout.redraw();
  };

  let held = false;
  let knobX = railX[0];
  let time = 0;
  return {
    exhibits: [root],
    handles: [knob],
    dragStart: () => {
      held = true;
    },
    drag: (x) => {
      knobX = Math.min(railX[1], Math.max(railX[0], x));
      setBatch(1 + Math.round(((knobX - railX[0]) / (railX[1] - railX[0])) * (MOST - 1)));
    },
    dragEnd: () => {
      held = false;
    },
    enter: () => {
      time = 0;
    },
    update: (dt, _exhibit, _stop, reduced) => {
      const goal = railX[0] + ((batch - 1) / (MOST - 1)) * (railX[1] - railX[0]);
      knob.position.x = held ? knobX : approach(knob.position.x, goal, dt * 0.7, reduced);
      time = reduced ? CYCLE - 1 : (time + dt) % CYCLE;
      const fade = reduced ? 1 : 1 - smooth(CYCLE - 0.6, CYCLE, time);
      xRows.forEach((x, r) => {
        x.shown = approach(x.shown, r < batch ? 1 : 0, dt * 0.45, reduced);
        x.row.scale.set(1, Math.max(0.0001, x.shown), 1);
        x.row.visible = x.shown > 0.01;
      });
      let active = 0;
      wColumns.forEach((w, j) => {
        const start = 0.3 + j * SLOT;
        const on = reduced ? 0 : bump(start, start + SLOT, time);
        active = Math.max(active, on);
        w.col.position.y = LIFT * on;
        w.glow.visible = on > 0.05;
        w.glow.position.y = 0.002 + LIFT * on;
        outputs.forEach((row, r) => {
          const grown = reduced ? 1 : smooth(start + 0.2, start + SLOT - 0.2, time);
          const height = grown * fade * xRows[r].shown;
          row[j].fill.scale.y = Math.max(0.0001, height);
          row[j].fill.visible = height > 0.02;
          row[j].base.visible = r < batch || xRows[r].shown > 0.01;
        });
      });
      xRows.forEach((x) => (x.row.position.y = LIFT * 0.6 * active));
    },
  };
}
