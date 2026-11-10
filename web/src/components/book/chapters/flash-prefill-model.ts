import * as THREE from "three";
import { fragmentCells } from "@/components/kernels/flash-prefill/figures";
import type { BookModel } from "../book-scene";
import { createCard, inkSegments, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";

export type FlashModelNumbers = { tile: number; half: number; calls: number; headDim: number; lanes: number };

const C = 0.018;
const AX = 0.1;
const AZ = -0.188;
const GAP = 0.024;
const CALL = 1.3;
const REST = 1.6;
const SCORE_H = 0.07;
const OPERAND_H = 0.014;

const approach = (value: number, goal: number, dt: number, reduced: boolean) => (reduced ? goal : value + (goal - value) * Math.min(1, dt * 2.2));

type Pen = (text: string, x: number, y: number, size: number, color: string, serif?: boolean) => void;
const pen =
  (ctx: CanvasRenderingContext2D, fonts: Fonts, w: number, h: number): Pen =>
  (text, x, y, size, color, serif = false) => {
    ctx.fillStyle = color;
    ctx.font = `${h * size}px ${serif ? fonts.serif : fonts.sans}`;
    ctx.fillText(text, w * x, h * y);
  };

function board(kit: Kit, x: number, z: number, cols: number, rows: number) {
  const group = new THREE.Group();
  const slab = kit.solid(new THREE.BoxGeometry(cols * C + 0.008, 0.004, rows * C + 0.008), { shadow: false });
  slab.position.set(x + (cols * C) / 2, 0.002, z + (rows * C) / 2);
  const lines: number[] = [];
  for (let c = 1; c < cols; c++) lines.push(x + c * C, 0.0046, z, x + c * C, 0.0046, z + rows * C);
  for (let r = 1; r < rows; r++) lines.push(x, 0.0046, z + r * C, x + cols * C, 0.0046, z + r * C);
  group.add(slab, inkSegments(lines, kit.thin, 0.0004));
  return group;
}

export function createFlashPrefillModel(kit: Kit, fonts: Fonts, n: FlashModelNumbers): BookModel {
  const root = new THREE.Group();
  const bx = AX + n.tile * C + GAP;
  const bz = AZ - GAP - n.tile * C;
  const wash = washPlane(kit.palette.blue, 0.3, 0.5, 0.12);
  wash.position.set(bx + (n.half * C) / 2, 0.002, AZ + (n.tile * C) / 2);
  root.add(wash, board(kit, AX, AZ, n.tile, n.tile), board(kit, bx, bz, n.half, n.tile), board(kit, bx, AZ, n.half, n.tile));

  const title = createCard(kit, 0.86, 0.1, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("mma.sync m16n8k16: one tensor core instruction for a whole warp", 0.03, 0.4, 0.28, kit.palette.ink, true);
    write(`C (${n.tile} × ${n.half} scores, FP32) += A (${n.tile} query rows × ${n.tile} dims) × B (${n.tile} dims × ${n.half} keys), both BF16`, 0.03, 0.82, 0.2, kit.palette.ink);
  });
  title.group.position.set(0.5, 0, -0.6);
  root.add(title.group);

  const tag = (x: number, z: number, width: number, head: string, sub: string, color: string) => {
    const card = createCard(kit, width, 0.065, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      write(head, 0.05, 0.46, 0.4, color, true);
      write(sub, 0.05, 0.86, 0.26, kit.palette.ink);
    });
    card.group.position.set(x, 0, z);
    root.add(card.group);
  };
  tag(AX + 0.144, bz + 0.2, 0.3, "A: the query tile", `${n.tile} rows × ${n.tile} of the head's ${n.headDim} dims`, kit.palette.ink);
  tag(AX + 0.144, bz + 0.07, 0.3, "B: half a key tile", `${n.tile} dims × ${n.half} keys, one key per column`, kit.palette.ink);
  tag(0.76, AZ + 0.12, 0.26, "C: the scores", "blue towers, in FP32", kit.palette.blue);

  const operand = (w: number, d: number) => {
    const g = kit.solid(new THREE.BoxGeometry(w * C - 0.003, OPERAND_H, d * C - 0.003), { shadow: false });
    root.add(g);
    return g;
  };
  const aBlocks = Array.from({ length: 4 }, () => operand(2, 1));
  const bBlocks = Array.from({ length: 2 }, () => operand(1, 2));
  const tower = new THREE.BoxGeometry(2 * C - 0.004, 1, C - 0.004);
  const cBlocks = Array.from({ length: 2 }, () => {
    const mesh = new THREE.Mesh(tower, kit.blue);
    mesh.castShadow = true;
    root.add(mesh);
    return mesh;
  });

  let lane = 5;
  let call = n.calls;
  const info = createCard(kit, 0.32, 0.2, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    const f = fragmentCells(lane);
    write(`lane ${lane} of ${n.lanes} holds`, 0.05, 0.17, 0.13, kit.palette.pink, true);
    write(`A: rows ${f.rows[0]} and ${f.rows[1]},`, 0.05, 0.34, 0.095, kit.palette.ink);
    write(`dims ${f.pairs.join(", ")} of this call`, 0.05, 0.47, 0.095, kit.palette.ink);
    write(`B: key ${f.group}, the same 4 dims`, 0.05, 0.63, 0.095, kit.palette.ink);
    write(`C: rows ${f.rows[0]} and ${f.rows[1]}`, 0.05, 0.79, 0.095, kit.palette.blue);
    write(`against keys ${f.pairs[0]} and ${f.pairs[1]}`, 0.05, 0.92, 0.095, kit.palette.blue);
  });
  info.group.position.set(0.76, 0, bz + 0.17);
  root.add(info.group);

  const readout = createCard(kit, 0.86, 0.1, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write(`call ${call} of ${n.calls}: dims ${(call - 1) * n.tile} to ${call * n.tile - 1} of the head, added to the same scores`, 0.03, 0.4, 0.26, kit.palette.ink, true);
    write(`${n.calls} calls finish half a key tile; nothing is passed between lanes`, 0.03, 0.82, 0.21, kit.palette.ink);
  });
  readout.group.position.set(0.5, 0, 0.13);
  root.add(readout.group);

  const railZ = 0.26;
  const railX = [0.2, 0.8];
  const rail = kit.solid(new THREE.BoxGeometry(railX[1] - railX[0] + 0.04, 0.016, 0.02));
  rail.position.set(0.5, 0.008, railZ);
  const tickX = (k: number) => railX[0] + (k / (n.lanes - 1)) * (railX[1] - railX[0]);
  const ticks: number[] = [];
  for (let k = 0; k < n.lanes; k += 4) ticks.push(tickX(k), 0.017, railZ - 0.018, tickX(k), 0.017, railZ + 0.018);
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.028, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(tickX(lane), 0.045, railZ);
  const railCard = createCard(kit, 0.6, 0.045, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("which lane of the warp", 0.02, 0.7, 0.5, kit.palette.faint);
    write(`0 to ${n.lanes - 1}`, 0.8, 0.7, 0.5, kit.palette.pink);
  });
  railCard.group.position.set(0.5, 0, railZ + 0.06);
  root.add(rail, inkSegments(ticks, kit.thin), knob, railCard.group);

  const show = (l: number, k: number) => {
    if (l !== lane) {
      lane = l;
      info.redraw();
    }
    if (k !== call) {
      call = k;
      readout.redraw();
    }
  };

  const place = (object: THREE.Object3D, x: number, y: number, z: number, dt: number, reduced: boolean, jump: boolean) => {
    if (jump || reduced) object.position.set(x, y, z);
    else object.position.set(approach(object.position.x, x, dt * 1.6, false), y, approach(object.position.z, z, dt * 1.6, false));
  };

  const tour = [5, 18, 31, 0];
  let held = false;
  let chosen = false;
  let knobX = tickX(lane);
  let clock = 0;
  let height = SCORE_H;
  let first = true;
  return {
    exhibits: [root],
    handles: [knob],
    dragStart: () => {
      held = true;
      chosen = true;
    },
    drag: (px) => {
      knobX = Math.min(railX[1], Math.max(railX[0], px));
      show(Math.round(((knobX - railX[0]) / (railX[1] - railX[0])) * (n.lanes - 1)), call);
    },
    dragEnd: () => {
      held = false;
    },
    enter: () => {
      clock = 0;
      chosen = false;
      first = true;
    },
    update: (dt, _exhibit, _stop, reduced) => {
      const span = n.calls * CALL + REST;
      if (reduced) show(chosen ? lane : tour[0], n.calls);
      else {
        clock = (clock + dt) % (span * tour.length);
        const t = clock % span;
        show(chosen ? lane : tour[Math.floor(clock / span)], Math.min(n.calls, 1 + Math.floor(t / CALL)));
      }
      knob.position.x = held ? knobX : approach(knob.position.x, tickX(lane), dt * 0.7, reduced);
      const f = fragmentCells(lane);
      const y = 0.004 + OPERAND_H / 2;
      aBlocks.forEach((b, i) => place(b, AX + (f.pairs[(i >> 1) * 2] + 1) * C, y, AZ + (f.rows[i & 1] + 0.5) * C, dt, reduced, first));
      bBlocks.forEach((b, i) => place(b, bx + (f.group + 0.5) * C, y, bz + (f.pairs[i * 2] + 1) * C, dt, reduced, first));
      height = approach(height, (SCORE_H * call) / n.calls, dt * 1.6, reduced);
      cBlocks.forEach((b, i) => {
        place(b, bx + (f.pairs[0] + 1) * C, 0.004 + height / 2, AZ + (f.rows[i] + 0.5) * C, dt, reduced, first);
        b.scale.y = Math.max(0.0001, height);
      });
      first = false;
    },
  };
}
