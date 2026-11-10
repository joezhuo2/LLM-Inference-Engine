import * as THREE from "three";
import type { BookModel } from "../book-scene";
import { createCard, inkSegments, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";

export type FlashAboutNumbers = { tile: number; tableMb: number; tileBytes: number; positions: number };

const N = 8;
const S = 0.05;
const X0 = 0.14;
const Z0 = -0.4;
const STEP = 1.5;
const REST = 2.4;
const DONE_H = 0.012;
const LIVE_H = 0.05;

const approach = (value: number, goal: number, dt: number, reduced: boolean) => (reduced ? goal : value + (goal - value) * Math.min(1, dt * 2.2));

type Pen = (text: string, x: number, y: number, size: number, color: string, serif?: boolean) => void;
const pen =
  (ctx: CanvasRenderingContext2D, fonts: Fonts, w: number, h: number): Pen =>
  (text, x, y, size, color, serif = false) => {
    ctx.fillStyle = color;
    ctx.font = `${h * size}px ${serif ? fonts.serif : fonts.sans}`;
    ctx.fillText(text, w * x, h * y);
  };

export function createFlashPrefillAboutModel(kit: Kit, fonts: Fonts, numbers: FlashAboutNumbers): BookModel {
  const root = new THREE.Group();
  const wash = washPlane(kit.palette.blue, 0.6, 0.6, 0.12);
  wash.position.set(X0 + (N * S) / 2, 0.002, Z0 + (N * S) / 2);
  root.add(wash);

  const title = createCard(kit, 0.86, 0.13, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("prefill attention: every query row against every key before it", 0.03, 0.29, 0.22, kit.palette.ink, true);
    write(`FlashAttention: one tile of ${numbers.tile} × ${numbers.tile} scores at a time, never the whole table`, 0.03, 0.59, 0.19, kit.palette.ink, true);
    write("each query row keeps a running max m, a running sum l and its weighted values acc", 0.03, 0.87, 0.16, kit.palette.ink);
  });
  title.group.position.set(0.5, 0, -0.57);
  root.add(title.group);

  const cx = (c: number) => X0 + (c + 0.5) * S;
  const cz = (r: number) => Z0 + (r + 0.5) * S;
  const slab = new THREE.BoxGeometry(S - 0.006, 0.004, S - 0.006);
  const unit = new THREE.BoxGeometry(S - 0.008, 1, S - 0.008);
  const tiles: { r: number; c: number; fill: THREE.Mesh; height: number }[] = [];
  for (let r = 0; r < N; r++) {
    for (let c = 0; c <= r; c++) {
      const base = kit.solid(slab, { shadow: false });
      base.position.set(cx(c), 0.002, cz(r));
      const fill = new THREE.Mesh(unit, kit.blue);
      fill.castShadow = true;
      fill.position.set(cx(c), 0, cz(r));
      fill.visible = false;
      root.add(base, fill);
      tiles.push({ r, c, fill, height: 0 });
    }
  }

  const keyBlock = new THREE.Mesh(new THREE.BoxGeometry(S - 0.008, 0.026, 0.026), kit.pink);
  keyBlock.castShadow = true;
  keyBlock.position.set(cx(0), 0.013, Z0 - 0.03);
  const queryBlock = kit.solid(new THREE.BoxGeometry(0.026, 0.026, S - 0.008));
  queryBlock.position.set(X0 - 0.03, 0.013, cz(0));
  const rails: number[] = [X0, 0.003, Z0 - 0.03, X0 + N * S, 0.003, Z0 - 0.03, X0 - 0.03, 0.003, Z0, X0 - 0.03, 0.003, Z0 + N * S];
  root.add(keyBlock, queryBlock, inkSegments(rails, kit.thin));

  const side = (z: number, height: number, lines: [string, string][]) => {
    const card = createCard(kit, 0.34, height, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      lines.forEach(([text, color], i) => write(text, 0.05, (i + 0.74) / lines.length, (i === 0 ? 0.8 : 0.66) / lines.length, color, i === 0));
    });
    card.group.position.set(0.74, 0, z);
    root.add(card.group);
  };
  side(-0.37, 0.1, [
    ["pink block: a tile of keys", kit.palette.pink],
    [`${numbers.tile} keys and values, loaded once`, kit.palette.ink],
    [`for all ${numbers.tile} rows of the query tile`, kit.palette.ink],
  ]);
  side(-0.22, 0.1, [
    ["tall blue tile: the scores", kit.palette.blue],
    [`${numbers.tileBytes.toLocaleString("en-US")} bytes, on the chip`, kit.palette.ink],
    ["used, then dropped", kit.palette.ink],
  ]);
  side(-0.07, 0.1, [
    ["the whole table instead", kit.palette.ink],
    [`${numbers.tableMb} MB per head at ${numbers.positions.toLocaleString("en-US")} tokens`, kit.palette.ink],
    ["(calculated), kept in DRAM", kit.palette.faint],
  ]);

  let row = 5;
  let col = row;
  const readout = createCard(kit, 0.86, 0.13, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write(`query tile ${row + 1} of ${N}: rows ${row * numbers.tile} to ${(row + 1) * numbers.tile - 1}`, 0.03, 0.28, 0.24, kit.palette.ink, true);
    write(`key tile ${col + 1} of ${row + 1}: keys ${col * numbers.tile} to ${(col + 1) * numbers.tile - 1}, scored against all ${numbers.tile} rows`, 0.03, 0.56, 0.18, kit.palette.blue);
    write(col === row ? "the last tile holds the rows' own keys: a key after its query is masked" : "its scores update each row's m, l and acc, and are then dropped", 0.03, 0.84, 0.17, col === row ? kit.palette.pink : kit.palette.ink);
  });
  readout.group.position.set(0.5, 0, 0.08);
  root.add(readout.group);

  const railZ = 0.25;
  const railX = [0.2, 0.8];
  const rail = kit.solid(new THREE.BoxGeometry(railX[1] - railX[0] + 0.04, 0.016, 0.02));
  rail.position.set(0.5, 0.008, railZ);
  const tickX = (k: number) => railX[0] + (k / (N - 1)) * (railX[1] - railX[0]);
  const ticks: number[] = [];
  for (let k = 0; k < N; k++) ticks.push(tickX(k), 0.017, railZ - 0.018, tickX(k), 0.017, railZ + 0.018);
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.028, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(tickX(row), 0.045, railZ);
  const railCard = createCard(kit, 0.6, 0.045, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("which query tile", 0.02, 0.7, 0.5, kit.palette.faint);
    write(`1 to ${N}`, 0.82, 0.7, 0.5, kit.palette.pink);
  });
  railCard.group.position.set(0.5, 0, railZ + 0.06);
  root.add(rail, inkSegments(ticks, kit.thin), knob, railCard.group);

  const show = (r: number, c: number) => {
    if (r === row && c === col) return;
    row = r;
    col = c;
    readout.redraw();
  };

  let held = false;
  let chosen = false;
  let knobX = tickX(row);
  let clock = 0;
  let sweep = 0;
  return {
    exhibits: [root],
    handles: [knob],
    dragStart: () => {
      held = true;
      chosen = true;
    },
    drag: (px) => {
      knobX = Math.min(railX[1], Math.max(railX[0], px));
      const r = Math.round(((knobX - railX[0]) / (railX[1] - railX[0])) * (N - 1));
      if (r !== row) {
        sweep = 0;
        show(r, 0);
      }
    },
    dragEnd: () => {
      held = false;
    },
    enter: () => {
      clock = 0;
      sweep = 0;
      chosen = false;
    },
    update: (dt, _exhibit, _stop, reduced) => {
      if (reduced) {
        if (!chosen) show(5, 5);
        else show(row, row);
      } else if (chosen) {
        sweep = (sweep + dt) % ((row + 1) * STEP + REST);
        show(row, Math.min(row, Math.floor(sweep / STEP)));
      } else {
        const rows = [3, 5, 7];
        const spans = rows.map((r) => (r + 1) * STEP + REST);
        clock = (clock + dt) % spans.reduce((a, b) => a + b, 0);
        let t = clock;
        let k = 0;
        while (t >= spans[k]) t -= spans[k++];
        show(rows[k], Math.min(rows[k], Math.floor(t / STEP)));
      }
      knob.position.x = held ? knobX : approach(knob.position.x, tickX(row), dt * 0.7, reduced);
      keyBlock.position.x = approach(keyBlock.position.x, cx(col), dt * 1.4, reduced);
      queryBlock.position.z = approach(queryBlock.position.z, cz(row), dt * 1.4, reduced);
      for (const t of tiles) {
        const goal = t.r !== row ? 0 : t.c < col ? DONE_H : t.c === col ? LIVE_H : 0;
        t.height = approach(t.height, goal, dt * 1.6, reduced);
        t.fill.visible = t.height > 0.002;
        t.fill.scale.y = Math.max(0.0001, t.height);
        t.fill.position.y = t.height / 2 + 0.004;
      }
    },
  };
}
