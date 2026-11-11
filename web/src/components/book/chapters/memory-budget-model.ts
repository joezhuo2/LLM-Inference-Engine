import * as THREE from "three";
import type { BookModel } from "../book-scene";
import { inkSegments, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";
import { block, joinParts, type Part, readoutCard, tagCard, titleCard, W, X0 } from "./segments";
import { approach, lay, liquid, makeRail, picker } from "./where-the-time-went-model";

export type BudgetRow = { batch: number; context: number; lines: [string, string, string] };
export type FitRow = { label: string; weights: number; pool: number; spare: number; over: boolean; contexts: number; lines: [string, string, string] };
export type MemoryNumbers = { fullContext: number; poolTokens: number; rows: BudgetRow[]; vram: number; vramText: string; rest: number; fits: FitRow[]; slots: number };

function wallExhibit(kit: Kit, fonts: Fonts, n: MemoryNumbers): Part {
  const root = new THREE.Group();
  const Z = -0.2;
  const TOP = 0.2;
  const D = 0.05;
  const most = n.rows[n.rows.length - 1].batch;
  const x = (batch: number) => X0 + (batch / most) * W;
  const y = (context: number) => (context / n.fullContext) * TOP;
  const wash = washPlane(kit.palette.blue, 0.9, 0.4, 0.1);
  wash.position.set(0.5, 0.002, Z);
  root.add(wash);
  titleCard(kit, fonts, root, "one pool: how many sequences, and how long each can grow", "width is sequences, height is tokens each can reach; the ink line is the budget (calculated)");

  const wall = kit.solid(new THREE.BoxGeometry(W + 0.02, TOP + 0.04, 0.004), { material: kit.glass, shadow: false });
  wall.position.set(0.5, (TOP + 0.04) / 2, Z - D / 2 - 0.006);
  root.add(wall);
  const curve: number[] = [];
  const zc = Z - D / 2 - 0.003;
  const at = (batch: number) => Math.min(n.fullContext, n.poolTokens / batch);
  for (let i = 0; i < 72; i++) {
    const a = 1 + ((most - 1) * i) / 72;
    const b = 1 + ((most - 1) * (i + 1)) / 72;
    curve.push(x(a), y(at(a)) + 0.003, zc, x(b), y(at(b)) + 0.003, zc);
  }
  root.add(inkSegments(curve, kit.line, 0.0004));

  const slab = liquid(kit.blue);
  const one = liquid(kit.pink);
  root.add(slab, one);
  let chosen = 0;
  let width = x(n.rows[0].batch) - X0;
  let height = y(n.rows[0].context);
  const readout = readoutCard(kit, fonts, root, 0.1, () => n.rows[chosen].lines);
  const rail = makeRail(kit, fonts, root, 0.27, n.rows.map((r) => x(r.batch)), "sequences sharing the pool", `${n.rows[0].batch} to ${most}`);
  const pick = picker(rail, 0, 1, 2.8, (i) => {
    chosen = i;
    readout.redraw();
  });
  return {
    root,
    pick,
    update: (dt, reduced) => {
      pick.update(dt, reduced);
      width = approach(width, x(n.rows[chosen].batch) - X0, dt * 1.2, reduced);
      height = approach(height, y(n.rows[chosen].context), dt * 1.2, reduced);
      lay(slab, X0, width, height, Z, D);
      lay(one, X0, Math.max(0.005, W / most), height, Z + D / 2 + 0.008, 0.012);
    },
  };
}

function fitExhibit(kit: Kit, fonts: Fonts, n: MemoryNumbers): Part {
  const root = new THREE.Group();
  const Z1 = -0.3;
  const Z2 = -0.12;
  const D = 0.06;
  const H = 0.05;
  const scale = W / n.vram;
  const wash = washPlane(kit.palette.blue, 0.9, 0.5, 0.1);
  wash.position.set(0.5, 0.002, -0.2);
  root.add(wash);
  titleCard(kit, fonts, root, "four models in the same memory", "the weights grow and the pool shrinks; TinyLlama measured, the rest calculated");
  tagCard(kit, fonts, root, Z1, "everything else, the weights, the KV pool, kept free", n.vramText, kit.palette.ink);
  tagCard(kit, fonts, root, Z2, `contexts of ${n.fullContext.toLocaleString("en-US")} tokens that fit in the pool`, `one slab each, up to ${n.slots}`, kit.palette.blue);

  const outline = block(kit, "glass");
  lay(outline, X0, W, H * 1.3, Z1, D * 1.15);
  const bars = [block(kit, "ink"), block(kit, "ink"), block(kit, "blue"), block(kit, "glass")];
  const over = liquid(kit.pink);
  root.add(outline, ...bars, over);
  const slabs = Array.from({ length: n.slots }, () => liquid(kit.blue));
  root.add(...slabs);
  const pitch = W / n.slots;

  let chosen = 0;
  const goal = (f: FitRow) => [n.rest * scale, Math.min(f.weights, n.vram - n.rest) * scale, f.pool * scale, f.spare * scale];
  const now = goal(n.fits[0]);
  let shown = n.fits[0].contexts;
  let spill = 0;
  const readout = readoutCard(kit, fonts, root, 0.1, () => n.fits[chosen].lines);
  const rail = makeRail(kit, fonts, root, 0.27, n.fits.map((_, i) => 0.2 + (0.6 * i) / Math.max(1, n.fits.length - 1)), "which model", `${n.fits[0].label} …`);
  const pick = picker(rail, 0, 0, 3.4, (i) => {
    chosen = i;
    readout.redraw();
  });
  return {
    root,
    pick,
    update: (dt, reduced) => {
      pick.update(dt, reduced);
      const f = n.fits[chosen];
      const to = goal(f);
      let at = X0;
      bars.forEach((bar, i) => {
        now[i] = approach(now[i], to[i], dt * 1.2, reduced);
        lay(bar, at, Math.max(0, now[i] - 0.002), i === 0 ? H * 0.5 : H, Z1, D);
        at += now[i];
      });
      spill = approach(spill, f.over ? 0.05 : 0, dt * 1.2, reduced);
      lay(over, X0 + W + 0.006, spill, H * 1.6, Z1, D);
      shown = approach(shown, Math.min(n.slots, f.contexts), dt * 1.2, reduced);
      slabs.forEach((slab, i) => {
        lay(slab, X0 + i * pitch, pitch * 0.72, H * 0.8, Z2, D);
        slab.visible = i < Math.round(shown);
      });
    },
  };
}

export function createMemoryModel(kit: Kit, fonts: Fonts, n: MemoryNumbers): BookModel {
  return joinParts([wallExhibit(kit, fonts, n), fitExhibit(kit, fonts, n)]);
}
