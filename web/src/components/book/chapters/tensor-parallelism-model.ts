import * as THREE from "three";
import type { BookModel } from "../book-scene";
import { createCard, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";
import { block, joinParts, type Part, readoutCard, titleCard, type Tone, W, X0 } from "./segments";
import { approach, lay, liquid, makeRail, pen, picker } from "./where-the-time-went-model";

type Lines = [string, string, string];
export type TpPiece = { value: number; tone: Tone };
export type TpTray = { n: number; parts: TpPiece[]; lane: string; lines: Lines };
export type TpMatrixCut = { label: string; sub: string; cut: "rows" | "cols"; bands: { label: string; share: number }[] };
export type TpCut = { title: string; sub: string; ranks: number; matrices: TpMatrixCut[]; lines: Lines[] };
export type TpStage = { label: string; shared: boolean; lines: Lines };
export type TpBytes = { n: number; parts: TpPiece[]; label: string; text: string; lines: Lines };
export type TpNumbers = { full: number; trays: TpTray[]; attention: TpCut; mlp: TpCut; stageRanks: number; stages: TpStage[]; whole: { parts: TpPiece[]; text: string }; bytes: TpBytes[] };

const H = 0.035;
const D = 0.05;
const GAP = 0.003;

function wash(root: THREE.Group, kit: Kit) {
  const plane = washPlane(kit.palette.blue, 0.9, 0.5, 0.1);
  plane.position.set(0.5, 0.002, -0.25);
  root.add(plane);
}

function laneCard(kit: Kit, fonts: Fonts, root: THREE.Group, x: number, width: number, z: number, text: () => [string, string]) {
  const card = createCard(kit, width, 0.032, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    const [left, right] = text();
    write(left, 0.02, 0.72, 0.58, kit.palette.ink);
    ctx.textAlign = "right";
    write(right, 0.98, 0.72, 0.58, kit.palette.blue);
    ctx.textAlign = "left";
  });
  card.group.position.set(x, 0, z);
  root.add(card.group);
  return card;
}

const railXs = (count: number) => Array.from({ length: count }, (_, i) => 0.2 + (0.6 * i) / Math.max(1, count - 1));

function lit(kit: Kit, tone: Tone) {
  return { plain: block(kit, tone), pink: liquid(kit.pink) };
}

export function traysExhibit(kit: Kit, fonts: Fonts, n: TpNumbers): Part {
  const root = new THREE.Group();
  wash(root, kit);
  titleCard(kit, fonts, root, "one model, cut across GPUs", "each row is one GPU; the blocks are to scale by their bytes (calculated)");
  const scale = W / n.full;
  const most = Math.max(...n.trays.map((t) => t.n));
  const laneZ = (k: number) => -0.46 + k * 0.12;
  let chosen = 0;
  const lanes = Array.from({ length: most }, (_, k) => {
    const tray = block(kit, "glass");
    const blocks = n.trays[0].parts.map((p) => block(kit, p.tone));
    root.add(tray, ...blocks);
    const card = laneCard(kit, fonts, root, 0.5, 0.84, laneZ(k) + 0.042, () => (k < n.trays[chosen].n ? [`GPU ${k}`, n.trays[chosen].lane] : ["", ""]));
    return { tray, blocks, card, p: k === 0 ? 1 : 0, widths: n.trays[0].parts.map((p) => p.value * scale) };
  });
  const readout = readoutCard(kit, fonts, root, 0.1, () => n.trays[chosen].lines);
  const rail = makeRail(kit, fonts, root, 0.27, railXs(n.trays.length), "how many GPUs", n.trays.map((t) => t.n).join(", "));
  const pick = picker(rail, 1, 1, 3.2, (i) => {
    chosen = i;
    readout.redraw();
    lanes.forEach((l) => l.card.redraw());
  });
  return {
    root,
    pick,
    update: (dt, reduced) => {
      pick.update(dt, reduced);
      const t = n.trays[chosen];
      lanes.forEach((l, k) => {
        l.p = approach(l.p, k < t.n ? 1 : 0, dt * 1.2, reduced);
        l.card.group.visible = l.p > 0.5;
        lay(l.tray, X0, W, H * 1.1 * l.p, laneZ(k), D * 1.1);
        let at = X0;
        l.blocks.forEach((b, i) => {
          l.widths[i] = approach(l.widths[i], t.parts[i].value * scale, dt * 1.2, reduced);
          lay(b, at, Math.max(0, l.widths[i] - GAP), H * l.p, laneZ(k), D);
          b.visible = b.visible && l.p > 0.03;
          at += l.widths[i];
        });
        l.tray.visible = l.p > 0.03;
      });
    },
  };
}

export function cutExhibit(kit: Kit, fonts: Fonts, c: TpCut): Part {
  const root = new THREE.Group();
  wash(root, kit);
  titleCard(kit, fonts, root, c.title, c.sub);
  const Z0 = -0.4;
  const DEPTH = 0.3;
  const MW = 0.3;
  const pieces: { r: number; x: number; w: number; z: number; d: number; plain: THREE.Object3D; pink: THREE.Object3D; h: number }[] = [];
  c.matrices.forEach((m, j) => {
    const x0 = j === 0 ? 0.14 : 0.58;
    const card = createCard(kit, 0.38, 0.064, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      write(m.label, 0.03, 0.44, 0.38, kit.palette.ink, true);
      write(m.sub, 0.03, 0.86, 0.3, kit.palette.ink);
    });
    card.group.position.set(x0 + 0.16, 0, -0.47);
    root.add(card.group);
    let z = Z0;
    m.bands.forEach((band) => {
      const bd = band.share * DEPTH;
      if (m.bands.length > 1) {
        const tag = createCard(kit, 0.05, 0.026, (ctx, w, h) => {
          const write = pen(ctx, fonts, w, h);
          ctx.textAlign = "center";
          write(band.label, 0.5, 0.75, 0.62, kit.palette.ink);
          ctx.textAlign = "left";
        });
        tag.group.position.set(x0 - 0.035, 0, z + Math.max(bd, 0.026) / 2);
        root.add(tag.group);
      }
      for (let r = 0; r < c.ranks; r++) {
        const p = lit(kit, "ink");
        root.add(p.plain, p.pink);
        if (m.cut === "rows") pieces.push({ r, x: x0, w: MW, z: z + (r * bd) / c.ranks, d: bd / c.ranks, ...p, h: H * 0.6 });
        else pieces.push({ r, x: x0 + (r * MW) / c.ranks, w: MW / c.ranks, z, d: bd, ...p, h: H * 0.6 });
      }
      z += bd;
    });
  });
  let chosen = 0;
  const readout = readoutCard(kit, fonts, root, 0.1, () => c.lines[chosen]);
  const rail = makeRail(kit, fonts, root, 0.27, railXs(c.ranks), "which rank", `rank 0 to ${c.ranks - 1} of ${c.ranks}`);
  const pick = picker(rail, 0, 0, 3, (i) => {
    chosen = i;
    readout.redraw();
  });
  return {
    root,
    pick,
    update: (dt, reduced) => {
      pick.update(dt, reduced);
      for (const p of pieces) {
        p.h = approach(p.h, p.r === chosen ? H * 1.8 : H * 0.6, dt * 1.4, reduced);
        const depth = Math.max(0.002, p.d - GAP);
        for (const o of [p.plain, p.pink]) lay(o, p.x, Math.max(0.002, p.w - GAP), p.h, p.z + p.d / 2, depth);
        p.plain.visible = p.r !== chosen;
        p.pink.visible = p.r === chosen;
      }
    },
  };
}

export function stagesExhibit(kit: Kit, fonts: Fonts, n: TpNumbers): Part {
  const root = new THREE.Group();
  wash(root, kit);
  titleCard(kit, fonts, root, "one layer, step by step", `${n.stageRanks} ranks; blue blocks run in each rank, ink ones once on the whole vector`);
  const left = 0.24;
  const pitch = (0.92 - left) / n.stages.length;
  const laneZ = (r: number) => -0.4 + r * 0.16;
  for (let r = 0; r < n.stageRanks; r++) laneCard(kit, fonts, root, 0.15, 0.13, laneZ(r) + 0.05, () => [`rank ${r}`, ""]);
  const blocks = n.stages.flatMap((s, i) =>
    s.shared
      ? [{ i, ...lit(kit, "ink"), z: (laneZ(0) + laneZ(n.stageRanks - 1)) / 2, d: laneZ(n.stageRanks - 1) - laneZ(0) + 0.08, h: H }]
      : Array.from({ length: n.stageRanks }, (_, r) => ({ i, ...lit(kit, "blue"), z: laneZ(r), d: 0.08, h: H })),
  );
  for (const b of blocks) root.add(b.plain, b.pink);
  let chosen = 0;
  const readout = readoutCard(kit, fonts, root, 0.1, () => n.stages[chosen].lines);
  const rail = makeRail(kit, fonts, root, 0.27, n.stages.map((_, i) => left + pitch * (i + 0.5)), "one layer, in order", `${n.stages[0].label} …`);
  const pick = picker(rail, 0, 0, 2.6, (i) => {
    chosen = i;
    readout.redraw();
  });
  return {
    root,
    pick,
    update: (dt, reduced) => {
      pick.update(dt, reduced);
      for (const b of blocks) {
        b.h = approach(b.h, b.i === chosen ? H * 2.6 : H, dt * 1.4, reduced);
        for (const o of [b.plain, b.pink]) lay(o, left + pitch * b.i + 0.006, pitch - 0.012, b.h, b.z, b.d);
        b.plain.visible = b.i !== chosen;
        b.pink.visible = b.i === chosen;
      }
    },
  };
}

export function bytesExhibit(kit: Kit, fonts: Fonts, n: TpNumbers): Part {
  const root = new THREE.Group();
  wash(root, kit);
  titleCard(kit, fonts, root, "what one rank holds", "MiB on the device as the engine loaded it (measured); the parts calculated from the shapes");
  const scale = W / n.full;
  const Z1 = -0.4;
  const Z2 = -0.2;
  laneCard(kit, fonts, root, 0.5, 0.84, Z1 + 0.042, () => ["the whole model", n.whole.text]);
  let at = X0;
  for (const p of n.whole.parts) {
    const b = block(kit, p.tone);
    lay(b, at, Math.max(0.001, p.value * scale - GAP), H, Z1, D);
    b.visible = true;
    root.add(b);
    at += p.value * scale;
  }
  let chosen = 0;
  const card = laneCard(kit, fonts, root, 0.5, 0.84, Z2 + 0.042, () => [n.bytes[chosen].label, n.bytes[chosen].text]);
  const bars = n.bytes[0].parts.map((p) => block(kit, p.tone));
  const widths = n.bytes[0].parts.map((p) => p.value * scale);
  const mark = liquid(kit.pink);
  root.add(...bars, mark);
  let markX = X0 + W;
  const readout = readoutCard(kit, fonts, root, 0.1, () => n.bytes[chosen].lines);
  const rail = makeRail(kit, fonts, root, 0.27, railXs(n.bytes.length), "one rank of", n.bytes.map((b) => b.n).join(", "));
  const pick = picker(rail, 1, 1, 3.2, (i) => {
    chosen = i;
    readout.redraw();
    card.redraw();
  });
  return {
    root,
    pick,
    update: (dt, reduced) => {
      pick.update(dt, reduced);
      const goal = n.bytes[chosen];
      let x = X0;
      bars.forEach((b, i) => {
        widths[i] = approach(widths[i], goal.parts[i].value * scale, dt * 1.2, reduced);
        lay(b, x, Math.max(0, widths[i] - GAP), H, Z2, D);
        x += widths[i];
      });
      markX = approach(markX, X0 + W / goal.n, dt * 1.2, reduced);
      lay(mark, markX - 0.002, 0.004, H * 2.2, Z2 - 0.01, D * 1.6);
      mark.visible = goal.n > 1;
    },
  };
}

export function createTensorModel(kit: Kit, fonts: Fonts, n: TpNumbers): BookModel {
  return joinParts([cutExhibit(kit, fonts, n.attention), cutExhibit(kit, fonts, n.mlp), stagesExhibit(kit, fonts, n), bytesExhibit(kit, fonts, n)]);
}
