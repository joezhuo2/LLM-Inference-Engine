import * as THREE from "three";
import type { BookModel } from "../book-scene";
import { createCard, type Kit, ribbonTube, washPlane } from "../ink";
import type { Fonts } from "../typeset";
import { block, joinParts, type Part, readoutCard, tagCard, titleCard, W, X0 } from "./segments";
import { approach, lay, liquid, makeRail, pen, picker } from "./where-the-time-went-model";

export type SpecStop = {
  index: number;
  rows: number;
  kept: number;
  ngram: number;
  back: number;
  found: string[];
  lead: number;
  tail: string[];
  draft: string[];
  from: number;
  width: number;
  lines: { steps: [string, string, string]; guess: [string, string, string]; check: [string, string, string] };
};
export type SpecBar = { k: number; tokensPerStep: number; ms: number; chatMs: number | null; lines: [string, string, string] };
export type SpecNumbers = { tokens: number; maxBatch: number; rest: number; blocks: { from: number; width: number; kept: number }[]; stops: SpecStop[]; bars: SpecBar[] };

const CELLS = 13;
const stopXs = (n: SpecNumbers) => n.stops.map((s) => X0 + ((s.from + s.width / 2) / n.tokens) * W);
const evenXs = (count: number) => Array.from({ length: count }, (_, i) => 0.14 + (0.72 * i) / Math.max(1, count - 1));

export function stepsExhibit(kit: Kit, fonts: Fonts, n: SpecNumbers): Part {
  const root = new THREE.Group();
  const Z1 = -0.36;
  const Z2 = -0.15;
  const D = 0.06;
  const H = 0.022;
  const px = W / n.tokens;
  const wash = washPlane(kit.palette.blue, 0.9, 0.46, 0.1);
  wash.position.set(0.5, 0.002, -0.24);
  root.add(wash);
  titleCard(kit, fonts, root, "the same answer in fewer steps", "a block is one step; its width is the tokens it wrote, its height the guesses kept (measured)");
  tagCard(kit, fonts, root, Z1, "plain decoding: one token per step", `${n.tokens} steps`, kit.palette.ink);
  tagCard(kit, fonts, root, Z2, "with guesses checked in the same step", `${n.blocks.length} steps`, kit.palette.blue);

  for (let i = 0; i < n.tokens; i++) {
    const tick = liquid(kit.ink);
    lay(tick, X0 + i * px, Math.max(0.0008, px * 0.7), H, Z1, D);
    tick.visible = true;
    root.add(tick);
  }
  const blocks = n.blocks.map((b) => {
    const mesh = liquid(b.kept > 0 ? kit.blue : kit.ink);
    root.add(mesh);
    return { mesh, height: H };
  });
  const lit = liquid(kit.pink);
  root.add(lit);

  let chosen = 0;
  let litX = X0;
  let litW = px;
  const readout = readoutCard(kit, fonts, root, 0.07, () => n.stops[chosen].lines.steps);
  const rail = makeRail(kit, fonts, root, 0.25, stopXs(n), "which step of the answer", `step ${n.stops[0].index} to ${n.stops[n.stops.length - 1].index}`);
  const pick = picker(rail, 0, n.rest, 2.8, (i) => {
    chosen = i;
    readout.redraw();
  });
  return {
    root,
    pick,
    update: (dt, reduced) => {
      pick.update(dt, reduced);
      blocks.forEach((b, i) => {
        const goal = H * (1 + n.blocks[i].kept * 0.45);
        b.height = approach(b.height, goal, dt * 1.2, reduced);
        lay(b.mesh, X0 + n.blocks[i].from * px, Math.max(0.0008, n.blocks[i].width * px - px * 0.3), b.height, Z2, D);
        b.mesh.visible = true;
      });
      const s = n.stops[chosen];
      litX = approach(litX, X0 + s.from * px, dt * 1.4, reduced);
      litW = approach(litW, Math.max(0.004, s.width * px - px * 0.3), dt * 1.4, reduced);
      lay(lit, litX, litW, H * (1.4 + s.kept * 0.45), Z2, D * 1.12);
    },
  };
}

function chipCard(kit: Kit, fonts: Fonts, root: THREE.Group, z: number, cells: () => { text: string; fill: string | null; ink: string }[]) {
  const card = createCard(kit, W + 0.02, 0.06, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    const cw = w / (CELLS + 0.3);
    ctx.textAlign = "center";
    cells().forEach((c, i) => {
      if (c.fill) {
        ctx.fillStyle = c.fill;
        ctx.fillRect(cw * (0.15 + i) + cw * 0.04, h * 0.14, cw * 0.92, h * 0.72);
      }
      write(c.text, (0.15 + i + 0.5) / (CELLS + 0.3), 0.63, 0.3, c.ink);
    });
    ctx.textAlign = "left";
  });
  card.group.position.set(0.5, 0, z);
  root.add(card.group);
  return card;
}

const clip = (text: string, most = 6) => (text.length > most ? `${text.slice(0, most - 1)}…` : text);

function guessExhibit(kit: Kit, fonts: Fonts, n: SpecNumbers): Part {
  const root = new THREE.Group();
  const Z1 = -0.39;
  const Z2 = -0.17;
  const H = 0.02;
  const D = 0.04;
  const wash = washPlane(kit.palette.blue, 0.9, 0.5, 0.1);
  wash.position.set(0.5, 0.002, -0.22);
  root.add(wash);
  titleCard(kit, fonts, root, "the guess is copied from earlier in the context", "blue: the last tokens and where they appeared before; pink: what followed, copied to the end");
  tagCard(kit, fonts, root, Z1 + 0.045, "earlier in the context", "the latest match", kit.palette.ink);
  tagCard(kit, fonts, root, Z2 + 0.045, "the end of the context, and the rows of the next step", "the guess", kit.palette.pink);

  let chosen = 0;
  const stop = () => n.stops[chosen];
  const tone = (kind: "plain" | "pattern" | "draft") => (kind === "pattern" ? { fill: kit.palette.blue, ink: kit.palette.paper } : kind === "draft" ? { fill: kit.palette.pink, ink: kit.palette.paper } : { fill: null, ink: kit.palette.ink });
  const top = chipCard(kit, fonts, root, Z1 + 0.036, () => stop().found.slice(0, CELLS).map((t, i) => ({ text: clip(t), ...tone(i < stop().lead ? "plain" : i < stop().lead + stop().ngram ? "pattern" : "draft") })));
  const end = chipCard(kit, fonts, root, Z2 + 0.036, () => [...stop().tail, ...stop().draft].slice(0, CELLS).map((t, i) => ({ text: clip(t), ...tone(i < stop().tail.length - stop().ngram ? "plain" : i < stop().tail.length ? "pattern" : "draft") })));

  const cellX = (i: number) => 0.5 - (W + 0.02) / 2 + ((W + 0.02) / (CELLS + 0.3)) * (0.15 + i);
  const cw = (W + 0.02) / (CELLS + 0.3);
  const tiles = (z: number) =>
    Array.from({ length: CELLS }, () => {
      const mesh = liquid(kit.ink);
      root.add(mesh);
      return { mesh, height: 0.004, z };
    });
  const topTiles = tiles(Z1);
  const endTiles = tiles(Z2);
  let arc: THREE.Mesh | null = null;
  const bridge = () => {
    if (arc) {
      root.remove(arc);
      arc.geometry.dispose();
    }
    const s = stop();
    if (s.draft.length === 0) {
      arc = null;
      return;
    }
    const a = cellX(s.lead + s.ngram + Math.min(s.draft.length, CELLS - s.lead - s.ngram) / 2);
    const b = cellX(s.tail.length + Math.min(s.draft.length, CELLS - s.tail.length) / 2);
    arc = ribbonTube([new THREE.Vector3(a, 0.05, Z1), new THREE.Vector3((a + b) / 2, 0.13, (Z1 + Z2) / 2), new THREE.Vector3(b, 0.05, Z2)], 0.004, kit.pink);
    root.add(arc);
  };

  const readout = readoutCard(kit, fonts, root, 0.1, () => stop().lines.guess);
  const rail = makeRail(kit, fonts, root, 0.27, evenXs(n.stops.length), "which step of the answer", `step ${n.stops[0].index} to ${n.stops[n.stops.length - 1].index}`);
  const pick = picker(rail, 0, n.rest, 3.4, (i) => {
    chosen = i;
    top.redraw();
    end.redraw();
    readout.redraw();
    bridge();
  });
  const place = (row: typeof topTiles, kinds: ("plain" | "pattern" | "draft" | "none")[], dt: number, reduced: boolean) => {
    row.forEach((t, i) => {
      const kind = kinds[i] ?? "none";
      t.mesh.material = kind === "pattern" ? kit.blue : kind === "draft" ? kit.pink : kit.ink;
      t.height = approach(t.height, kind === "none" ? 0 : kind === "plain" ? H * 0.5 : kind === "pattern" ? H * 1.6 : H * 2.4, dt * 1.4, reduced);
      lay(t.mesh, cellX(i) + cw * 0.08, cw * 0.84, Math.max(0.0005, t.height), t.z, D);
      t.mesh.visible = kind !== "none" && t.height > 0.001;
    });
  };
  return {
    root,
    pick,
    update: (dt, reduced) => {
      pick.update(dt, reduced);
      const s = stop();
      place(
        topTiles,
        s.found.map((_, i) => (i < s.lead ? "plain" : i < s.lead + s.ngram ? "pattern" : "draft")),
        dt,
        reduced,
      );
      place(
        endTiles,
        [...s.tail, ...s.draft].map((_, i) => (i < s.tail.length - s.ngram ? "plain" : i < s.tail.length ? "pattern" : "draft")),
        dt,
        reduced,
      );
    },
  };
}

function checkExhibit(kit: Kit, fonts: Fonts, n: SpecNumbers): Part {
  const root = new THREE.Group();
  const Z0 = -0.44;
  const DEPTH = 0.42;
  const pitch = DEPTH / n.maxBatch;
  const LONG = W * 0.6;
  const OUT = X0 + W * 0.68;
  const H = 0.03;
  const wash = washPlane(kit.palette.blue, 0.9, 0.5, 0.1);
  wash.position.set(0.5, 0.002, -0.23);
  root.add(wash);
  titleCard(kit, fonts, root, "one step checks every guess", `the ${n.maxBatch} rows of a step, front to back; a bar is a row through the model, a tile the token it kept`);
  const rows = Array.from({ length: n.maxBatch }, (_, i) => {
    const real = block(kit, "ink");
    const kept = liquid(kit.blue);
    const dropped = liquid(kit.pink);
    const spare = block(kit, "glass");
    const token = liquid(i === 0 ? kit.ink : kit.blue);
    root.add(real, kept, dropped, spare, token);
    return { real, kept, dropped, spare, token, height: 0.004, out: 0 };
  });

  let chosen = 0;
  const readout = readoutCard(kit, fonts, root, 0.1, () => n.stops[chosen].lines.check);
  const rail = makeRail(kit, fonts, root, 0.27, evenXs(n.stops.length), "which step of the answer", `step ${n.stops[0].index} to ${n.stops[n.stops.length - 1].index}`);
  const pick = picker(rail, 0, n.rest, 3.4, (i) => {
    chosen = i;
    readout.redraw();
  });
  return {
    root,
    pick,
    update: (dt, reduced) => {
      pick.update(dt, reduced);
      const s = n.stops[chosen];
      rows.forEach((r, i) => {
        const z = Z0 + (n.maxBatch - 1 - i) * pitch;
        const kind = i === 0 ? "real" : i <= s.kept ? "kept" : i < s.rows ? "dropped" : "spare";
        r.height = approach(r.height, kind === "spare" ? 0.004 : kind === "dropped" ? H * 0.35 : H, dt * 1.4, reduced);
        r.out = approach(r.out, i <= s.kept ? W * 0.3 : 0, dt * 1.2, reduced);
        for (const [name, mesh] of [["real", r.real], ["kept", r.kept], ["dropped", r.dropped], ["spare", r.spare]] as const) {
          lay(mesh, X0, LONG, r.height, z, pitch * 0.7);
          mesh.visible = name === kind;
        }
        lay(r.token, OUT, r.out, H * 1.2, z, pitch * 0.7);
      });
    },
  };
}

function payoffExhibit(kit: Kit, fonts: Fonts, n: SpecNumbers): Part {
  const root = new THREE.Group();
  const Z1 = -0.34;
  const Z2 = -0.12;
  const D = 0.05;
  const xs = evenXs(n.bars.length);
  const bw = Math.min(0.05, (0.72 / n.bars.length) * 0.36);
  const mostTokens = Math.max(...n.bars.map((b) => b.tokensPerStep));
  const mostMs = Math.max(...n.bars.map((b) => Math.max(b.ms, b.chatMs ?? 0)));
  const wash = washPlane(kit.palette.blue, 0.9, 0.5, 0.1);
  wash.position.set(0.5, 0.002, -0.22);
  root.add(wash);
  titleCard(kit, fonts, root, "what the guesses buy, by how many a step may carry", "one request at a time; answers that repeat their prompt against chat answers (measured)");
  tagCard(kit, fonts, root, Z1, "tokens written per step, answers that repeat their prompt", `up to ${mostTokens.toFixed(2)}`, kit.palette.blue);
  tagCard(kit, fonts, root, Z2, "time per token: repeat answers in ink, chat answers in glass", "lower is better", kit.palette.ink);

  const bars = n.bars.map(() => {
    const tokens = liquid(kit.blue);
    const lit = liquid(kit.pink);
    const ms = block(kit, "ink");
    const chat = block(kit, "glass");
    root.add(tokens, lit, ms, chat);
    return { tokens, lit, ms, chat, grown: 0 };
  });
  let chosen = 0;
  const readout = readoutCard(kit, fonts, root, 0.1, () => n.bars[chosen].lines);
  const rail = makeRail(kit, fonts, root, 0.27, xs, "the most guesses a step may carry", `${n.bars[0].k} to ${n.bars[n.bars.length - 1].k}`);
  const pick = picker(rail, 0, n.bars.length - 1, 3, (i) => {
    chosen = i;
    readout.redraw();
  });
  return {
    root,
    pick,
    update: (dt, reduced) => {
      pick.update(dt, reduced);
      bars.forEach((b, i) => {
        const bar = n.bars[i];
        b.grown = approach(b.grown, 1, dt * 1.2, reduced);
        const h = Math.max(0.002, (bar.tokensPerStep / mostTokens) * 0.14 * b.grown);
        lay(b.tokens, xs[i] - bw, bw * 2, h, Z1, D);
        lay(b.lit, xs[i] - bw, bw * 2, h, Z1, D);
        b.tokens.visible = i !== chosen;
        b.lit.visible = i === chosen;
        lay(b.ms, xs[i] - bw * 1.25, bw * 1.2, Math.max(0.002, (bar.ms / mostMs) * 0.13 * b.grown), Z2, D);
        if (bar.chatMs !== null) lay(b.chat, xs[i] + bw * 0.05, bw * 1.2, Math.max(0.002, (bar.chatMs / mostMs) * 0.13 * b.grown), Z2, D);
        else b.chat.visible = false;
      });
    },
  };
}

export function createSpeculativeModel(kit: Kit, fonts: Fonts, n: SpecNumbers): BookModel {
  return joinParts([guessExhibit(kit, fonts, n), checkExhibit(kit, fonts, n), payoffExhibit(kit, fonts, n)]);
}
