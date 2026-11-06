import * as THREE from "three";
import { chunkedPrefillAboutNumbers } from "@/components/kernels/chunked-prefill/about";
import { chunkedSteps, fmt, prefillMs } from "@/content/kernels/chunked-prefill";
import type { BookModel } from "../book-scene";
import { createCard, inkSegments, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";

const { lengths, promptTokens, decodeSteps, decoders, budget } = chunkedPrefillAboutNumbers;
const tokens = lengths.map((l) => Number(l.replace(",", "")));

const CYCLE = 16;
const PREFILL = [0.8, 3.0];
const DECODE_AT = (k: number) => 4.0 + k * 2.2;
const SWEEP = [1.0, 12.0];
const FADE = CYCLE - 0.8;

const CUBE = 0.034;
const PITCH = 0.044;
const PROMPT_X = 0.09;
const ROW_Z = -0.37;
const MODEL_X = 0.43;
const CACHE_X = 0.55;
const CACHE_Z = -0.42;
const OUT_Z = -0.33;

const LANE = 0.026;
const STRIP_X = [0.15, 0.94];
const STRIP_Z = [-0.135, 0.015];
const PER_TOKEN = 0.58 / tokens[tokens.length - 1];
const MIN_W = 0.03;
const GAP = 0.006;
const BAR_H = 0.018;

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const approach = (value: number, goal: number, dt: number, reduced: boolean) => (reduced ? goal : value + (goal - value) * Math.min(1, dt * 4));

type Pen = (text: string, x: number, y: number, size: number, color: string, serif?: boolean) => void;
const pen =
  (ctx: CanvasRenderingContext2D, fonts: Fonts, w: number, h: number): Pen =>
  (text, x, y, size, color, serif = false) => {
    ctx.fillStyle = color;
    ctx.font = `${h * size}px ${serif ? fonts.serif : fonts.sans}`;
    ctx.fillText(text, w * x, h * y);
  };

function block(kit: Kit, w: number, h: number, d: number, material?: THREE.Material) {
  const pivot = new THREE.Group();
  const body = kit.solid(new THREE.BoxGeometry(w, h, d), { material, threshold: 40 });
  body.position.y = h / 2;
  pivot.add(body);
  return pivot;
}

function slot(points: number[], x: number, z: number, w: number, d: number) {
  const a = [x - w / 2, z - d / 2];
  const b = [x + w / 2, z + d / 2];
  points.push(a[0], 0.002, a[1], b[0], 0.002, a[1], b[0], 0.002, a[1], b[0], 0.002, b[1], b[0], 0.002, b[1], a[0], 0.002, b[1], a[0], 0.002, b[1], a[0], 0.002, a[1]);
}

function dashed(points: number[], x0: number, x1: number, z: number, d: number) {
  const pieces = Math.max(2, Math.round((x1 - x0) / 0.014));
  for (const zz of [z - d / 2, z + d / 2]) {
    for (let k = 0; k < pieces; k += 2) points.push(x0 + ((x1 - x0) * k) / pieces, 0.003, zz, x0 + ((x1 - x0) * (k + 1)) / pieces, 0.003, zz);
  }
  points.push(x0, 0.003, z - d / 2, x0, 0.003, z + d / 2, x1, 0.003, z - d / 2, x1, 0.003, z + d / 2);
}

type Piece = { start: number; width: number; rise: THREE.Group[]; bar?: THREE.Group; wait?: THREE.Object3D };

function buildStrip(kit: Kit, length: number, chunked: boolean, z0: number) {
  const group = new THREE.Group();
  const pieces: Piece[] = [];
  const lane = (i: number) => z0 + i * LANE;
  let x = STRIP_X[0];
  const push = (width: number, decode: boolean[], chunk: boolean, wait: boolean) => {
    const piece: Piece = { start: x, width, rise: [] };
    decode.forEach((d, i) => {
      if (!d) return;
      const cube = block(kit, MIN_W - 0.006, BAR_H, LANE - 0.007, kit.blue);
      cube.position.set(x + (MIN_W - 0.006) / 2, 0, lane(i));
      group.add(cube);
      piece.rise.push(cube);
    });
    if (chunk) {
      const offset = decode[0] ? MIN_W - 0.003 : 0;
      const w = Math.max(0.004, width - offset);
      const bar = new THREE.Group();
      const body = block(kit, w, BAR_H, LANE - 0.007, kit.pink);
      body.position.x = w / 2;
      bar.add(body);
      bar.position.set(x + offset, 0, lane(3));
      group.add(bar);
      piece.bar = bar;
    }
    if (wait) {
      const points: number[] = [];
      for (let i = 0; i < 3; i++) dashed(points, x, x + width, lane(i), LANE - 0.007);
      const lines = inkSegments(points, kit.thin, 0.0004);
      group.add(lines);
      piece.wait = lines;
    }
    pieces.push(piece);
    x += width + GAP;
  };
  push(MIN_W, [true, true, true, false], false, false);
  if (chunked) {
    for (const s of chunkedSteps(budget, decoders, length)) push(Math.max(MIN_W, (s.decode + s.chunk) * PER_TOKEN), [true, true, true, false], true, false);
  } else {
    push(Math.max(MIN_W, length * PER_TOKEN), [false, false, false, false], true, true);
  }
  while (x + MIN_W <= STRIP_X[1]) push(MIN_W, [true, true, true, true], false, false);
  return { group, pieces };
}

function dispose(object: THREE.Object3D) {
  object.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (mesh.geometry) mesh.geometry.dispose();
  });
}

export function createChunkedPrefillAboutModel(kit: Kit, fonts: Fonts): BookModel {
  const root = new THREE.Group();
  const wash = washPlane(kit.palette.pink, 1, 0.42, 0.1);
  wash.position.set(0.5, 0.002, -0.36);
  root.add(wash);
  const washLow = washPlane(kit.palette.blue, 1, 0.36, 0.1);
  washLow.position.set(0.52, 0.002, -0.04);
  root.add(washLow);

  const title = createCard(kit, 0.86, 0.11, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("prefill: the whole prompt in one pass, fills the KV cache, gives token 1", 0.03, 0.32, 0.2, kit.palette.ink, true);
    write("decode: one new token for each request in each step", 0.03, 0.62, 0.2, kit.palette.ink, true);
    write("chunked prefill: the prompt fed in pieces, beside the decode rows, under a token budget per step", 0.03, 0.88, 0.13, kit.palette.faint);
  });
  title.group.position.set(0.5, 0, -0.6);
  root.add(title.group);

  const prompt = Array.from({ length: promptTokens }, (_, i) => {
    const cube = block(kit, CUBE, 0.03, CUBE, kit.pink);
    cube.position.set(PROMPT_X + i * PITCH, 0, ROW_Z);
    root.add(cube);
    return cube;
  });
  const input = block(kit, CUBE, 0.03, CUBE, kit.blue);
  const inputX = PROMPT_X + (promptTokens - 1) * PITCH;
  input.position.set(inputX, 0, ROW_Z);
  input.visible = false;
  root.add(input);

  const model = block(kit, 0.1, 0.07, 0.13);
  model.position.set(MODEL_X, 0, ROW_Z);
  root.add(model);

  const outlines: number[] = [];
  const cache = Array.from({ length: promptTokens + decodeSteps }, (_, i) => {
    const x = CACHE_X + i * PITCH;
    slot(outlines, x, CACHE_Z, CUBE, 0.03);
    const slab = block(kit, CUBE, 0.035, 0.03, i < promptTokens ? kit.pink : kit.blue);
    slab.position.set(x, 0, CACHE_Z);
    root.add(slab);
    return slab;
  });
  const outs = Array.from({ length: decodeSteps + 1 }, (_, i) => {
    const x = CACHE_X + i * PITCH;
    slot(outlines, x, OUT_Z, CUBE, CUBE);
    const cube = block(kit, CUBE, 0.03, CUBE, kit.blue);
    cube.position.set(x, 0, OUT_Z);
    root.add(cube);
    return cube;
  });
  root.add(inkSegments(outlines, kit.thin, 0.0004));

  let phase = -1;
  const phaseText = (p: number) => {
    if (p === 0) return `prefill: all ${promptTokens} prompt tokens in one pass; the cache fills and token 1 comes out`;
    if (p <= decodeSteps) return `decode step ${p}: one token in, one more row of cache, token ${p + 1} out`;
    return "prefill fills the cache at once; each decode step adds one row and one token";
  };
  const top = createCard(kit, 0.86, 0.06, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write(`prompt, ${promptTokens} tokens`, 0.02, 0.38, 0.32, kit.palette.pink);
    write("model, 22 layers", 0.33, 0.38, 0.32, kit.palette.ink);
    write("KV cache (back), tokens out (front)", 0.56, 0.38, 0.32, kit.palette.ink);
    write(phaseText(Math.max(0, phase)), 0.02, 0.82, 0.3, phase > 0 && phase <= decodeSteps ? kit.palette.blue : kit.palette.faint);
  });
  top.group.position.set(0.5, 0, -0.26);
  root.add(top.group);

  const header = (z: number, draw: (write: Pen) => void) => {
    const card = createCard(kit, 0.82, 0.03, (ctx, w, h) => draw(pen(ctx, fonts, w, h)));
    card.group.position.set(0.53, 0, z);
    root.add(card.group);
  };
  header(STRIP_Z[0] - 0.04, (write) => {
    write("the prompt as one step:", 0.01, 0.72, 0.62, kit.palette.pink);
    write("A, B and C wait for all of it", 0.3, 0.72, 0.62, kit.palette.ink);
  });
  header(STRIP_Z[1] - 0.035, (write) => {
    write(`in chunks under a budget of ${budget} rows:`, 0.01, 0.72, 0.62, kit.palette.pink);
    write("A, B and C get a token in every step", 0.44, 0.72, 0.62, kit.palette.blue);
  });
  const tagTilt = -1.2;
  const tagH = (4 * LANE) / Math.sin(-tagTilt);
  STRIP_Z.forEach((z0) => {
    const tag = createCard(
      kit,
      0.034,
      tagH,
      (ctx, w, h) => {
        ctx.font = `${(h / 4) * 0.62}px ${fonts.serif}`;
        ctx.textAlign = "center";
        ["A", "B", "C", "P"].forEach((name, i) => {
          ctx.fillStyle = i < 3 ? kit.palette.blue : kit.palette.pink;
          ctx.fillText(name, w * 0.5, (h * (i + 0.5)) / 4 + (h / 4) * 0.22);
        });
      },
      tagTilt,
    );
    tag.group.position.set(0.11, 0, z0 - 0.5 * LANE);
    root.add(tag.group);
  });

  let pick = tokens.length - 1;
  const readout = createCard(kit, 0.86, 0.08, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    const length = tokens[pick];
    const ms = prefillMs("engine", lengths[pick]);
    const steps = chunkedSteps(budget, decoders, length).length;
    write(`a ${lengths[pick]}-token prompt: its prefill alone takes ${fmt(ms)} ms, measured`, 0.03, 0.34, 0.3, kit.palette.pink, true);
    write(`as one step: ${lengths[pick]} rows, and every decoding request waits for all of them`, 0.03, 0.62, 0.21, kit.palette.ink);
    write(`in chunks: ${steps} step${steps > 1 ? "s" : ""} of at most ${budget} rows, with ${decoders} decode rows in each`, 0.03, 0.88, 0.21, kit.palette.blue);
  });
  readout.group.position.set(0.5, 0, 0.165);
  root.add(readout.group);

  const railZ = 0.27;
  const railX = [0.3, 0.86];
  const stop = (k: number) => railX[0] + (k / (tokens.length - 1)) * (railX[1] - railX[0]);
  const rail = kit.solid(new THREE.BoxGeometry(railX[1] - railX[0] + 0.04, 0.016, 0.02));
  rail.position.set((railX[0] + railX[1]) / 2, 0.008, railZ);
  const ticks: number[] = [];
  tokens.forEach((_, k) => ticks.push(stop(k), 0.017, railZ - 0.018, stop(k), 0.017, railZ + 0.018));
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.028, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(stop(pick), 0.045, railZ);
  const railCard = createCard(kit, 0.86, 0.045, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("prompt tokens", 0.01, 0.7, 0.46, kit.palette.faint);
    ctx.textAlign = "center";
    lengths.forEach((l, k) => write(l, (stop(k) - 0.07) / 0.86, 0.7, 0.46, kit.palette.pink));
    ctx.textAlign = "left";
  });
  railCard.group.position.set(0.5, 0, railZ + 0.05);
  const note = createCard(kit, 0.86, 0.035, (ctx, w, h) => {
    pen(ctx, fonts, w, h)(`illustrative: steps drawn in proportion to their rows, ${decoders} decoders; prefill times from docs/results.md`, 0.03, 0.7, 0.5, kit.palette.faint);
  });
  note.group.position.set(0.5, 0, 0.385);
  root.add(rail, inkSegments(ticks, kit.thin), knob, railCard.group, note.group);

  let strips = [buildStrip(kit, tokens[pick], false, STRIP_Z[0]), buildStrip(kit, tokens[pick], true, STRIP_Z[1])];
  const stripRoot = new THREE.Group();
  strips.forEach((s) => stripRoot.add(s.group));
  root.add(stripRoot);
  let shown = 1;

  const setPick = (k: number) => {
    if (k === pick) return;
    pick = k;
    readout.redraw();
    strips.forEach((s) => {
      stripRoot.remove(s.group);
      dispose(s.group);
    });
    strips = [buildStrip(kit, tokens[pick], false, STRIP_Z[0]), buildStrip(kit, tokens[pick], true, STRIP_Z[1])];
    strips.forEach((s) => stripRoot.add(s.group));
    shown = 0;
  };

  let held = false;
  let knobX = stop(pick);
  let time = 0;
  return {
    exhibits: [root],
    handles: [knob],
    dragStart: () => {
      held = true;
    },
    drag: (x) => {
      knobX = Math.min(railX[1], Math.max(railX[0], x));
      setPick(Math.round(((knobX - railX[0]) / (railX[1] - railX[0])) * (tokens.length - 1)));
    },
    dragEnd: () => {
      held = false;
    },
    enter: () => {
      time = 0;
    },
    update: (dt, _exhibit, _stop, reduced) => {
      knob.position.x = held ? knobX : approach(knob.position.x, stop(pick), dt * 0.7, reduced);
      time = reduced ? FADE - 0.1 : (time + dt) % CYCLE;
      const fade = reduced ? 1 : smooth(0, 0.7, time) * (1 - smooth(FADE, CYCLE, time));

      const pass = reduced ? 0 : smooth(PREFILL[0], PREFILL[1], time);
      prompt.forEach((cube, i) => {
        const s = Math.max(0.0001, reduced ? 1 : smooth(0, 0.7, time) * (1 - smooth(PREFILL[1] - 0.8, PREFILL[1], time)));
        cube.position.x = PROMPT_X + i * PITCH + 0.07 * pass;
        cube.scale.set(s, s, s);
        cube.visible = s > 0.01;
      });
      let inX = 0;
      let inS = 0;
      for (let k = 0; k < decodeSteps; k++) {
        const t0 = DECODE_AT(k);
        if (time >= t0 && time < t0 + 1.6) {
          inS = smooth(t0, t0 + 0.5, time) * (1 - smooth(t0 + 1.0, t0 + 1.5, time));
          inX = 0.07 * smooth(t0 + 0.4, t0 + 1.4, time);
        }
      }
      input.visible = !reduced && inS > 0.01;
      input.position.x = inputX + inX;
      input.scale.set(Math.max(0.0001, inS), Math.max(0.0001, inS), Math.max(0.0001, inS));

      cache.forEach((slab, i) => {
        const at = i < promptTokens ? PREFILL[1] - 0.9 : DECODE_AT(i - promptTokens) + 0.9;
        const h = reduced ? 1 : smooth(at, at + 0.8, time) * fade;
        slab.scale.y = Math.max(0.0001, h);
        slab.visible = h > 0.01;
      });
      outs.forEach((cube, i) => {
        const at = i === 0 ? PREFILL[1] - 0.2 : DECODE_AT(i - 1) + 1.2;
        const h = reduced ? 1 : smooth(at, at + 0.8, time) * fade;
        cube.scale.y = Math.max(0.0001, h);
        cube.visible = h > 0.01;
      });

      let next = decodeSteps + 1;
      if (!reduced) {
        if (time < DECODE_AT(0)) next = 0;
        else for (let k = 0; k < decodeSteps; k++) if (time >= DECODE_AT(k) && time < DECODE_AT(k) + 2.2) next = k + 1;
      }
      if (next !== phase) {
        phase = next;
        top.redraw();
      }

      shown = approach(shown, 1, dt * 0.45, reduced);
      const sweep = reduced ? STRIP_X[1] + 0.1 : STRIP_X[0] + (STRIP_X[1] - STRIP_X[0] + 0.05) * smooth(SWEEP[0], SWEEP[1], time);
      const strip = fade * shown;
      for (const s of strips) {
        for (const p of s.pieces) {
          const rise = smooth(p.start, p.start + 0.03, sweep) * strip;
          for (const cube of p.rise) {
            cube.scale.y = Math.max(0.0001, rise);
            cube.visible = rise > 0.01;
          }
          if (p.bar) {
            const across = Math.min(1, Math.max(0, (sweep - p.start) / Math.max(p.width, 0.03)));
            p.bar.scale.set(Math.max(0.0001, across), Math.max(0.0001, strip), 1);
            p.bar.visible = across > 0.005 && strip > 0.01;
          }
          if (p.wait) p.wait.visible = sweep > p.start && strip > 0.3;
        }
      }
    },
  };
}
