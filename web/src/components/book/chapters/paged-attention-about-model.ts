import * as THREE from "three";
import { aboutBlock, aboutScores, aboutTable, aboutTokens, aboutWeights } from "@/components/kernels/attention/about";
import type { BookModel } from "../book-scene";
import { createCard, inkSegments, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";

const N = aboutTokens.length;
const NEWEST = N - 1;
const POOL = 4;
const tokX = (i: number) => 0.12 + i * 0.108;
const blockX = (p: number) => 0.17 + p * 0.22;
const slotX = (p: number, s: number) => blockX(p) + (s - (aboutBlock - 1) / 2) * 0.044;
const where = (i: number) => ({ logical: Math.floor(i / aboutBlock), slot: i % aboutBlock, physical: aboutTable[Math.floor(i / aboutBlock)] });
const BAR_Z = -0.43;
const WORD_Z = -0.33;
const TABLE_Z = -0.235;
const POOL_Z = -0.1;
const CELL = 0.034;
const SLOT_H = 0.022;
const TALL = 0.1;
const STEP = 0.8;
const SCAN = 0.5;
const SOFT = SCAN + N * STEP + 0.3;
const CYCLE = SOFT + 6.4;

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const bump = (a: number, b: number, x: number) => smooth(a, a + 0.35, x) * (1 - smooth(b - 0.35, b, x));
const approach = (value: number, goal: number, dt: number, reduced: boolean) => (reduced ? goal : value + (goal - value) * Math.min(1, dt * 4));

type Pen = (text: string, x: number, y: number, size: number, color: string, serif?: boolean, align?: CanvasTextAlign) => void;
const pen =
  (ctx: CanvasRenderingContext2D, fonts: Fonts, w: number, h: number): Pen =>
  (text, x, y, size, color, serif = false, align = "left") => {
    ctx.fillStyle = color;
    ctx.font = `${h * size}px ${serif ? fonts.serif : fonts.sans}`;
    ctx.textAlign = align;
    ctx.fillText(text, w * x, h * y);
    ctx.textAlign = "left";
  };

function column(kit: Kit, material: THREE.Material | undefined, width: number) {
  const pivot = new THREE.Group();
  const body = kit.solid(new THREE.BoxGeometry(width, 1, width), { material });
  body.position.y = 0.5;
  pivot.add(body);
  pivot.scale.y = 0.0001;
  return pivot;
}

function frame(kit: Kit, w: number, d: number) {
  const group = new THREE.Group();
  const t = 0.005;
  const bar = (bw: number, bd: number, x: number, z: number) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(bw, 0.004, bd), kit.ink);
    m.position.set(x, 0.002, z);
    group.add(m);
  };
  bar(w, t, 0, -d / 2);
  bar(w, t, 0, d / 2);
  bar(t, d, -w / 2, 0);
  bar(t, d, w / 2, 0);
  return group;
}

export function createPagedAttentionAboutModel(kit: Kit, fonts: Fonts): BookModel {
  const root = new THREE.Group();
  const wash = washPlane(kit.palette.blue, 1, 0.5, 0.12);
  wash.position.set(0.5, 0.002, -0.3);
  root.add(wash);

  const title = createCard(kit, 0.86, 0.11, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("paged attention", 0.03, 0.34, 0.3, kit.palette.ink, true);
    write("attention: the newest token's query q is scored against every key k in the cache", 0.03, 0.63, 0.17, kit.palette.ink);
    write("paged: keys and values sit in fixed-size blocks, found through a block table", 0.03, 0.88, 0.17, kit.palette.faint);
  });
  title.group.position.set(0.5, 0, -0.585);
  root.add(title.group);

  let selected = 1;

  const tokens = aboutTokens.map((word, i) => {
    const score = column(kit, undefined, 0.03);
    score.position.set(tokX(i) - 0.018, 0, BAR_Z);
    const weight = column(kit, kit.blue, 0.03);
    weight.position.set(tokX(i) + 0.018, 0, BAR_Z);
    const card = createCard(kit, 0.096, 0.055, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      const color = i === NEWEST ? kit.palette.pink : kit.palette.ink;
      write(word, 0.5, 0.5, 0.42, color, true, "center");
      write(i === NEWEST ? "newest: q" : `token ${i}`, 0.5, 0.86, 0.24, i === NEWEST ? kit.palette.pink : kit.palette.faint, false, "center");
      if (i === selected) {
        ctx.strokeStyle = kit.palette.ink;
        ctx.lineWidth = h * 0.06;
        ctx.strokeRect(h * 0.05, h * 0.05, w - h * 0.1, h * 0.9);
      }
    });
    card.group.position.set(tokX(i), 0, WORD_Z);
    root.add(score, weight, card.group);
    return { score, weight, card };
  });

  const query = new THREE.Mesh(new THREE.SphereGeometry(0.016, 24, 18), kit.pink);
  query.castShadow = true;
  query.position.set(tokX(NEWEST), 0.03, BAR_Z + 0.045);
  root.add(query);

  const lines: number[] = [];
  aboutTable.forEach((p, k) => {
    const width = aboutBlock * 0.108 - 0.012;
    const x = (tokX(k * aboutBlock) + tokX(k * aboutBlock + aboutBlock - 1)) / 2;
    const entry = createCard(kit, width, 0.045, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      write("block table:", 0.04, 0.68, 0.42, kit.palette.faint);
      write(`logical ${k} → physical ${p}`, 0.34, 0.68, 0.46, kit.palette.ink, true);
    });
    entry.group.position.set(x, 0, TABLE_Z);
    root.add(entry.group);
    lines.push(x, 0.003, TABLE_Z + 0.006, blockX(p), 0.003, POOL_Z - 0.04);
  });
  root.add(inkSegments(lines, kit.line));

  const slots = new Map<number, THREE.Group>();
  for (let p = 0; p < POOL; p++) {
    const plate = kit.solid(new THREE.BoxGeometry(0.19, 0.008, 0.07));
    plate.position.set(blockX(p), 0.004, POOL_Z);
    root.add(plate);
    if (p === 2) continue;
    for (let s = 0; s < aboutBlock; s++) {
      const pivot = new THREE.Group();
      const cube = kit.solid(new THREE.BoxGeometry(CELL, SLOT_H, CELL * 1.3));
      cube.position.y = 0.008 + SLOT_H / 2;
      pivot.add(cube);
      pivot.position.set(slotX(p, s), 0, POOL_Z);
      root.add(pivot);
      if (p !== 0) slots.set(p * aboutBlock + s, pivot);
    }
  }
  const pick = frame(kit, CELL + 0.014, CELL * 1.3 + 0.014);
  root.add(pick);

  for (let p = 0; p < POOL; p++) {
    const card = createCard(kit, 0.19, 0.04, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      write(`block ${p}`, 0.04, 0.72, 0.52, aboutTable.includes(p) ? kit.palette.ink : kit.palette.faint, true);
      if (p === 0) write("another request", 0.42, 0.72, 0.4, kit.palette.faint);
      if (p === 2) write("free", 0.42, 0.72, 0.4, kit.palette.faint);
      if (aboutTable.includes(p)) write(`logical ${aboutTable.indexOf(p)}`, 0.42, 0.72, 0.4, kit.palette.faint);
    });
    card.group.position.set(blockX(p), 0, POOL_Z + 0.05);
    root.add(card.group);
  }

  const readout = createCard(kit, 0.86, 0.1, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    const at = where(selected);
    write(`"${aboutTokens[selected]}", token ${selected}: score ${aboutScores[selected].toFixed(1)}, weight ${aboutWeights[selected].toFixed(2)}`, 0.03, 0.34, 0.27, kit.palette.blue, true);
    write(`its key and value: logical block ${at.logical}, slot ${at.slot} → block table → physical block ${at.physical}, slot ${at.slot}`, 0.03, 0.62, 0.21, kit.palette.ink);
    write(`out = Σ weight × value over all ${N} tokens; the ${N} weights add up to 1`, 0.03, 0.88, 0.21, kit.palette.ink);
  });
  readout.group.position.set(0.5, 0, 0.1);
  root.add(readout.group);

  const railZ = 0.235;
  const railX = [tokX(0), tokX(NEWEST)];
  const rail = kit.solid(new THREE.BoxGeometry(railX[1] - railX[0] + 0.04, 0.016, 0.02));
  rail.position.set(0.5 * (railX[0] + railX[1]), 0.008, railZ);
  const ticks: number[] = [];
  for (let i = 0; i < N; i++) ticks.push(tokX(i), 0.017, railZ - 0.018, tokX(i), 0.017, railZ + 0.018);
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.028, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(tokX(selected), 0.045, railZ);
  const railCard = createCard(kit, 0.6, 0.045, (ctx, w, h) => {
    pen(ctx, fonts, w, h)("pick a token: where its key and value are read from", 0.02, 0.7, 0.5, kit.palette.faint);
  });
  railCard.group.position.set(0.5, 0, railZ + 0.055);
  const note = createCard(kit, 0.86, 0.04, (ctx, w, h) => {
    pen(ctx, fonts, w, h)("illustrative: made-up scores, blocks of 4; the engine uses 16-token blocks and 64 dimensions per head", 0.03, 0.7, 0.46, kit.palette.faint);
  });
  note.group.position.set(0.5, 0, 0.38);
  root.add(rail, inkSegments(ticks, kit.thin), knob, railCard.group, note.group);

  const select = (i: number) => {
    if (i === selected) return;
    const old = selected;
    selected = i;
    tokens[old].card.redraw();
    tokens[i].card.redraw();
    readout.redraw();
  };
  pick.position.set(slotX(where(selected).physical, where(selected).slot), 0, POOL_Z);

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
      select(Math.round((knobX - railX[0]) / 0.108));
    },
    dragEnd: () => {
      held = false;
    },
    enter: () => {
      time = 0;
    },
    update: (dt, _exhibit, _stop, reduced) => {
      knob.position.x = held ? knobX : approach(knob.position.x, tokX(selected), dt * 0.7, reduced);
      const at = where(selected);
      pick.position.x = approach(pick.position.x, slotX(at.physical, at.slot), dt * 0.5, reduced);
      time = reduced ? CYCLE - 1 : (time + dt) % CYCLE;
      const fade = reduced ? 1 : 1 - smooth(CYCLE - 0.8, CYCLE, time);
      const soft = reduced ? 1 : smooth(SOFT, SOFT + 1.2, time);
      let qx = tokX(NEWEST);
      tokens.forEach((t, i) => {
        const start = SCAN + (NEWEST - i) * STEP;
        const scored = reduced ? 1 : smooth(start, start + STEP, time);
        const s = (aboutScores[i] / Math.max(...aboutScores)) * TALL * scored * (1 - soft) * fade;
        t.score.scale.y = Math.max(0.0001, s);
        t.score.visible = s > 0.002;
        const w = (aboutWeights[i] / Math.max(...aboutWeights)) * TALL * soft * fade;
        t.weight.scale.y = Math.max(0.0001, w);
        t.weight.visible = w > 0.002;
        const on = reduced ? 0 : bump(start, start + STEP, time);
        const slot = slots.get(where(i).physical * aboutBlock + where(i).slot);
        if (slot) slot.position.y = 0.03 * on;
        if (time >= start && time < start + STEP) qx = THREE.MathUtils.lerp(tokX(Math.min(NEWEST, i + 1)), tokX(i), smooth(start, start + 0.6, time));
      });
      if (time >= SCAN + N * STEP) qx = tokX(0);
      query.position.x = reduced ? tokX(NEWEST) : time > SOFT ? approach(query.position.x, tokX(NEWEST), dt * 0.4, false) : qx;
    },
  };
}
