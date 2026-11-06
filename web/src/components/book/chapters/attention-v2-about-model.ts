import * as THREE from "three";
import type { OnlineState } from "@/components/kernels/attention-v2/about";
import type { BookModel } from "../book-scene";
import { createCard, inkSegments, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";

const PEG = 0.044;
const PITCH = 0.074;
const X0 = 0.38;
const SCORE_Z = -0.36;
const TERM_Z = -0.17;
const BASE_H = 0.012;
const PER_SCORE = 0.05;
const TERM_H = 0.09;
const STEP = 2.4;
const REST = 3.6;

const approach = (value: number, goal: number, dt: number, reduced: boolean) => (reduced ? goal : value + (goal - value) * Math.min(1, dt * 2.2));

type Pen = (text: string, x: number, y: number, size: number, color: string, serif?: boolean) => void;
const pen =
  (ctx: CanvasRenderingContext2D, fonts: Fonts, w: number, h: number): Pen =>
  (text, x, y, size, color, serif = false) => {
    ctx.fillStyle = color;
    ctx.font = `${h * size}px ${serif ? fonts.serif : fonts.sans}`;
    ctx.fillText(text, w * x, h * y);
  };

function column(kit: Kit, height: number, material?: THREE.Material) {
  const pivot = new THREE.Group();
  const body = kit.solid(new THREE.BoxGeometry(PEG, height, PEG), { material });
  body.position.y = height / 2;
  pivot.add(body);
  return pivot;
}

export function createAttentionV2AboutModel(kit: Kit, fonts: Fonts, states: OnlineState[]): BookModel {
  const n = states.length;
  const root = new THREE.Group();
  const wash = washPlane(kit.palette.blue, 1, 0.42, 0.12);
  wash.position.set(0.5, 0.002, -0.27);
  root.add(wash);

  const title = createCard(kit, 0.86, 0.13, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("softmax: scores in, weights that add up to 1 out", 0.03, 0.29, 0.23, kit.palette.ink, true);
    write("weight = e^(s − m) / l,   m = the largest s,   l = Σ e^(s − m)", 0.03, 0.59, 0.21, kit.palette.ink, true);
    write("online softmax: keep m and l as running values, read the scores once", 0.03, 0.87, 0.16, kit.palette.ink);
  });
  title.group.position.set(0.5, 0, -0.55);
  root.add(title.group);

  const x = (i: number) => X0 + i * PITCH;
  const height = (s: number) => BASE_H + s * PER_SCORE;
  const pegs = states.map((st, i) => {
    const peg = column(kit, height(st.score));
    peg.position.set(x(i), 0, SCORE_Z);
    root.add(peg);
    const base = new THREE.Mesh(new THREE.BoxGeometry(PEG, 0.003, PEG), kit.ink);
    base.position.set(x(i), 0.0015, TERM_Z);
    const term = column(kit, TERM_H, kit.blue);
    term.position.set(x(i), 0, TERM_Z);
    term.scale.y = 0.0001;
    term.visible = false;
    root.add(base, term);
    return { peg, term, value: 0 };
  });
  const ceiling: number[] = [];
  states.forEach((_, i) => ceiling.push(x(i) - PEG / 2, TERM_H, TERM_Z, x(i) + PEG / 2, TERM_H, TERM_Z));
  root.add(inkSegments(ceiling, kit.thin));

  const slabWidth = (count: number) => (count - 1) * PITCH + PEG + 0.03;
  const slab = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 1, 16), kit.pink);
  slab.rotation.z = Math.PI / 2;
  slab.visible = false;
  root.add(slab);
  let slabY = height(states[0].score);
  let slabW = slabWidth(1);

  const label = (z: number, head: string, sub: string, color: string) => {
    const card = createCard(kit, 0.25, 0.08, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      write(head, 0.05, 0.5, 0.42, color, true);
      write(sub, 0.05, 0.86, 0.22, kit.palette.ink);
    });
    card.group.position.set(0.195, 0, z);
    root.add(card.group);
  };
  label(SCORE_Z + 0.02, "scores s", "pink rod: the running max m", kit.palette.pink);
  label(TERM_Z + 0.02, "e^(s − m)", "blue: the terms of l, at most 1", kit.palette.blue);

  let seen = n;
  const readout = createCard(kit, 0.86, 0.13, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    const st = states[seen - 1];
    const head = `${seen} of ${n} scores seen: m = ${st.max.toFixed(1)}, l = ${st.sum.toFixed(2)}`;
    write(head, 0.03, 0.28, 0.24, kit.palette.ink, true);
    const what =
      seen === 1
        ? `score 1 = ${st.score.toFixed(1)} starts the run: m = ${st.score.toFixed(1)}, l = e^0 = 1`
        : st.rose
          ? `score ${seen} = ${st.score.toFixed(1)} beats m: l × e^(${st.before.toFixed(1)} − ${st.max.toFixed(1)}) = × ${Math.exp(st.before - st.max).toFixed(2)}, then + 1`
          : `score ${seen} = ${st.score.toFixed(1)} is below m: l + e^(${st.score.toFixed(1)} − ${st.max.toFixed(1)}) = + ${Math.exp(st.score - st.max).toFixed(2)}`;
    write(what, 0.03, 0.56, 0.18, st.rose ? kit.palette.pink : kit.palette.blue);
    write(seen < n ? "no weight is final yet: a later score could still raise m" : "all seen: m and l are exact, each weight is its blue term ÷ l", 0.03, 0.84, 0.17, kit.palette.ink);
  });
  readout.group.position.set(0.5, 0, 0.04);
  root.add(readout.group);

  const railZ = 0.22;
  const railX = [0.2, 0.8];
  const rail = kit.solid(new THREE.BoxGeometry(railX[1] - railX[0] + 0.04, 0.016, 0.02));
  rail.position.set(0.5, 0.008, railZ);
  const ticks: number[] = [];
  const tickX = (k: number) => railX[0] + ((k - 1) / (n - 1)) * (railX[1] - railX[0]);
  for (let k = 1; k <= n; k++) ticks.push(tickX(k), 0.017, railZ - 0.018, tickX(k), 0.017, railZ + 0.018);
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.028, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(railX[1], 0.045, railZ);
  const railCard = createCard(kit, 0.6, 0.045, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("scores seen so far", 0.02, 0.7, 0.5, kit.palette.faint);
    write(`1 to ${n}`, 0.82, 0.7, 0.5, kit.palette.pink);
  });
  railCard.group.position.set(0.5, 0, railZ + 0.06);
  const note = createCard(kit, 0.86, 0.04, (ctx, w, h) => {
    pen(ctx, fonts, w, h)(`illustrative: ${n} example scores; a real query head has one for every token of the context`, 0.03, 0.7, 0.5, kit.palette.faint);
  });
  note.group.position.set(0.5, 0, 0.38);
  root.add(rail, inkSegments(ticks, kit.thin), knob, railCard.group, note.group);

  const setSeen = (k: number) => {
    if (k === seen) return;
    seen = k;
    readout.redraw();
  };

  let held = false;
  let chosen = false;
  let knobX = railX[1];
  let clock = 0;
  return {
    exhibits: [root],
    handles: [knob],
    dragStart: () => {
      held = true;
      chosen = true;
    },
    drag: (px) => {
      knobX = Math.min(railX[1], Math.max(railX[0], px));
      setSeen(1 + Math.round(((knobX - railX[0]) / (railX[1] - railX[0])) * (n - 1)));
    },
    dragEnd: () => {
      held = false;
    },
    enter: () => {
      clock = 0;
      chosen = false;
    },
    update: (dt, _exhibit, _stop, reduced) => {
      if (!chosen) {
        if (reduced) setSeen(n);
        else {
          clock = (clock + dt) % (n * STEP + REST);
          setSeen(Math.min(n, 1 + Math.floor(clock / STEP)));
        }
      }
      knob.position.x = held ? knobX : approach(knob.position.x, tickX(seen), dt * 0.7, reduced);
      const st = states[seen - 1];
      pegs.forEach((p, i) => {
        const goal = i < seen ? Math.exp(states[i].score - st.max) : 0;
        p.value = approach(p.value, goal, dt, reduced);
        p.term.scale.y = Math.max(0.0001, p.value);
        p.term.visible = p.value > 0.01;
      });
      slabY = approach(slabY, height(st.max), dt, reduced);
      slabW = approach(slabW, slabWidth(seen), dt, reduced);
      slab.visible = true;
      slab.scale.y = slabW;
      slab.position.set(X0 - PEG / 2 - 0.015 + slabW / 2, slabY + 0.008, SCORE_Z + PEG / 2);
    },
  };
}
