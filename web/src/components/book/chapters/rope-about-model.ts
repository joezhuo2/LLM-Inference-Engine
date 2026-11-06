import * as THREE from "three";
import { ABOUT_PAIR, aboutTheta } from "@/components/kernels/rope/about";
import { PAIRS } from "@/components/kernels/rope/rotation";
import type { BookModel } from "../book-scene";
import { createCard, inkSegments, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";

const SLOTS = 12;
const MOST = 7;
const START = 3;
const PITCH = 0.064;
const STRIP_X = 0.5 - (SLOTS * PITCH) / 2;
const STRIP_Z = -0.4;
const DIAL_Z = -0.14;
const SMALL = 0.08;
const BIG = 0.11;
const Q_X = 0.17;
const K_X = 0.41;
const M_X = 0.73;
const STEP = 3.2;

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const approach = (value: number, goal: number, dt: number, reduced: boolean) => (reduced ? goal : value + (goal - value) * Math.min(1, dt * 4));
const slotX = (p: number) => STRIP_X + PITCH / 2 + p * PITCH;

type Pen = (text: string, x: number, y: number, size: number, color: string, serif?: boolean) => void;
const pen =
  (ctx: CanvasRenderingContext2D, fonts: Fonts, w: number, h: number): Pen =>
  (text, x, y, size, color, serif = false) => {
    ctx.fillStyle = color;
    ctx.font = `${h * size}px ${serif ? fonts.serif : fonts.sans}`;
    ctx.fillText(text, w * x, h * y);
  };

function dial(kit: Kit, x: number, radius: number) {
  const group = new THREE.Group();
  group.position.set(x, 0, DIAL_Z);
  const drum = kit.solid(new THREE.CylinderGeometry(radius, radius * 1.04, 0.04, 56), { hull: true, threshold: 35 });
  drum.position.y = 0.02;
  const ticks: number[] = [];
  for (let t = 0; t < 16; t++) {
    const a = (t / 16) * Math.PI * 2;
    const inner = t % 4 === 0 ? radius * 0.78 : radius * 0.88;
    ticks.push(Math.cos(a) * inner, 0.0415, Math.sin(a) * inner, Math.cos(a) * radius * 0.96, 0.0415, Math.sin(a) * radius * 0.96);
  }
  group.add(drum, inkSegments(ticks, kit.thin));
  return group;
}

function arm(kit: Kit, radius: number, material: THREE.Material, y: number) {
  const group = new THREE.Group();
  group.position.y = y;
  const bar = new THREE.Mesh(new THREE.BoxGeometry(radius * 0.84, 0.008, 0.01), material);
  bar.position.x = radius * 0.42;
  bar.castShadow = true;
  const tip = new THREE.Mesh(new THREE.SphereGeometry(0.014, 18, 14), material);
  tip.position.x = radius * 0.84;
  tip.castShadow = true;
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.014, 16), kit.ink);
  group.add(bar, tip, hub);
  return group;
}

export function createRopeAboutModel(kit: Kit, fonts: Fonts): BookModel {
  const root = new THREE.Group();
  const wash = washPlane(kit.palette.blue, 1, 0.56, 0.12);
  wash.position.set(0.5, 0.002, -0.26);
  root.add(wash);

  const title = createCard(kit, 0.86, 0.11, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("RoPE: Rotary Position Embedding", 0.03, 0.4, 0.32, kit.palette.ink, true);
    write("each query and key is turned by an angle proportional to its token's position", 0.03, 0.66, 0.16, kit.palette.ink);
    write("from the RoFormer paper (2021); used by the Llama models, TinyLlama among them", 0.03, 0.9, 0.13, kit.palette.faint);
  });
  title.group.position.set(0.5, 0, -0.57);
  root.add(title.group);

  for (let p = 0; p < SLOTS; p++) {
    const box = kit.solid(new THREE.BoxGeometry(PITCH - 0.014, 0.018, 0.04));
    box.position.set(slotX(p), 0.009, STRIP_Z);
    root.add(box);
  }
  const marker = (material: THREE.Material, height: number) => {
    const m = kit.solid(new THREE.BoxGeometry(PITCH - 0.02, height, 0.034), { material });
    m.position.y = 0.018 + height / 2;
    const holder = new THREE.Group();
    holder.position.z = STRIP_Z;
    holder.add(m);
    root.add(holder);
    return holder;
  };
  const qMark = marker(kit.pink, 0.04);
  const kMark = marker(kit.blue, 0.025);
  const stripCard = createCard(kit, SLOTS * PITCH, 0.066, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    ctx.textAlign = "center";
    for (let p = 0; p < SLOTS; p++) write(String(p), (p + 0.5) / SLOTS, 0.4, 0.34, kit.palette.ink, true);
    ctx.textAlign = "left";
    write("positions in the prompt:", 0.01, 0.84, 0.24, kit.palette.faint);
    write("the query's token", 0.3, 0.84, 0.24, kit.palette.pink);
    write("the key's token, earlier", 0.53, 0.84, 0.24, kit.palette.blue);
  });
  stripCard.group.position.set(0.5, 0, STRIP_Z + 0.04);
  root.add(stripCard.group);

  const qDial = dial(kit, Q_X, SMALL);
  const kDial = dial(kit, K_X, SMALL);
  const mDial = dial(kit, M_X, BIG);
  const qArm = arm(kit, SMALL, kit.pink, 0.046);
  const kArm = arm(kit, SMALL, kit.blue, 0.046);
  const mqArm = arm(kit, BIG, kit.pink, 0.056);
  const mkArm = arm(kit, BIG, kit.blue, 0.046);
  qDial.add(qArm);
  kDial.add(kArm);
  mDial.add(mqArm, mkArm);
  const wedgeMaterial = new THREE.MeshBasicMaterial({ color: kit.palette.pink, transparent: true, opacity: 0.32, depthWrite: false, side: THREE.DoubleSide });
  const wedgeHolder = new THREE.Group();
  wedgeHolder.position.y = 0.043;
  const wedge = new THREE.Mesh(new THREE.CircleGeometry(BIG * 0.8, 48, 0, 0.001), wedgeMaterial);
  wedge.rotation.x = -Math.PI / 2;
  wedgeHolder.add(wedge);
  mDial.add(wedgeHolder);
  root.add(qDial, kDial, mDial);

  const label = (x: number, width: number, head: string, sub: string, color: string, z: number) => {
    const card = createCard(kit, width, 0.07, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      write(head, 0.05, 0.48, 0.42, color, true);
      write(sub, 0.05, 0.84, 0.26, kit.palette.ink);
    });
    card.group.position.set(x, 0, z);
    root.add(card.group);
  };
  label(Q_X, 0.22, "query", "at m, turned m × θ", kit.palette.pink, DIAL_Z + SMALL + 0.03);
  label(K_X, 0.22, "key", "at n, turned n × θ", kit.palette.blue, DIAL_Z + SMALL + 0.03);
  label(M_X, 0.27, "both together", "angle between: (m − n) × θ", kit.palette.ink, DIAL_Z + BIG + 0.02);

  let gap = START;
  let shift = 0;
  const readout = createCard(kit, 0.86, 0.1, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    const n = Math.round(shift);
    const a = gap * aboutTheta;
    write(`query at ${n + gap}, key at ${n}: ${gap} position${gap === 1 ? "" : "s"} apart`, 0.03, 0.36, 0.3, kit.palette.ink, true);
    write(`angle between them: ${gap} × ${aboutTheta.toFixed(2)} = ${a.toFixed(2)} radians, score q · k = cos(angle) = ${Math.cos(a).toFixed(2)}`, 0.03, 0.64, 0.2, kit.palette.ink);
    write("move both along: each arm turns further, the angle and the score stay put", 0.03, 0.9, 0.2, kit.palette.ink);
  });
  readout.group.position.set(0.5, 0, 0.1);
  root.add(readout.group);

  const railZ = 0.24;
  const railX = [0.2, 0.8];
  const rail = kit.solid(new THREE.BoxGeometry(railX[1] - railX[0] + 0.04, 0.016, 0.02));
  rail.position.set(0.5, 0.008, railZ);
  const ticks: number[] = [];
  for (let g = 0; g <= MOST; g++) {
    const x = railX[0] + (g / MOST) * (railX[1] - railX[0]);
    ticks.push(x, 0.017, railZ - 0.018, x, 0.017, railZ + 0.018);
  }
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.028, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(railX[0] + (START / MOST) * (railX[1] - railX[0]), 0.045, railZ);
  const railCard = createCard(kit, 0.6, 0.045, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("gap m − n, from 0 to 7 positions", 0.02, 0.7, 0.5, kit.palette.faint);
  });
  railCard.group.position.set(0.5, 0, railZ + 0.055);
  const note = createCard(kit, 0.86, 0.04, (ctx, w, h) => {
    pen(ctx, fonts, w, h)(`illustrative: one pair at pair ${ABOUT_PAIR}'s real rate, θ = ${aboutTheta.toFixed(2)}; a head turns ${PAIRS} pairs, each at its own rate`, 0.03, 0.7, 0.5, kit.palette.faint);
  });
  note.group.position.set(0.5, 0, 0.375);
  root.add(rail, inkSegments(ticks, kit.thin), knob, railCard.group, note.group);

  let held = false;
  let knobX = knob.position.x;
  let shown = START;
  let wedgeGap = -1;
  let printed = "";
  let time = 0;
  const span = () => SLOTS - 1 - gap;
  const tri = (k: number) => {
    const s = span();
    const j = k % (2 * s);
    return j <= s ? j : 2 * s - j;
  };

  return {
    exhibits: [root],
    handles: [knob],
    dragStart: () => {
      held = true;
    },
    drag: (x) => {
      knobX = Math.min(railX[1], Math.max(railX[0], x));
      const g = Math.round(((knobX - railX[0]) / (railX[1] - railX[0])) * MOST);
      if (g !== gap) {
        gap = g;
        shift = Math.min(shift, span());
        time = Math.round(shift) * STEP;
      }
    },
    dragEnd: () => {
      held = false;
    },
    enter: () => {
      time = 0;
    },
    update: (dt, _exhibit, _stop, reduced) => {
      const goal = railX[0] + (gap / MOST) * (railX[1] - railX[0]);
      knob.position.x = held ? knobX : approach(knob.position.x, goal, dt * 0.7, reduced);
      shown = approach(shown, gap, dt * 0.45, reduced);
      if (reduced) {
        shift = 2;
      } else {
        time += dt;
        const k = Math.floor(time / STEP);
        const phase = time - k * STEP;
        const a = tri(k);
        const b = tri(k + 1);
        shift = a + (b - a) * smooth(0.3, 1.5, phase);
      }
      const n = shift;
      const m = shift + shown;
      kMark.position.x = slotX(n);
      qMark.position.x = slotX(m);
      qArm.rotation.y = m * aboutTheta;
      mqArm.rotation.y = m * aboutTheta;
      kArm.rotation.y = n * aboutTheta;
      mkArm.rotation.y = n * aboutTheta;
      wedgeHolder.rotation.y = n * aboutTheta;
      if (Math.abs(shown - wedgeGap) > 0.002) {
        wedgeGap = shown;
        wedge.geometry.dispose();
        wedge.geometry = new THREE.CircleGeometry(BIG * 0.8, 48, 0, Math.max(0.001, shown * aboutTheta));
      }
      const text = `${gap} ${Math.round(shift)}`;
      if (text !== printed) {
        printed = text;
        readout.redraw();
      }
    },
  };
}
