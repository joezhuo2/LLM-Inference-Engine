import * as THREE from "three";
import type { BookModel } from "../book-scene";
import { createCard, inkSegments, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";

export type AddRmsnormAboutNumbers = { hidden: string; blocks: number };

const S0 = [0.8, 1.2, 0.5, 1.0, 0.7, 1.1];
const A = [0.3, 0.2, 0.5, 0.25, 0.4, 0.2];
const M = [0.25, 0.4, 0.2, 0.35, 0.3, 0.3];
const W1 = [1.0, 0.9, 1.1, 1.0, 0.95, 1.05];
const W2 = [1.05, 1.0, 0.9, 1.1, 1.0, 0.95];
const rms = (v: number[]) => Math.sqrt(v.reduce((a, x) => a + x * x, 0) / v.length + 1e-5);
const normed = (v: number[], w: number[]) => v.map((x, i) => (w[i] * x) / rms(v));

const N = S0.length;
const BAR = 0.013;
const PITCH = 0.02;
const DEPTH = 0.022;
const KS = 0.036;
const KN = 0.05;
const STREAM_Z = -0.05;
const NORM_Z = -0.22;
const BLOCK_Z = -0.4;
const STATIONS = [0.17, 0.5, 0.84];
const NORMS = [0.3, 0.64];
const PLUS = [0.4, 0.74];
const MIN = 0.5;
const MAX = 4;
const CYCLE = 11;

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const bump = (a: number, b: number, x: number) => smooth(a, a + 0.4, x) * (1 - smooth(b - 0.4, b, x));
const approach = (value: number, goal: number, dt: number, reduced: boolean) => (reduced ? goal : value + (goal - value) * Math.min(1, dt * 3));

type Pen = (text: string, x: number, y: number, size: number, color: string, serif?: boolean) => void;
const pen =
  (ctx: CanvasRenderingContext2D, fonts: Fonts, w: number, h: number): Pen =>
  (text, x, y, size, color, serif = false) => {
    ctx.fillStyle = color;
    ctx.font = `${h * size}px ${serif ? fonts.serif : fonts.sans}`;
    ctx.fillText(text, w * x, h * y);
  };

type Bar = { pivot: THREE.Group };

export function createAddRmsnormAboutModel(kit: Kit, fonts: Fonts, numbers: AddRmsnormAboutNumbers): BookModel {
  const root = new THREE.Group();
  const wash = washPlane(kit.palette.blue, 1, 0.62, 0.12);
  wash.position.set(0.5, 0.002, -0.2);
  root.add(wash);

  const unit = new THREE.BoxGeometry(BAR, 1, DEPTH);
  const bar = (x: number, z: number, material?: THREE.Material): Bar => {
    const pivot = new THREE.Group();
    const body = kit.solid(unit, { material, threshold: 40 });
    body.position.y = 0.5;
    pivot.add(body);
    pivot.position.set(x, 0.006, z);
    pivot.scale.y = 0.0001;
    root.add(pivot);
    return { pivot };
  };
  const setBar = (b: Bar, height: number, under = 0) => {
    b.pivot.scale.y = Math.max(0.0001, height);
    b.pivot.position.y = 0.006 + under;
    b.pivot.visible = height > 0.0008;
  };
  const xOf = (center: number, i: number) => center + (i - (N - 1) / 2) * PITCH;

  const title = createCard(kit, 0.86, 0.11, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("RMSNorm: Root Mean Square normalization", 0.03, 0.34, 0.24, kit.palette.ink, true);
    write("residual add: a block's output added back onto the stream", 0.03, 0.66, 0.22, kit.palette.ink, true);
    write(`a norm before each of TinyLlama's ${numbers.blocks} attention and MLP blocks, an add after each`, 0.03, 0.9, 0.13, kit.palette.faint);
  });
  title.group.position.set(0.5, 0, -0.56);
  root.add(title.group);

  const lane = kit.solid(new THREE.BoxGeometry(0.88, 0.006, 0.03));
  lane.position.set(0.51, 0.003, STREAM_Z);
  root.add(lane);
  const lines: number[] = [0.95, 0.006, STREAM_Z, 0.925, 0.006, STREAM_Z - 0.02, 0.95, 0.006, STREAM_Z, 0.925, 0.006, STREAM_Z + 0.02];

  const blockBoxes = [0, 1].map((b) => {
    const center = (NORMS[b] + PLUS[b]) / 2;
    const box = kit.solid(new THREE.BoxGeometry(0.2, 0.03, 0.07));
    box.position.set(center, 0.015, BLOCK_Z);
    const card = createCard(kit, 0.16, 0.04, (ctx, w, h) => {
      pen(ctx, fonts, w, h)(b === 0 ? "attention" : "MLP", 0.06, 0.72, 0.6, kit.palette.ink, true);
    });
    card.group.position.set(center, 0, BLOCK_Z + 0.05);
    root.add(box, card.group);
    lines.push(NORMS[b], 0.004, NORM_Z - 0.03, NORMS[b], 0.004, BLOCK_Z + 0.035);
    lines.push(PLUS[b], 0.004, BLOCK_Z + 0.035, PLUS[b], 0.004, STREAM_Z - 0.022);
    lines.push(STATIONS[b] + 0.04, 0.004, STREAM_Z - 0.02, NORMS[b] - 0.04, 0.004, NORM_Z + 0.02);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.016, 0.003, 10, 32), kit.ink);
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(PLUS[b], 0.008, STREAM_Z);
    root.add(ring);
    lines.push(PLUS[b] - 0.009, 0.009, STREAM_Z, PLUS[b] + 0.009, 0.009, STREAM_Z, PLUS[b], 0.009, STREAM_Z - 0.009, PLUS[b], 0.009, STREAM_Z + 0.009);
    return box;
  });

  const normBars = NORMS.map((center) => {
    const plate = kit.solid(new THREE.BoxGeometry(N * PITCH + 0.014, 0.006, DEPTH + 0.02));
    plate.position.set(center, 0.003, NORM_Z);
    root.add(plate);
    return S0.map((_, i) => bar(xOf(center, i), NORM_Z, kit.pink));
  });
  NORMS.forEach((center) => {
    const card = createCard(kit, 0.15, 0.05, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      write("RMSNorm", 0.05, 0.48, 0.42, kit.palette.pink, true);
      write("RMS 1, then × w", 0.05, 0.86, 0.26, kit.palette.ink);
    });
    card.group.position.set(center, 0, NORM_Z + 0.07);
    root.add(card.group);
  });

  const stationBars = STATIONS.map((center) => S0.map((_, i) => ({ base: bar(xOf(center, i), STREAM_Z, kit.blue), added: bar(xOf(center, i), STREAM_Z) })));
  root.add(inkSegments(lines, kit.thin));

  let shown = 1;
  const streams = (s: number) => {
    const s0 = S0.map((v) => v * s);
    const s1 = s0.map((v, i) => v + A[i]);
    const s2 = s1.map((v, i) => v + M[i]);
    return [s0, s1, s2];
  };

  let scale = 1;
  const heads = ["stream in", "after attention", "after the MLP"];
  const stationCards = STATIONS.map((center, k) => {
    const card = createCard(kit, 0.2, 0.065, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      write(heads[k], 0.05, 0.42, 0.34, kit.palette.ink, true);
      write(`RMS ${rms(streams(scale)[k]).toFixed(2)}`, 0.05, 0.84, 0.3, kit.palette.blue);
    });
    card.group.position.set(center, 0, STREAM_Z + 0.07);
    root.add(card.group);
    return card;
  });

  const readout = createCard(kit, 0.86, 0.085, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write(`stream arriving × ${scale.toFixed(1)}: each block still reads RMS 1, then × w`, 0.03, 0.4, 0.3, kit.palette.pink, true);
    write("blue is the stream, hatched is what each block added, pink is the normalized copy a block reads", 0.03, 0.82, 0.2, kit.palette.ink);
  });
  readout.group.position.set(0.5, 0, 0.15);
  root.add(readout.group);

  const railZ = 0.25;
  const railX = [0.2, 0.8];
  const rail = kit.solid(new THREE.BoxGeometry(railX[1] - railX[0] + 0.04, 0.016, 0.02));
  rail.position.set(0.5, 0.008, railZ);
  const ticks: number[] = [];
  for (let s = 1; s <= MAX; s++) {
    const x = railX[0] + ((s - MIN) / (MAX - MIN)) * (railX[1] - railX[0]);
    ticks.push(x, 0.017, railZ - 0.018, x, 0.017, railZ + 0.018);
  }
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.028, 28, 22), kit.pink);
  knob.castShadow = true;
  const knobAt = (s: number) => railX[0] + ((s - MIN) / (MAX - MIN)) * (railX[1] - railX[0]);
  knob.position.set(knobAt(scale), 0.045, railZ);
  const railCard = createCard(kit, 0.6, 0.045, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("size of the stream arriving", 0.02, 0.7, 0.5, kit.palette.faint);
    write(`× ${MIN} to × ${MAX}`, 0.66, 0.7, 0.5, kit.palette.blue);
  });
  railCard.group.position.set(0.5, 0, railZ + 0.055);
  const note = createCard(kit, 0.86, 0.04, (ctx, w, h) => {
    pen(ctx, fonts, w, h)(`illustrative: 6 channels with made-up values; TinyLlama's stream has ${numbers.hidden} per token`, 0.03, 0.7, 0.5, kit.palette.faint);
  });
  note.group.position.set(0.5, 0, 0.38);
  root.add(rail, inkSegments(ticks, kit.thin), knob, railCard.group, note.group);

  const setScale = (s: number) => {
    const rounded = Math.round(s * 10) / 10;
    if (rounded === scale) return;
    scale = rounded;
    readout.redraw();
    stationCards.forEach((c) => c.redraw());
  };

  let held = false;
  let knobX = knobAt(scale);
  let time = 0;
  return {
    exhibits: [root],
    handles: [knob],
    dragStart: () => {
      held = true;
    },
    drag: (x) => {
      knobX = Math.min(railX[1], Math.max(railX[0], x));
      setScale(MIN + ((knobX - railX[0]) / (railX[1] - railX[0])) * (MAX - MIN));
    },
    dragEnd: () => {
      held = false;
    },
    enter: () => {
      time = 0;
    },
    update: (dt, _exhibit, _stop, reduced) => {
      knob.position.x = held ? knobX : approach(knob.position.x, knobAt(scale), dt * 0.7, reduced);
      shown = approach(shown, scale, dt * 0.5, reduced);
      time = reduced ? CYCLE - 2 : (time + dt) % CYCLE;
      const fade = reduced ? 1 : 1 - smooth(CYCLE - 1.2, CYCLE - 0.2, time);
      const [s0, s1] = streams(shown);
      const inputs = [s0, s1];
      const added = [A, M];
      const phase = (k: number) => {
        const t0 = 0.3 + k * 3.2;
        return {
          norm: reduced ? 1 : smooth(t0, t0 + 1, time) * fade,
          lift: reduced ? 0 : bump(t0 + 0.9, t0 + 2.1, time),
          base: reduced ? 1 : smooth(t0 + 1.9, t0 + 2.5, time) * fade,
          add: reduced ? 1 : smooth(t0 + 2.3, t0 + 3.1, time) * fade,
        };
      };
      S0.forEach((_, i) => {
        setBar(stationBars[0][i].base, s0[i] * KS);
        setBar(stationBars[0][i].added, 0);
      });
      [0, 1].forEach((b) => {
        const p = phase(b);
        const n = normed(inputs[b], b === 0 ? W1 : W2);
        blockBoxes[b].position.y = 0.015 + 0.03 * p.lift;
        S0.forEach((_, i) => {
          setBar(normBars[b][i], n[i] * KN * p.norm);
          const base = inputs[b][i] * KS * p.base;
          setBar(stationBars[b + 1][i].base, base);
          setBar(stationBars[b + 1][i].added, added[b][i] * KS * p.add * p.base, base);
        });
      });
    },
  };
}
