import * as THREE from "three";
import { aboutContext, drawFrom } from "@/components/kernels/sampling/about";
import { mulberry } from "@/components/kernels/sampling/figures";
import type { BookModel } from "../book-scene";
import { createCard, inkSegments, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";

const SHOWN = aboutContext.tokens.length;
const COLS = SHOWN + 1;
const LEFT = 0.25;
const PITCH = 0.078;
const WIDTH = 0.05;
const LOGIT_Z = -0.35;
const PROB_Z = -0.2;
const LOGIT_H = 0.011;
const PROB_H = 0.3;
const T_MAX = 2;
const T_STEP = 0.05;
const HOP = 2.6;
const MOVE = 1.1;

const colX = (i: number) => LEFT + i * PITCH;
const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const approach = (value: number, goal: number, dt: number, reduced: boolean) => (reduced ? goal : value + (goal - value) * Math.min(1, dt * 4));
const pct = (v: number) => (v >= 0.995 ? "100%" : v < 0.005 ? "0%" : `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`);
const fmt = (v: number) => v.toLocaleString("en-US");
const nameOf = (i: number) => (i < SHOWN ? aboutContext.tokens[i] : "others");

type Pen = (text: string, x: number, y: number, size: number, color: string, serif?: boolean, align?: CanvasTextAlign) => void;
const pen =
  (ctx: CanvasRenderingContext2D, fonts: Fonts, w: number, h: number): Pen =>
  (text, x, y, size, color, serif = false, align = "left") => {
    ctx.fillStyle = color;
    ctx.textAlign = align;
    ctx.font = `${h * size}px ${serif ? fonts.serif : fonts.sans}`;
    ctx.fillText(text, w * x, h * y);
    ctx.textAlign = "left";
  };

function column(kit: Kit, material?: THREE.Material) {
  const pivot = new THREE.Group();
  const body = kit.solid(new THREE.BoxGeometry(WIDTH, 1, WIDTH), { material });
  body.position.y = 0.5;
  pivot.add(body);
  return pivot;
}

export function createSamplingAboutModel(kit: Kit, fonts: Fonts): BookModel {
  const root = new THREE.Group();
  const wash = washPlane(kit.palette.blue, 1, 0.42, 0.12);
  wash.position.set(0.52, 0.002, -0.28);
  root.add(wash);

  const title = createCard(kit, 0.86, 0.11, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write(`sampling: ${fmt(aboutContext.vocab)} logits in, one token id out`, 0.03, 0.34, 0.24, kit.palette.ink, true);
    write(`p = softmax(logit / T): each e^(logit / T) divided by their sum over all ${fmt(aboutContext.vocab)}`, 0.03, 0.62, 0.17, kit.palette.ink);
    write("greedy takes the largest logit; sampling draws at random with p", 0.03, 0.88, 0.14, kit.palette.faint);
  });
  title.group.position.set(0.5, 0, -0.57);
  root.add(title.group);

  aboutContext.logits.forEach((l, i) => {
    const c = column(kit);
    c.scale.y = l * LOGIT_H;
    c.position.set(colX(i), 0, LOGIT_Z);
    root.add(c);
  });
  const dots: number[] = [];
  for (let d = 0; d < 3; d++) dots.push(colX(SHOWN) - 0.02 + d * 0.02, 0.004, LOGIT_Z, colX(SHOWN) - 0.016 + d * 0.02, 0.004, LOGIT_Z);
  root.add(inkSegments(dots, kit.line));

  const probs = Array.from({ length: COLS }, (_, i) => {
    const base = new THREE.Mesh(new THREE.BoxGeometry(WIDTH, 0.003, WIDTH), kit.ink);
    base.position.set(colX(i), 0.0015, PROB_Z);
    const fill = column(kit, kit.blue);
    fill.position.set(colX(i), 0, PROB_Z);
    root.add(base, fill);
    return { fill, height: 0 };
  });

  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.019, 24, 18), kit.pink);
  ball.castShadow = true;
  root.add(ball);

  let temperature = 1;
  let odds = aboutContext.odds(temperature);
  const table = createCard(kit, 0.9, 0.1, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    const at = (i: number) => (colX(i) - 0.05) / 0.9;
    write("token", 0.015, 0.3, 0.2, kit.palette.faint);
    write("logit", 0.015, 0.58, 0.2, kit.palette.faint);
    write(`p at T = ${temperature.toFixed(2)}`, 0.015, 0.86, 0.2, kit.palette.blue);
    for (let i = 0; i < COLS; i++) {
      write(i < SHOWN ? aboutContext.tokens[i] : `${fmt(aboutContext.others)} others`, at(i), 0.3, i < SHOWN ? 0.22 : 0.17, kit.palette.ink, true, "center");
      write(i < SHOWN ? aboutContext.logits[i].toFixed(2) : "lower", at(i), 0.58, 0.19, kit.palette.ink, false, "center");
      write(pct(odds[i]), at(i), 0.86, 0.19, kit.palette.blue, false, "center");
    }
  });
  table.group.position.set(0.5, 0, PROB_Z + 0.12);
  root.add(table.group);

  const rand = mulberry(17);
  let pick = 0;
  let from = 0;
  const history: number[] = [];
  const readout = createCard(kit, 0.86, 0.1, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    const mode = temperature === 0 ? "T = 0: greedy, the largest logit wins" : temperature < 1 ? "T below 1: the gaps grow, the odds sharpen" : temperature > 1 ? "T above 1: the gaps shrink, the odds flatten" : "T = 1: the model's own odds";
    write(mode, 0.03, 0.34, 0.27, kit.palette.ink, true);
    write(`greedy: ${aboutContext.tokens[0]}, ${aboutContext.tokens[0]}, ${aboutContext.tokens[0]}, ${aboutContext.tokens[0]} ... every time`, 0.03, 0.63, 0.19, kit.palette.ink);
    write(`draws: ${history.length ? history.map(nameOf).join(", ") : "..."}`, 0.03, 0.9, 0.19, kit.palette.pink);
  });
  readout.group.position.set(0.5, 0, 0.07);
  root.add(readout.group);

  const railZ = 0.22;
  const railX = [0.2, 0.8];
  const toX = (t: number) => railX[0] + (t / T_MAX) * (railX[1] - railX[0]);
  const rail = kit.solid(new THREE.BoxGeometry(railX[1] - railX[0] + 0.04, 0.016, 0.02));
  rail.position.set(0.5, 0.008, railZ);
  const ticks: number[] = [];
  for (let t = 0; t <= T_MAX; t += 0.5) ticks.push(toX(t), 0.017, railZ - 0.018, toX(t), 0.017, railZ + 0.018);
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.028, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(toX(temperature), 0.045, railZ);
  const railCard = createCard(kit, 0.86, 0.045, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    const at = (t: number) => (toX(t) - 0.07) / 0.86;
    write("T = 0 greedy", at(0), 0.7, 0.46, kit.palette.faint, false, "center");
    write("0.5 sharper", at(0.5), 0.7, 0.46, kit.palette.faint, false, "center");
    write("1 as trained", at(1), 0.7, 0.46, kit.palette.blue, false, "center");
    write("1.5 flatter", at(1.5), 0.7, 0.46, kit.palette.faint, false, "center");
    write("2", at(2), 0.7, 0.46, kit.palette.faint, false, "center");
  });
  railCard.group.position.set(0.5, 0, railZ + 0.06);
  const note = createCard(kit, 0.86, 0.04, (ctx, w, h) => {
    pen(ctx, fonts, w, h)(`TinyLlama's real logits after “${aboutContext.answer}”; heights to scale, draws simulated`, 0.03, 0.7, 0.46, kit.palette.faint);
  });
  note.group.position.set(0.5, 0, 0.38);
  root.add(rail, inkSegments(ticks, kit.thin), knob, railCard.group, note.group);

  const setTemperature = (t: number) => {
    if (t === temperature) return;
    temperature = t;
    odds = aboutContext.odds(t);
    table.redraw();
    readout.redraw();
  };

  const nextDraw = () => {
    from = pick;
    pick = temperature === 0 ? 0 : drawFrom(odds, rand());
    history.push(pick);
    if (history.length > 6) history.shift();
    readout.redraw();
  };

  let held = false;
  let knobX = toX(temperature);
  let time = 0;
  return {
    exhibits: [root],
    handles: [knob],
    dragStart: () => {
      held = true;
    },
    drag: (x) => {
      knobX = Math.min(railX[1], Math.max(railX[0], x));
      setTemperature(Math.round(((knobX - railX[0]) / (railX[1] - railX[0])) * (T_MAX / T_STEP)) * T_STEP);
    },
    dragEnd: () => {
      held = false;
    },
    enter: () => {
      time = 0;
      if (history.length === 0) nextDraw();
    },
    update: (dt, _exhibit, _stop, reduced) => {
      knob.position.x = held ? knobX : approach(knob.position.x, toX(temperature), dt * 0.7, reduced);
      probs.forEach((p, i) => {
        p.height = approach(p.height, odds[i] * PROB_H, dt * 0.45, reduced);
        p.fill.scale.y = Math.max(0.0001, p.height);
        p.fill.visible = p.height > 0.001;
      });
      if (reduced) {
        if (history.length === 0) nextDraw();
        from = pick;
        time = HOP - 0.01;
      } else {
        time += dt;
        if (time >= HOP) {
          time -= HOP;
          nextDraw();
        }
      }
      const t = reduced ? 1 : smooth(0, MOVE, time);
      const x = colX(from) + (colX(pick) - colX(from)) * t;
      const top = (h: number) => Math.max(0.003, h) + 0.022;
      const y = top(probs[from].height) + (top(probs[pick].height) - top(probs[from].height)) * t + Math.sin(Math.PI * t) * 0.08;
      ball.position.set(x, y, PROB_Z);
    },
  };
}
