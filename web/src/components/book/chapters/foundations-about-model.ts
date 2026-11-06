import * as THREE from "three";
import { aboutNumbers } from "@/components/kernels/foundations/about";
import type { BookModel } from "../book-scene";
import { createCard, inkSegments, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";

const n = aboutNumbers;
const fmt = (v: number) => v.toLocaleString("en-US");
const TOKENS = n.tokens.length;
const HEAD = (n.kernels - n.layers * n.perLayer) / 2;

const PIPE_Z = -0.22;
const PIPE_D = 0.18;
const EMB = [0.07, 0.17];
const EMB_H = 0.03;
const PLATE_X0 = 0.225;
const PLATE_PITCH = 0.016;
const PLATE_W = 0.008;
const PLATE_H = 0.08;
const PLATE_D = 0.15;
const LIFT = 0.022;
const LM = [0.6, 0.69];
const LM_H = 0.045;
const BARS_X0 = 0.735;
const BAR_PITCH = 0.018;
const BAR_W = 0.012;
const BAR_MAX = 0.11;
const BARS = [0.25, 0.4, 0.2, 0.55, 0.32, 0.18, 1, 0.38, 0.28, 0.45, 0.22, 0.34];
const WINNER = BARS.indexOf(1);
const ROW_Y = PLATE_H + LIFT + 0.014;
const SENT_Z = -0.43;
const SENT_W = 0.86;
const SENT_H = 0.065;

const T = { row: 0.9, travel: 4.6, head: 5.3, bars: 6.1, arc: 7.4, hold: 8.6, cycle: 9.4 };

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const approach = (value: number, goal: number, dt: number, reduced: boolean) => (reduced ? goal : value + (goal - value) * Math.min(1, dt * 4));

type Pen = (text: string, x: number, y: number, size: number, color: string, serif?: boolean, maxW?: number, align?: CanvasTextAlign) => void;
const pen =
  (ctx: CanvasRenderingContext2D, fonts: Fonts, w: number, h: number): Pen =>
  (text, x, y, size, color, serif = false, maxW = 1 - x, align = "left") => {
    let px = h * size;
    const font = () => `${px}px ${serif ? fonts.serif : fonts.sans}`;
    ctx.font = font();
    while (px > 4 && ctx.measureText(text).width > maxW * w) {
      px *= 0.94;
      ctx.font = font();
    }
    ctx.fillStyle = color;
    ctx.textAlign = align;
    ctx.fillText(text, w * x, h * y);
    ctx.textAlign = "left";
  };

const inputOf = (k: number) => (k === 0 ? n.promptLast : n.tokens[k - 1]);

export function createFoundationsAboutModel(kit: Kit, fonts: Fonts): BookModel {
  const root = new THREE.Group();
  const wash = washPlane(kit.palette.blue, 1, 0.42, 0.12);
  wash.position.set(0.5, 0.002, -0.26);
  root.add(wash);

  const title = createCard(kit, 0.86, 0.11, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("an inference engine runs a trained model to write text", 0.03, 0.34, 0.24, kit.palette.ink, true, 0.94);
    write(`one decode step: token id → embedding row → ${n.layers} layers → LM head → ${fmt(n.vocab)} logits → one token`, 0.03, 0.62, 0.15, kit.palette.ink, false, 0.94);
    write(`every piece is GPU kernels: ${n.kernels} per step in the profile, then one to pick the token`, 0.03, 0.88, 0.13, kit.palette.faint, false, 0.94);
  });
  title.group.position.set(0.5, 0, -0.565);
  root.add(title.group);

  let token = 0;
  let produced = false;
  const slots = (() => {
    const widths = n.tokens.map((t) => Math.max(2.2, t.length + 1.4));
    const total = widths.reduce((a, b) => a + b, 0) + 6 + TOKENS * 0.6;
    const unit = 0.86 / total;
    let x = 0.12;
    return widths.map((wd) => {
      const s = { x, w: wd * unit };
      x += s.w + 0.6 * unit;
      return s;
    });
  })();
  const sentence = createCard(kit, SENT_W, SENT_H, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("answer:", 0.015, 0.62, 0.3, kit.palette.faint);
    n.tokens.forEach((t, i) => {
      const s = slots[i];
      const past = i < token || (i === token && produced);
      const current = i === token;
      ctx.lineWidth = h * 0.03;
      ctx.strokeStyle = current ? kit.palette.pink : past ? kit.palette.ink : kit.palette.faint;
      ctx.setLineDash(current && !produced ? [h * 0.08, h * 0.06] : past ? [] : [h * 0.03, h * 0.05]);
      ctx.strokeRect(w * s.x, h * 0.18, w * s.w, h * 0.62);
      ctx.setLineDash([]);
      const label = current && !produced ? "?" : t;
      const color = current ? kit.palette.pink : past ? kit.palette.ink : kit.palette.faint;
      write(label, s.x + s.w / 2, 0.64, 0.36, color, true, s.w * 0.92, "center");
    });
  });
  sentence.group.position.set(0.5, 0, SENT_Z);
  root.add(sentence.group);
  const slotX = (k: number) => 0.5 - SENT_W / 2 + (slots[k].x + slots[k].w / 2) * SENT_W;

  const emb = kit.solid(new THREE.BoxGeometry(EMB[1] - EMB[0], EMB_H, PIPE_D));
  emb.position.set((EMB[0] + EMB[1]) / 2, EMB_H / 2, PIPE_Z);
  const rowLines: number[] = [];
  for (let i = 0; i < 12; i++) {
    const z = PIPE_Z - PIPE_D / 2 + 0.012 + i * ((PIPE_D - 0.024) / 11);
    rowLines.push(EMB[0] + 0.006, EMB_H + 0.001, z, EMB[1] - 0.006, EMB_H + 0.001, z);
  }
  root.add(emb, inkSegments(rowLines, kit.thin));

  const packet = kit.solid(new THREE.BoxGeometry(0.036, 0.016, 0.05), { material: kit.pink });
  packet.position.set((EMB[0] + EMB[1]) / 2, EMB_H, PIPE_Z);
  root.add(packet);

  const plates = Array.from({ length: n.layers }, (_, i) => {
    const pivot = new THREE.Group();
    const body = kit.solid(new THREE.BoxGeometry(PLATE_W, PLATE_H, PLATE_D));
    body.position.y = PLATE_H / 2;
    pivot.add(body);
    pivot.position.set(PLATE_X0 + i * PLATE_PITCH, 0, PIPE_Z);
    root.add(pivot);
    return pivot;
  });
  const plateX = (i: number) => PLATE_X0 + i * PLATE_PITCH;

  const lm = kit.solid(new THREE.BoxGeometry(LM[1] - LM[0], LM_H, PIPE_D));
  lm.position.set((LM[0] + LM[1]) / 2, LM_H / 2, PIPE_Z);
  root.add(lm);

  const bars = BARS.map((v, i) => {
    const pivot = new THREE.Group();
    const body = kit.solid(new THREE.BoxGeometry(BAR_W, BAR_MAX * v, 0.03), { material: i === WINNER ? kit.pink : kit.blue });
    body.position.y = (BAR_MAX * v) / 2;
    pivot.add(body);
    pivot.position.set(BARS_X0 + i * BAR_PITCH, 0, PIPE_Z);
    pivot.scale.y = 0.0001;
    root.add(pivot);
    return pivot;
  });
  const base: number[] = [];
  base.push(BARS_X0 - 0.01, 0.001, PIPE_Z + 0.022, BARS_X0 + (BARS.length - 1) * BAR_PITCH + 0.01, 0.001, PIPE_Z + 0.022);
  root.add(inkSegments(base, kit.thin));

  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.012, 20, 16), kit.pink);
  ball.visible = false;
  root.add(ball);

  const labelZ = -0.035;
  const label = (x0: number, x1: number, head: string, sub: string, color: string) => {
    const width = x1 - x0;
    const card = createCard(kit, width, 0.075, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      write(head, 0.05, 0.48, 0.42, color, true, 0.9);
      write(sub, 0.05, 0.84, 0.24, kit.palette.ink, false, 0.9);
    });
    card.group.position.set((x0 + x1) / 2, 0, labelZ);
    root.add(card.group);
  };
  label(0.05, 0.2, "embedding", `${fmt(n.vocab)} rows × ${fmt(n.hidden)}`, kit.palette.ink);
  label(0.21, 0.58, `${n.layers} layers`, `${n.perLayer} kernels each: attention, then the MLP`, kit.palette.ink);
  label(0.59, 0.715, "LM head", `${fmt(n.hidden)} → ${fmt(n.vocab)}`, kit.palette.ink);
  label(0.725, 0.95, "logits", "one score per token", kit.palette.blue);

  let stage = "";
  let count = "";
  const readout = createCard(kit, 0.86, 0.1, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write(`step ${token + 1} of ${TOKENS}: "${inputOf(token)}" in, "${n.tokens[token]}" out`, 0.03, 0.36, 0.3, kit.palette.pink, true, 0.94);
    write(stage, 0.03, 0.64, 0.21, kit.palette.ink, false, 0.94);
    write(count, 0.03, 0.9, 0.21, kit.palette.ink, false, 0.94);
  });
  readout.group.position.set(0.5, 0, 0.085);
  root.add(readout.group);

  const railZ = 0.235;
  const railX = [0.12, 0.88];
  const tickX = (k: number) => railX[0] + (k / (TOKENS - 1)) * (railX[1] - railX[0]);
  const rail = kit.solid(new THREE.BoxGeometry(railX[1] - railX[0] + 0.04, 0.016, 0.02));
  rail.position.set(0.5, 0.008, railZ);
  const ticks: number[] = [];
  for (let k = 0; k < TOKENS; k++) ticks.push(tickX(k), 0.017, railZ - 0.018, tickX(k), 0.017, railZ + 0.018);
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.028, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(railX[0], 0.045, railZ);
  const railCard = createCard(kit, 0.86, 0.045, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    n.tokens.forEach((t, k) => {
      const x = (tickX(k) - 0.07) / 0.86;
      write(t, x, 0.72, 0.62, k === token ? kit.palette.pink : k < token ? kit.palette.ink : kit.palette.faint, true, 0.095, "center");
    });
  });
  railCard.group.position.set(0.5, 0, railZ + 0.055);
  const note = createCard(kit, 0.86, 0.04, (ctx, w, h) => {
    pen(ctx, fonts, w, h)(`illustrative: ${n.layers} plates for ${n.layers} layers; ${BARS.length} bars stand for ${fmt(n.vocab)} logits, heights made up`, 0.03, 0.7, 0.5, kit.palette.faint, false, 0.94);
  });
  note.group.position.set(0.5, 0, 0.375);
  root.add(rail, inkSegments(ticks, kit.thin), knob, railCard.group, note.group);

  const setToken = (k: number) => {
    if (k === token) return;
    token = k;
    produced = false;
    sentence.redraw();
    railCard.redraw();
    readout.redraw();
  };
  const setProduced = (p: boolean) => {
    if (p === produced) return;
    produced = p;
    sentence.redraw();
  };
  const setText = (s: string, c: string) => {
    if (s === stage && c === count) return;
    stage = s;
    count = c;
    readout.redraw();
  };

  let held = false;
  let touched = false;
  let knobX = railX[0];
  let time = 0;
  return {
    exhibits: [root],
    handles: [knob],
    dragStart: () => {
      held = true;
      touched = true;
    },
    drag: (x) => {
      knobX = Math.min(railX[1], Math.max(railX[0], x));
      const k = Math.round(((knobX - railX[0]) / (railX[1] - railX[0])) * (TOKENS - 1));
      if (k !== token) {
        setToken(k);
        time = 0;
      }
    },
    dragEnd: () => {
      held = false;
    },
    enter: () => {
      time = 0;
    },
    update: (dt, _exhibit, _stop, reduced) => {
      knob.position.x = held ? knobX : approach(knob.position.x, tickX(token), dt * 0.7, reduced);
      if (reduced) time = T.hold - 0.01;
      else {
        time += dt;
        if (time >= T.cycle) {
          time = 0;
          if (!touched && !held) setToken((token + 1) % TOKENS);
        }
      }
      const t = time;
      const rise = smooth(0, T.row, t);
      const go = smooth(T.row, T.travel, t);
      const into = smooth(T.travel, T.head, t);
      const grow = smooth(T.head, T.bars, t) * (1 - smooth(T.hold, T.cycle - 0.1, t));
      const arc = smooth(T.bars, T.arc, t);

      const startX = (EMB[0] + EMB[1]) / 2;
      const endX = plateX(n.layers - 1) + 0.025;
      const px = startX + (endX - startX) * go + ((LM[0] + LM[1]) / 2 - endX) * into;
      packet.position.set(px, EMB_H + (ROW_Y - EMB_H) * rise, PIPE_Z);
      packet.scale.setScalar(Math.max(0.0001, 1 - into));
      packet.visible = !reduced && into < 0.98 && t < T.head;

      let passed = 0;
      plates.forEach((p, i) => {
        const d = Math.abs(px - plateX(i));
        const on = reduced ? 0 : Math.max(0, 1 - d / (PLATE_PITCH * 1.6));
        p.position.y = LIFT * on * on * (3 - 2 * on);
        if (px > plateX(i)) passed = i + 1;
      });

      bars.forEach((b) => {
        b.scale.y = Math.max(0.0001, grow);
        b.visible = grow > 0.01;
      });

      const from = new THREE.Vector3(BARS_X0 + WINNER * BAR_PITCH, BAR_MAX + 0.012, PIPE_Z);
      const to = new THREE.Vector3(slotX(token), 0.03, SENT_Z + 0.01);
      ball.visible = !reduced && arc > 0.01 && arc < 0.99;
      ball.position.lerpVectors(from, to, arc);
      ball.position.y += Math.sin(Math.PI * arc) * 0.14;
      setProduced(reduced || arc >= 0.99);

      const word = `"${n.tokens[token]}"`;
      if (reduced || t >= T.arc) setText(`greedy: the highest logit is picked, and ${word} becomes the input of the next step`, `kernels this step: ${n.kernels}, then 1 to pick the token`);
      else if (t >= T.head) setText(`the LM head turned ${fmt(n.hidden)} numbers into ${fmt(n.vocab)} logits; the highest wins`, `kernels this step: ${n.kernels}, then 1 to pick the token`);
      else if (t >= T.row && passed > 0) setText(`the row is in layer ${passed} of ${n.layers}`, `kernels so far: ${HEAD + n.perLayer * passed} of ${n.kernels}`);
      else setText(`the id of "${inputOf(token)}" picks one row of ${fmt(n.hidden)} numbers from the embedding table`, `kernels so far: 1 of ${n.kernels}`);
    },
  };
}
