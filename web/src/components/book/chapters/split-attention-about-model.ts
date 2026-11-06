import * as THREE from "three";
import { splitAboutSizes } from "@/components/kernels/split-attention/about";
import type { BookModel } from "../book-scene";
import { createCard, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";

const { context, tiles: TILES, drawnSms: SMS, heads, sms: realSms } = splitAboutSizes;
const CHOICES = [1, 2, 4, 8];
const PITCH = 0.03;
const TILE = 0.026;
const GAP = 0.01;
const STRIP_X = 0.1;
const STRIP_Z = -0.4;
const SM_Z = -0.2;
const MERGE_X = 0.82;
const OUT_Z = -0.4;
const LIFT = 0.055;
const LAUNCH = 0.8;
const WALK = 4;
const RISE = 0.8;
const TRAVEL = 1.4;
const GROW = 0.9;
const HOLD = 1.2;
const FADE = 0.8;

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const approach = (value: number, goal: number, dt: number, reduced: boolean) => (reduced ? goal : value + (goal - value) * Math.min(1, dt * 4));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const fmt = (n: number) => n.toLocaleString("en-US");

type Pen = (text: string, x: number, y: number, size: number, color: string, serif?: boolean) => void;
const pen =
  (ctx: CanvasRenderingContext2D, fonts: Fonts, w: number, h: number): Pen =>
  (text, x, y, size, color, serif = false) => {
    ctx.fillStyle = color;
    ctx.font = `${h * size}px ${serif ? fonts.serif : fonts.sans}`;
    ctx.fillText(text, w * x, h * y);
  };

const smCenter = (j: number) => STRIP_X + PITCH * (2 * j + 1);
const smFor = (i: number, s: number) => i * (SMS / s) + Math.floor((SMS / s - 1) / 2);
const rangeOf = (k: number, s: number) => Math.floor((k * s) / TILES);
const tileX = (k: number, s: number) => STRIP_X + PITCH * k + PITCH / 2 + rangeOf(k, s) * GAP - ((s - 1) * GAP) / 2;

export function createSplitAboutModel(kit: Kit, fonts: Fonts): BookModel {
  const root = new THREE.Group();
  const wash = washPlane(kit.palette.blue, 1, 0.5, 0.12);
  wash.position.set(0.5, 0.002, -0.3);
  root.add(wash);

  const title = createCard(kit, 0.86, 0.11, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("SM: Streaming Multiprocessor, one of the GPU's processors", 0.03, 0.34, 0.22, kit.palette.ink, true);
    write("thread block: a group of threads that runs on one SM", 0.03, 0.64, 0.22, kit.palette.ink, true);
    write("splitting: one head's context cut into pieces, one thread block per piece", 0.03, 0.9, 0.13, kit.palette.faint);
  });
  title.group.position.set(0.5, 0, -0.57);
  root.add(title.group);

  const tileMeshes = Array.from({ length: TILES }, () => {
    const t = kit.solid(new THREE.BoxGeometry(TILE, 0.012, 0.04));
    t.position.set(0, 0.006, STRIP_Z);
    root.add(t);
    return t;
  });

  const bars = Array.from({ length: SMS }, () => {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(1, 0.006, 0.022), kit.blue);
    bar.position.set(0, 0.016, STRIP_Z);
    bar.visible = false;
    root.add(bar);
    return bar;
  });

  for (let j = 0; j < SMS; j++) {
    const plate = kit.solid(new THREE.BoxGeometry(0.052, 0.01, 0.09));
    plate.position.set(smCenter(j), 0.005, SM_Z);
    root.add(plate);
  }

  const blocks = Array.from({ length: SMS }, () => {
    const pivot = new THREE.Group();
    const body = kit.solid(new THREE.BoxGeometry(0.034, 0.03, 0.044), { material: kit.pink });
    body.position.y = 0.015;
    pivot.add(body);
    pivot.position.set(smCenter(0), 0.01, SM_Z);
    pivot.visible = false;
    root.add(pivot);
    return { pivot, x: smCenter(0) };
  });

  const packets = Array.from({ length: SMS }, () => {
    const group = new THREE.Group();
    const record = new THREE.Mesh(new THREE.BoxGeometry(0.024, 0.028, 0.024), kit.blue);
    record.position.y = 0.014;
    group.add(record);
    group.visible = false;
    root.add(group);
    return group;
  });

  const merge = kit.solid(new THREE.BoxGeometry(0.1, 0.03, 0.07));
  merge.position.set(MERGE_X, 0.015, SM_Z);
  const lid = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.004, 0.07), kit.blue);
  lid.position.set(MERGE_X, 0.032, SM_Z);
  lid.visible = false;
  const outBase = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.003, 0.05), kit.ink);
  outBase.position.set(MERGE_X, 0.0015, OUT_Z);
  const outPivot = new THREE.Group();
  const outBody = kit.solid(new THREE.BoxGeometry(0.05, 0.06, 0.05), { material: kit.blue });
  outBody.position.y = 0.03;
  outPivot.add(outBody);
  outPivot.position.set(MERGE_X, 0, OUT_Z);
  outPivot.scale.y = 0.0001;
  root.add(merge, lid, outBase, outPivot);

  const label = (x: number, z: number, width: number, head: string, sub: string, color: string) => {
    const card = createCard(kit, width, 0.07, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      write(head, 0.03, 0.46, 0.4, color, true);
      write(sub, 0.03, 0.84, 0.26, kit.palette.ink);
    });
    card.group.position.set(x, 0, z);
    root.add(card.group);
  };
  label(0.345, -0.33, 0.55, "one head's context", `${fmt(context)} cached keys and values, drawn as ${TILES} tiles`, kit.palette.ink);
  label(0.81, -0.33, 0.29, "out", "the head's 64 values", kit.palette.blue);
  label(0.32, -0.11, 0.52, `${SMS} SMs`, `drawn of this GPU's ${realSms}; a block sits on the SM running it`, kit.palette.ink);
  label(0.81, -0.11, 0.29, "merge kernel", "adds the partial results", kit.palette.ink);

  let splits = 4;
  const readout = createCard(kit, 0.86, 0.1, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    const many = splits > 1;
    write(`${splits} split${many ? "s" : ""}: ${splits} thread block${many ? "s" : ""}, ${splits} of ${SMS} SMs busy`, 0.03, 0.36, 0.3, kit.palette.pink, true);
    if (many) {
      write(`each block walks ${fmt(context / splits)} of the ${fmt(context)} tokens and writes a partial result:`, 0.03, 0.64, 0.21, kit.palette.ink);
      write("m, its largest score, l, its sum of exponentials, and its weighted values", 0.03, 0.9, 0.21, kit.palette.ink);
    } else {
      write(`one block walks all ${fmt(context)} tokens alone`, 0.03, 0.64, 0.21, kit.palette.ink);
      write("it holds the whole softmax, so it writes the output itself", 0.03, 0.9, 0.21, kit.palette.ink);
    }
  });
  readout.group.position.set(0.5, 0, 0.0);
  root.add(readout.group);

  const railZ = 0.17;
  const railX = [0.2, 0.8];
  const stopX = (i: number) => railX[0] + (i / (CHOICES.length - 1)) * (railX[1] - railX[0]);
  const rail = kit.solid(new THREE.BoxGeometry(railX[1] - railX[0] + 0.04, 0.016, 0.02));
  rail.position.set(0.5, 0.008, railZ);
  const ticks: number[] = [];
  CHOICES.forEach((_, i) => ticks.push(stopX(i), 0.017, railZ - 0.018, stopX(i), 0.017, railZ + 0.018));
  const tickGeometry = new THREE.BufferGeometry();
  tickGeometry.setAttribute("position", new THREE.Float32BufferAttribute(ticks, 3));
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.028, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(stopX(CHOICES.indexOf(splits)), 0.045, railZ);
  const railCard = createCard(kit, 0.78, 0.045, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    ctx.textAlign = "center";
    CHOICES.forEach((c, i) => write(`${c} split${c > 1 ? "s" : ""}`, (stopX(i) - 0.11) / 0.78, 0.7, 0.5, kit.palette.faint));
    ctx.textAlign = "left";
  });
  railCard.group.position.set(0.5, 0, railZ + 0.06);
  const note = createCard(kit, 0.86, 0.04, (ctx, w, h) => {
    pen(ctx, fonts, w, h)(`illustrative: one head of ${heads}, ${TILES} tiles for ${fmt(context)} tokens, ${SMS} SMs of ${realSms}; walk times not to scale`, 0.03, 0.7, 0.5, kit.palette.faint);
  });
  note.group.position.set(0.5, 0, 0.36);
  root.add(rail, new THREE.LineSegments(tickGeometry, new THREE.LineBasicMaterial({ color: kit.palette.ink })), knob, railCard.group, note.group);

  const timeline = () => {
    const walkEnd = LAUNCH + WALK / splits;
    const riseEnd = walkEnd + RISE;
    const landed = riseEnd + TRAVEL + 0.08 * (splits - 1);
    const grown = landed + GROW;
    return { walkEnd, riseEnd, landed, grown, cycle: grown + HOLD + FADE };
  };

  let time = 0;
  const setSplits = (s: number) => {
    if (s === splits) return;
    splits = s;
    time = 0;
    readout.redraw();
  };

  let held = false;
  let knobX = knob.position.x;
  return {
    exhibits: [root],
    handles: [knob],
    dragStart: () => {
      held = true;
    },
    drag: (x) => {
      knobX = Math.min(railX[1], Math.max(railX[0], x));
      setSplits(CHOICES[Math.round(((knobX - railX[0]) / (railX[1] - railX[0])) * (CHOICES.length - 1))]);
    },
    dragEnd: () => {
      held = false;
    },
    enter: () => {
      time = 0;
    },
    update: (dt, _exhibit, _stop, reduced) => {
      const goal = stopX(CHOICES.indexOf(splits));
      knob.position.x = held ? knobX : approach(knob.position.x, goal, dt * 0.7, reduced);
      const tl = timeline();
      time = reduced ? tl.grown + 0.1 : (time + dt) % tl.cycle;
      const fade = reduced ? 1 : 1 - smooth(tl.cycle - FADE, tl.cycle, time);

      tileMeshes.forEach((t, k) => {
        t.position.x = approach(t.position.x, tileX(k, splits), dt * 0.6, reduced);
      });

      const up = (reduced ? 1 : smooth(0, LAUNCH, time)) * fade;
      blocks.forEach((b, i) => {
        const on = i < splits;
        b.x = approach(b.x, smCenter(smFor(Math.min(i, splits - 1), splits)), dt * 0.6, reduced);
        b.pivot.position.x = b.x;
        b.pivot.scale.y = Math.max(0.0001, on ? up : 0);
        b.pivot.visible = on && up > 0.01;
      });

      const per = TILES / splits;
      bars.forEach((bar, i) => {
        if (i >= splits) {
          bar.visible = false;
          return;
        }
        const first = tileX(i * per, splits) - TILE / 2;
        const last = tileX((i + 1) * per - 1, splits) + TILE / 2;
        const p = (reduced ? 1 : smooth(LAUNCH, tl.walkEnd, time)) * fade;
        const len = (last - first) * p;
        bar.visible = len > 0.002;
        bar.scale.x = Math.max(0.0001, len);
        bar.position.x = first + len / 2;
      });

      const target = splits === 1 ? { z: OUT_Z, y: 0.003 } : { z: SM_Z, y: 0.034 };
      let landedAny = false;
      packets.forEach((p, i) => {
        if (i >= splits) {
          p.visible = false;
          return;
        }
        const bx = blocks[i].x;
        if (reduced) {
          p.visible = true;
          p.position.set(bx, 0.04, SM_Z);
          return;
        }
        const start = tl.riseEnd + 0.08 * i;
        const rise = smooth(tl.walkEnd, tl.riseEnd, time);
        const travel = smooth(start, start + TRAVEL * 0.7, time);
        const down = smooth(start + TRAVEL * 0.7, start + TRAVEL, time);
        p.visible = time > tl.walkEnd && down < 0.98 && fade > 0.99;
        if (down >= 0.98) landedAny = true;
        p.position.set(lerp(bx, MERGE_X, travel), lerp(lerp(0.04, LIFT, rise), target.y, down), lerp(SM_Z, target.z, travel));
      });

      lid.visible = splits > 1 && fade > 0.05 && (reduced || landedAny);
      const grow = (reduced ? 1 : smooth(tl.landed, tl.grown, time)) * fade;
      outPivot.scale.y = Math.max(0.0001, grow);
      outPivot.visible = grow > 0.01;
    },
  };
}
