import * as THREE from "three";
import { chunkedSteps, fmt } from "@/content/kernels/chunked-prefill";
import { createCard, inkSegments, type Kit, washPlane } from "../ink";
import type { BookModel } from "../book-scene";
import type { Fonts } from "../typeset";

export type ChunkedPrefillNumbers = {
  alone: number;
  aloneTokens: number;
  decoders: number;
  prompt: number;
  budgets: number[];
  largestGap: number;
  medianGap: number;
  stallPrompt: number;
  stallDecoding: number;
  chunkBudget: number;
  gapNone: number;
  gapBudget: number;
  ttftNone: number;
  ttftBudget: number;
};

const NAMES = ["A", "B", "C"];

const placed = <T extends THREE.Object3D>(object: T, x: number, y: number, z: number): T => {
  object.position.set(x, y, z);
  return object;
};
const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

function boxOutline(kit: Kit, w: number, h: number, d: number, x: number, y: number, z: number, dashed = false) {
  const hx = w / 2;
  const hz = d / 2;
  const corners = [
    [-hx, -hz],
    [hx, -hz],
    [hx, hz],
    [-hx, hz],
  ];
  const points: number[] = [];
  const edge = (a: number[], b: number[]) => {
    if (!dashed) {
      points.push(...a, ...b);
      return;
    }
    const pieces = Math.max(2, Math.round(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) / 0.016));
    for (let k = 0; k < pieces; k += 2) {
      const s = k / pieces;
      const e = (k + 1) / pieces;
      points.push(a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s, a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e, a[2] + (b[2] - a[2]) * e);
    }
  };
  for (let i = 0; i < 4; i++) {
    const [ax, az] = corners[i];
    const [bx, bz] = corners[(i + 1) % 4];
    edge([ax, 0, az], [bx, 0, bz]);
    edge([ax, h, az], [bx, h, bz]);
    edge([ax, 0, az], [ax, h, az]);
  }
  const lines = inkSegments(points, kit.thin, 0.0006);
  lines.position.set(x, y, z);
  return lines;
}

function stallExhibit(kit: Kit, fonts: Fonts, n: ChunkedPrefillNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.pink, 0.7, 0.5, 0.12), 0.52, 0.002, -0.24));
  const laneZ = [-0.46, -0.36, -0.26, -0.14];
  const x0 = 0.14;
  const x1 = 0.94;
  laneZ.forEach((z, i) => {
    const rail = kit.solid(new THREE.BoxGeometry(x1 - x0, 0.006, 0.07), { edges: true });
    rail.position.set((x0 + x1) / 2, 0.003, z);
    root.add(rail);
    const name = i < 3 ? NAMES[i] : "P";
    const tag = createCard(kit, 0.05, 0.05, (ctx, w, h) => {
      ctx.fillStyle = i < 3 ? kit.palette.blue : kit.palette.pink;
      ctx.font = `${h * 0.7}px ${fonts.serif}`;
      ctx.textAlign = "center";
      ctx.fillText(name, w * 0.5, h * 0.76);
    }, -0.6);
    tag.group.position.set(0.095, 0, z - 0.01);
    root.add(tag.group);
  });

  const cubeW = 0.056;
  const decodeX = [0.19, 0.255, 0.785, 0.85];
  const decodeAt = [0.4, 1.1, 6.6, 7.3];
  const cubes: { mesh: THREE.Group; at: number }[] = [];
  decodeX.forEach((x, s) => {
    for (let i = 0; i < 3; i++) {
      const cube = kit.solid(new THREE.BoxGeometry(cubeW, 0.04, 0.05), { material: kit.blue, threshold: 40 });
      cube.position.set(x, 0.026, laneZ[i]);
      cubes.push({ mesh: cube, at: decodeAt[s] });
      root.add(cube);
    }
  });

  const slabStart = 0.3;
  const slabEnd = 0.74;
  const slab = new THREE.Group();
  slab.position.set(slabStart, 0.006, laneZ[3]);
  const slabBody = kit.solid(new THREE.BoxGeometry(1, 0.06, 0.056), { material: kit.pink, threshold: 40 });
  slabBody.position.set(0.5, 0.03, 0);
  slab.add(slabBody);
  root.add(slab);

  const waits = new THREE.Group();
  for (let i = 0; i < 3; i++) {
    waits.add(boxOutline(kit, slabEnd - slabStart - 0.01, 0.04, 0.05, (slabStart + slabEnd) / 2, 0.006, laneZ[i], true));
  }
  root.add(waits);

  const stepCard = createCard(kit, 0.8, 0.045, (ctx, w, h) => {
    ctx.font = `${h * 0.5}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.faint;
    ctx.fillText("schedule(): every step is all decode or all prefill, prefill first", w * 0.01, h * 0.68);
  }, -0.5);
  stepCard.group.position.set(0.54, 0, -0.6);
  const waitCard = createCard(kit, 0.8, 0.05, (ctx, w, h) => {
    ctx.font = `${h * 0.5}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.pink;
    ctx.fillText("one step holds the whole prompt;", w * 0.01, h * 0.68);
    ctx.fillStyle = kit.palette.ink;
    ctx.fillText("A, B and C wait for it", w * 0.56, h * 0.68);
  }, -0.5);
  waitCard.group.position.set(0.54, 0, -0.06);
  const aloneCard = createCard(kit, 0.38, 0.1, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.2}px ${fonts.sans}`;
    ctx.fillText(`a ${fmt(n.aloneTokens)}-token prompt, alone`, w * 0.04, h * 0.3);
    ctx.fillStyle = kit.palette.pink;
    ctx.font = `${h * 0.48}px ${fonts.serif}`;
    ctx.fillText(`${fmt(n.alone)} ms`, w * 0.04, h * 0.82);
  });
  aloneCard.group.position.set(0.3, 0, 0.1);
  const gapCard = createCard(kit, 0.38, 0.1, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.2}px ${fonts.sans}`;
    ctx.fillText("longest wait between two tokens", w * 0.04, h * 0.3);
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.48}px ${fonts.serif}`;
    ctx.fillText(`${fmt(n.largestGap)} ms`, w * 0.04, h * 0.82);
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.2}px ${fonts.sans}`;
    ctx.fillText(`median ${fmt(n.medianGap)} ms`, w * 0.56, h * 0.82);
  });
  gapCard.group.position.set(0.72, 0, 0.1);
  const noteCard = createCard(kit, 0.8, 0.04, (ctx, w, h) => {
    ctx.font = `${h * 0.5}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.faint;
    ctx.fillText("4 requests per second, Poisson arrivals; step widths are drawn, not measured", w * 0.01, h * 0.68);
  }, -0.6);
  noteCard.group.position.set(0.52, 0, 0.28);
  root.add(stepCard.group, waitCard.group, aloneCard.group, gapCard.group, noteCard.group);

  let time = 0;
  const loop = 10.5;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 9 : (time + dt) % loop;
      for (const { mesh, at } of cubes) {
        const rise = smooth(at, at + 0.8, time);
        mesh.scale.y = Math.max(0.001, rise);
        mesh.visible = rise > 0.001;
      }
      const grow = smooth(2.0, 6.0, time);
      slab.scale.x = Math.max(0.001, grow * (slabEnd - slabStart));
      slab.visible = grow > 0.001;
      const waiting = smooth(1.9, 2.7, time) * (1 - smooth(6.1, 6.6, time));
      waits.visible = waiting > 0.02;
      waits.scale.y = Math.max(0.001, waiting);
    },
  };
}

function budgetExhibit(kit: Kit, fonts: Fonts, n: ChunkedPrefillNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 0.9, 0.55, 0.12), 0.5, 0.002, -0.26));
  const budget = n.budgets[0];
  const chunk = budget - n.decoders;
  const slot = 0.8 / budget;
  const x0 = 0.12;
  const trayZ = [-0.44, -0.16];
  const drops: { mesh: THREE.Group; at: number }[] = [];
  trayZ.forEach((z, s) => {
    const tray = kit.solid(new THREE.BoxGeometry(0.8 + 0.012, 0.008, 0.08));
    tray.position.set(x0 + 0.4, 0.004, z);
    root.add(tray);
    const walls: number[] = [];
    for (let i = 0; i <= budget; i++) {
      const x = x0 + i * slot;
      walls.push(x, 0.009, z - 0.04, x, 0.009, z + 0.04);
    }
    root.add(inkSegments(walls, kit.thin, 0.0004));
    const base = s * 5.2;
    for (let i = 0; i < budget; i++) {
      const decode = i < n.decoders;
      const cube = kit.solid(new THREE.BoxGeometry(slot - 0.008, 0.036, 0.056), { material: decode ? kit.blue : kit.pink, threshold: 40 });
      cube.position.set(x0 + (i + 0.5) * slot, 0.026, z);
      drops.push({ mesh: cube, at: base + (decode ? 0.4 + i * 0.35 : 1.8 + (i - n.decoders) * 0.12) });
      root.add(cube);
    }
    const first = s * chunk;
    const card = createCard(kit, 0.8, 0.05, (ctx, w, h) => {
      ctx.font = `${h * 0.5}px ${fonts.sans}`;
      NAMES.forEach((name, i) => {
        ctx.fillStyle = kit.palette.blue;
        ctx.textAlign = "center";
        ctx.fillText(name, (i + 0.5) * (w / budget), h * 0.68);
      });
      ctx.textAlign = "left";
      ctx.fillStyle = kit.palette.pink;
      ctx.fillText(`prompt tokens ${first} to ${first + chunk - 1}`, n.decoders * (w / budget) + w * 0.01, h * 0.68);
      ctx.fillStyle = kit.palette.faint;
      ctx.textAlign = "right";
      ctx.fillText(`step ${s + 1}`, w * 0.99, h * 0.68);
    }, -0.55);
    card.group.position.set(x0 + 0.4, 0, z + 0.06);
    root.add(card.group);
  });

  const titleCard = createCard(kit, 0.8, 0.045, (ctx, w, h) => {
    ctx.font = `${h * 0.5}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.ink;
    ctx.fillText(`token budget ${budget} = ${budget} rows in one step`, w * 0.01, h * 0.68);
  }, -0.5);
  titleCard.group.position.set(0.52, 0, -0.6);
  const rules = createCard(kit, 0.8, 0.15, (ctx, w, h) => {
    ctx.font = `${h * 0.15}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.blue;
    ctx.fillText("1  every decoding sequence gets its row first", w * 0.02, h * 0.26);
    ctx.fillStyle = kit.palette.pink;
    ctx.fillText("2  the rest goes to the one sequence in prefill", w * 0.02, h * 0.53);
    ctx.fillStyle = kit.palette.ink;
    ctx.fillText("3  a new request is admitted only if budget is left", w * 0.02, h * 0.8);
  });
  rules.group.position.set(0.52, 0, 0.04);
  const note = createCard(kit, 0.8, 0.04, (ctx, w, h) => {
    ctx.font = `${h * 0.5}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.faint;
    ctx.fillText(`an illustrative budget of ${budget}; step 2 continues the prompt where step 1 stopped`, w * 0.01, h * 0.68);
  }, -0.6);
  note.group.position.set(0.52, 0, 0.28);
  root.add(titleCard.group, rules.group, note.group);

  let time = 0;
  const loop = 12;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 11 : (time + dt) % loop;
      const out = 1 - smooth(11.2, 11.9, time);
      for (const { mesh, at } of drops) {
        const fall = smooth(at, at + 0.7, time);
        mesh.position.y = 0.026 + (1 - fall) * 0.14;
        mesh.visible = fall > 0.001 && out > 0.001;
        mesh.scale.setScalar(Math.max(0.001, out));
      }
    },
  };
}

function spreadExhibit(kit: Kit, fonts: Fonts, n: ChunkedPrefillNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.pink, 0.9, 0.5, 0.12), 0.5, 0.002, -0.3));
  const minB = n.budgets[0];
  const maxB = n.budgets[n.budgets.length - 1];
  const most = chunkedSteps(minB, n.decoders, n.prompt).length;
  const x0 = 0.12;
  const pitch = 0.8 / most;
  const colW = pitch * 0.72;
  const colZ = -0.4;
  const unit = 0.24 / maxB;
  const floor = kit.solid(new THREE.BoxGeometry(0.82, 0.006, 0.1));
  floor.position.set(x0 + 0.4, 0.003, colZ);
  root.add(floor);
  const columns = Array.from({ length: most }, (_, k) => {
    const decode = kit.solid(new THREE.BoxGeometry(colW, 1, 0.07), { material: kit.blue, threshold: 40 });
    const chunk = kit.solid(new THREE.BoxGeometry(colW, 1, 0.07), { material: kit.pink, threshold: 40 });
    decode.position.x = chunk.position.x = x0 + (k + 0.5) * pitch;
    decode.position.z = chunk.position.z = colZ;
    root.add(decode, chunk);
    return { decode, chunk, from: 0, to: 0, now: 0 };
  });
  const ceiling = new THREE.Group();
  const ceilingGlass = new THREE.Mesh(new THREE.PlaneGeometry(0.82, 0.1), kit.glass);
  ceilingGlass.rotation.x = -Math.PI / 2;
  ceiling.add(ceilingGlass, boxOutline(kit, 0.82, 0.0001, 0.1, 0, 0, 0, true));
  ceiling.position.set(x0 + 0.4, 0, colZ);
  root.add(ceiling);

  let budget = n.budgets[0];
  let target = budget;
  let steps = chunkedSteps(budget, n.decoders, n.prompt);
  const numbers = createCard(kit, 0.82, 0.04, (ctx, w, h) => {
    ctx.font = `${h * 0.55}px ${fonts.sans}`;
    ctx.textAlign = "center";
    for (let k = 0; k < steps.length; k++) {
      ctx.fillStyle = k === steps.length - 1 ? kit.palette.pink : kit.palette.faint;
      ctx.fillText(String(k + 1), ((k + 0.5) / most) * (w * (0.8 / 0.82)) + w * (0.01 / 0.82), h * 0.7);
    }
  }, -0.6);
  numbers.group.position.set(x0 + 0.4, 0, colZ + 0.075);
  const readout = createCard(kit, 0.8, 0.1, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.2}px ${fonts.sans}`;
    ctx.fillText(`a ${n.prompt}-token prompt among ${n.decoders} decoders`, w * 0.02, h * 0.3);
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.44}px ${fonts.serif}`;
    ctx.fillText(`budget ${target}: ${steps.length} steps`, w * 0.02, h * 0.8);
    ctx.fillStyle = kit.palette.pink;
    ctx.font = `${h * 0.22}px ${fonts.sans}`;
    ctx.fillText(`first token at step ${steps.length}`, w * 0.6, h * 0.8);
  });
  readout.group.position.set(0.52, 0, -0.22);

  const railZ = 0.0;
  const railX0 = 0.14;
  const railX1 = 0.9;
  const rail = kit.solid(new THREE.BoxGeometry(railX1 - railX0 + 0.04, 0.018, 0.022));
  rail.position.set((railX0 + railX1) / 2, 0.009, railZ);
  const ticks: number[] = [];
  for (const b of n.budgets) {
    const x = railX0 + ((b - minB) / (maxB - minB)) * (railX1 - railX0);
    ticks.push(x, 0.019, railZ - 0.02, x, 0.019, railZ + 0.02);
  }
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.03, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(railX0, 0.05, railZ);
  const railCard = createCard(kit, railX1 - railX0 + 0.04, 0.04, (ctx, w, h) => {
    ctx.font = `${h * 0.55}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.faint;
    n.budgets.forEach((b) => {
      const x = (0.02 + ((b - minB) / (maxB - minB)) * (railX1 - railX0)) / (railX1 - railX0 + 0.04);
      ctx.textAlign = x < 0.1 ? "left" : x > 0.9 ? "right" : "center";
      ctx.fillText(`budget ${b}`, x * w, h * 0.7);
    });
  }, -0.6);
  railCard.group.position.set((railX0 + railX1) / 2, 0, railZ + 0.045);
  const measuredCard = createCard(kit, 0.8, 0.13, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.14}px ${fonts.sans}`;
    ctx.fillText(`measured: a ${fmt(n.stallPrompt)}-token prompt arrives among ${n.stallDecoding} decoding requests`, w * 0.02, h * 0.2);
    ctx.font = `${h * 0.2}px ${fonts.serif}`;
    ctx.fillStyle = kit.palette.ink;
    ctx.fillText("no budget", w * 0.02, h * 0.52);
    ctx.fillText(`budget ${n.chunkBudget}`, w * 0.02, h * 0.84);
    ctx.fillStyle = kit.palette.blue;
    ctx.fillText(`longest gap ${fmt(n.gapNone)} ms`, w * 0.3, h * 0.52);
    ctx.fillText(`longest gap ${fmt(n.gapBudget)} ms`, w * 0.3, h * 0.84);
    ctx.fillStyle = kit.palette.pink;
    ctx.fillText(`first token ${fmt(n.ttftNone)} ms`, w * 0.66, h * 0.52);
    ctx.fillText(`first token ${fmt(n.ttftBudget)} ms`, w * 0.66, h * 0.84);
  });
  measuredCard.group.position.set(0.52, 0, 0.14);
  root.add(numbers.group, readout.group, rail, inkSegments(ticks, kit.line, 0.0004), knob, railCard.group, measuredCard.group);

  let held = false;
  let tween = 1;
  let ceilFrom = budget;
  let ceilNow = budget;
  let instant = false;
  const retarget = () => {
    steps = chunkedSteps(target, n.decoders, n.prompt);
    columns.forEach((c, k) => {
      c.from = c.now;
      c.to = k < steps.length ? n.decoders + steps[k].chunk : 0;
    });
    ceilFrom = ceilNow;
    tween = 0;
    numbers.redraw();
    readout.redraw();
  };
  const apply = (t: number) => {
    const e = t * t * (3 - 2 * t);
    columns.forEach((c) => {
      c.now = c.from + (c.to - c.from) * e;
      const total = c.now;
      const decodeH = Math.min(total, n.decoders) * unit;
      const chunkH = Math.max(0, total - n.decoders) * unit;
      c.decode.visible = total > 0.05;
      c.chunk.visible = chunkH > 0.0005;
      c.decode.scale.y = Math.max(0.0001, decodeH);
      c.decode.position.y = 0.006 + decodeH / 2;
      c.chunk.scale.y = Math.max(0.0001, chunkH);
      c.chunk.position.y = 0.006 + decodeH + chunkH / 2;
    });
    ceilNow = ceilFrom + (target - ceilFrom) * e;
    ceiling.position.y = 0.007 + ceilNow * unit;
  };
  const setBudget = (b: number) => {
    const next = Math.round(Math.min(maxB, Math.max(minB, b)));
    budget = b;
    knob.position.x = railX0 + ((Math.min(maxB, Math.max(minB, b)) - minB) / (maxB - minB)) * (railX1 - railX0);
    if (next !== target) {
      target = next;
      retarget();
    }
  };
  target = -1;
  setBudget(minB);
  return {
    root,
    knob,
    setHeld: (value: boolean) => {
      held = value;
      if (!value) {
        const snap = n.budgets.reduce((a, b) => (Math.abs(b - budget) < Math.abs(a - budget) ? b : a));
        if (Math.abs(snap - budget) < 3) setBudget(snap);
      }
    },
    drag: (x: number) => setBudget(minB + ((x - railX0) / (railX1 - railX0)) * (maxB - minB)),
    enter: () => {
      columns.forEach((c) => {
        c.from = 0;
        c.now = 0;
      });
      ceilFrom = ceilNow;
      tween = 0;
      instant = false;
    },
    update: (dt: number, reduced: boolean) => {
      if (reduced) instant = true;
      tween = instant ? 1 : Math.min(1, tween + dt / (held ? 0.35 : 0.9));
      apply(tween);
    },
  };
}

function preemptExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.pink, 0.9, 0.7, 0.1), 0.5, 0.002, -0.14));
  const runZ = -0.42;
  const runX = [0.2, 0.39, 0.58, 0.77];
  const boxW = 0.15;
  const mk = (material: THREE.Material, w = boxW) => kit.solid(new THREE.BoxGeometry(w, 0.036, 0.08), { material, threshold: 40 });
  NAMES.forEach((_, i) => {
    const box = mk(kit.ink);
    box.position.set(runX[i], 0.018, runZ);
    root.add(box);
    const stripe = new THREE.Mesh(new THREE.BoxGeometry(boxW * 0.7, 0.006, 0.014), kit.blue);
    stripe.position.set(runX[i], 0.039, runZ);
    root.add(stripe);
  });
  const prompt = new THREE.Group();
  const promptBox = mk(kit.ink);
  promptBox.position.y = 0.018;
  const barBack = new THREE.Mesh(new THREE.BoxGeometry(boxW * 0.8, 0.005, 0.016), new THREE.MeshLambertMaterial({ color: kit.palette.faint }));
  barBack.position.set(0, 0.038, 0);
  const bar = new THREE.Mesh(new THREE.BoxGeometry(1, 0.008, 0.02), kit.pink);
  bar.castShadow = true;
  prompt.add(promptBox, barBack, bar);
  root.add(prompt);

  const runCard = createCard(kit, 0.8, 0.045, (ctx, w, h) => {
    ctx.font = `${h * 0.5}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.faint;
    ctx.fillText("running list, in admission order: back is on the right", w * 0.01, h * 0.68);
  }, -0.5);
  runCard.group.position.set(0.5, 0, -0.6);
  const names = createCard(kit, 0.8, 0.045, (ctx, w, h) => {
    ctx.font = `${h * 0.5}px ${fonts.sans}`;
    ctx.textAlign = "center";
    const at = (x: number) => ((x - 0.1) / 0.8) * w;
    NAMES.forEach((name, i) => {
      ctx.fillStyle = kit.palette.blue;
      ctx.fillText(`${name} decoding`, at(runX[i]), h * 0.68);
    });
    ctx.fillStyle = kit.palette.pink;
    ctx.fillText("P in prefill", at(runX[3]), h * 0.68);
  }, -0.55);
  names.group.position.set(0.5, 0, runZ + 0.065);
  root.add(runCard.group, names.group);

  const poolZ = -0.2;
  const owners = [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3];
  const cell = 0.06;
  const poolX = 0.5 - (owners.length * cell) / 2;
  const blue = new THREE.Color(kit.palette.blue);
  const pink = new THREE.Color(kit.palette.pink);
  const paper = new THREE.Color(kit.palette.paper);
  const blocks = owners.map((owner, i) => {
    const material = new THREE.MeshLambertMaterial({ color: owner === 3 ? pink : blue, emissive: owner === 3 ? pink : blue, emissiveIntensity: 0.45 });
    const block = kit.solid(new THREE.BoxGeometry(cell - 0.01, 0.03, 0.05), { material, threshold: 40 });
    block.position.set(poolX + (i + 0.5) * cell, 0.015, poolZ);
    root.add(block);
    return { block, material, owner };
  });
  const poolCard = createCard(kit, 0.8, 0.045, (ctx, w, h) => {
    ctx.font = `${h * 0.5}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.ink;
    ctx.fillText("KV pool: every block held.", w * 0.01, h * 0.68);
    ctx.fillStyle = kit.palette.pink;
    ctx.fillText("P took blocks for its whole prompt at admission", w * 0.37, h * 0.68);
  }, -0.55);
  poolCard.group.position.set(0.5, 0, poolZ + 0.05);
  root.add(poolCard.group);

  const ghost = boxOutline(kit, cell - 0.01, 0.03, 0.05, runX[0] + boxW / 2 + 0.035, 0.0, runZ, true);
  root.add(ghost);
  const needCard = createCard(kit, 0.3, 0.04, (ctx, w, h) => {
    ctx.font = `${h * 0.5}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.ink;
    ctx.fillText("A needs a block, none is free", w * 0.02, h * 0.68);
  }, -0.6);
  needCard.group.position.set(0.26, 0, -0.27);
  root.add(needCard.group);

  const waitZ = 0.02;
  const tray = kit.solid(new THREE.BoxGeometry(0.8, 0.006, 0.11));
  tray.position.set(0.5, 0.003, waitZ);
  root.add(tray, boxOutline(kit, 0.8, 0.05, 0.11, 0.5, 0.003, waitZ, true));
  const other = new THREE.Group();
  const otherBox = mk(kit.ink);
  otherBox.position.y = 0.024;
  other.add(otherBox);
  root.add(other);
  const waitCard = createCard(kit, 0.8, 0.045, (ctx, w, h) => {
    ctx.font = `${h * 0.5}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.faint;
    ctx.fillText("waiting queue, front on the left", w * 0.01, h * 0.68);
  }, -0.55);
  waitCard.group.position.set(0.5, 0, waitZ + 0.07);
  const ruleCard = createCard(kit, 0.8, 0.12, (ctx, w, h) => {
    ctx.font = `${h * 0.19}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.ink;
    ctx.fillText("victim = back of the running list = the sequence in prefill", w * 0.02, h * 0.3);
    ctx.fillStyle = kit.palette.pink;
    ctx.fillText("its blocks are freed; it goes to the front of the queue", w * 0.02, h * 0.58);
    ctx.fillText("and starts again at position 0", w * 0.02, h * 0.86);
  });
  ruleCard.group.position.set(0.5, 0, 0.2);
  const note = createCard(kit, 0.8, 0.035, (ctx, w, h) => {
    ctx.font = `${h * 0.55}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.faint;
    ctx.fillText("the sizes are a sketch: 12 blocks, P 5 of 8 chunks in", w * 0.01, h * 0.7);
  }, -0.6);
  note.group.position.set(0.5, 0, 0.36);
  root.add(waitCard.group, ruleCard.group, note.group);

  const pStart = new THREE.Vector3(runX[3], 0, runZ);
  const pEnd = new THREE.Vector3(0.2, 0.006, waitZ);
  const otherStart = 0.2;
  const otherEnd = 0.39;
  const freed = blocks.filter((b) => b.owner === 3);
  const given = freed[0];
  const givenFrom = given.block.position.clone();
  const givenTo = new THREE.Vector3(runX[0] + boxW / 2 + 0.035, 0.015, runZ);
  let time = 0;
  const loop = 12;
  const color = new THREE.Color();
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 10 : (time + dt) % loop;
      const reset = smooth(11.2, 11.9, time);
      const need = smooth(1.0, 1.8, time) * (1 - smooth(7.4, 8.0, time));
      ghost.visible = need > 0.02;
      ghost.scale.y = Math.max(0.001, need);
      needCard.group.visible = need > 0.02;
      needCard.group.scale.y = Math.max(0.001, need);
      const free = smooth(2.6, 3.6, time) * (1 - reset);
      for (const b of freed) {
        color.copy(pink).lerp(paper, free);
        b.material.color.copy(color);
        b.material.emissiveIntensity = 0.45 * (1 - free);
        b.material.emissive.copy(pink);
      }
      const progress = (5 / 8) * (1 - smooth(2.6, 3.6, time)) + (5 / 8) * reset;
      bar.scale.x = Math.max(0.001, boxW * 0.8 * progress);
      bar.position.x = -boxW * 0.4 + (boxW * 0.8 * progress) / 2;
      const move = smooth(3.8, 5.6, time) * (1 - reset);
      prompt.position.lerpVectors(pStart, pEnd, move);
      prompt.position.y += Math.sin(move * Math.PI) * 0.12;
      other.position.set(otherStart + (otherEnd - otherStart) * smooth(3.8, 5.0, time) * (1 - reset), 0.006, waitZ);
      const give = smooth(6.0, 7.4, time) * (1 - reset);
      given.block.position.lerpVectors(givenFrom, givenTo, give);
      given.block.position.y += Math.sin(give * Math.PI) * 0.1;
      if (give > 0.01) {
        color.copy(paper).lerp(blue, give);
        given.material.color.copy(color);
        given.material.emissive.copy(blue);
        given.material.emissiveIntensity = 0.45 * give;
      }
    },
  };
}

function sampledExhibit(kit: Kit, fonts: Fonts, n: ChunkedPrefillNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 0.9, 0.7, 0.1), 0.5, 0.002, -0.2));
  const rows = [
    { z: -0.48, tray: -0.34, chunk: 13, last: false, at: 0.4 },
    { z: -0.16, tray: -0.02, chunk: 5, last: true, at: 5.4 },
  ];
  const decodeX = (i: number) => 0.14 + i * 0.07;
  const pitch = 0.034;
  const chunkX = 0.36;
  const tokens: { mesh: THREE.Mesh; from: THREE.Vector3; to: THREE.Vector3; at: number }[] = [];
  rows.forEach((row, r) => {
    for (let i = 0; i < n.decoders; i++) {
      const cube = kit.solid(new THREE.BoxGeometry(0.056, 0.036, 0.056), { material: kit.blue, threshold: 40 });
      cube.position.set(decodeX(i), 0.018, row.z);
      root.add(cube);
    }
    for (let i = 0; i < row.chunk; i++) {
      const cube = kit.solid(new THREE.BoxGeometry(pitch - 0.006, 0.03, 0.05), { material: kit.pink, threshold: 40 });
      cube.position.set(chunkX + (i + 0.5) * pitch, 0.015, row.z);
      root.add(cube);
    }
    if (row.last) {
      const wall = kit.solid(new THREE.BoxGeometry(0.006, 0.07, 0.08));
      wall.position.set(chunkX + row.chunk * pitch + 0.008, 0.035, row.z);
      root.add(wall);
    }
    const tray = kit.solid(new THREE.BoxGeometry(0.3, 0.006, 0.07));
    tray.position.set(0.14 + 0.105, 0.003, row.tray);
    root.add(tray);
    const sampled = row.last ? n.decoders + 1 : n.decoders;
    for (let i = 0; i < sampled; i++) {
      const isPrompt = i === n.decoders;
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.02, 20, 14), isPrompt ? kit.pink : kit.blue);
      mesh.castShadow = true;
      const from = isPrompt ? new THREE.Vector3(chunkX + (row.chunk - 0.5) * pitch, 0.05, row.z) : new THREE.Vector3(decodeX(i), 0.05, row.z);
      const to = new THREE.Vector3(decodeX(i), 0.026, row.tray);
      tokens.push({ mesh, from, to, at: row.at + 1.2 + i * 0.4 });
      root.add(mesh);
    }
    if (!row.last) root.add(boxOutline(kit, 0.05, 0.04, 0.05, decodeX(n.decoders), 0.006, row.tray, true));
    const title = createCard(kit, 0.5, 0.04, (ctx, w, h) => {
      ctx.font = `${h * 0.55}px ${fonts.sans}`;
      ctx.fillStyle = kit.palette.faint;
      ctx.fillText(row.last ? `step ${r + 1}: the chunk reaches the end of the prompt` : `step ${r + 1}: the chunk stops short`, w * 0.01, h * 0.7);
    }, -0.5);
    title.group.position.set(0.37, 0, row.z - 0.09);
    const end = row.last
      ? createCard(kit, 0.24, 0.04, (ctx, w, h) => {
          ctx.font = `${h * 0.55}px ${fonts.sans}`;
          ctx.fillStyle = kit.palette.ink;
          ctx.fillText("end of prompt", w * 0.02, h * 0.7);
        }, -0.6)
      : null;
    if (end) {
      end.group.position.set(chunkX + row.chunk * pitch + 0.14, 0, row.z - 0.01);
      root.add(end.group);
    }
    const tag = createCard(kit, 0.5, 0.045, (ctx, w, h) => {
      ctx.font = `${h * 0.5}px ${fonts.sans}`;
      ctx.fillStyle = kit.palette.ink;
      ctx.fillText("sampled, in row order", w * 0.02, h * 0.68);
      ctx.fillStyle = row.last ? kit.palette.pink : kit.palette.faint;
      ctx.fillText(row.last ? "P joins after the decoders" : "P: no entry, no logits", w * 0.5, h * 0.68);
    }, -0.55);
    tag.group.position.set(0.37, 0, row.tray + 0.055);
    root.add(title.group, tag.group);
  });
  const update = createCard(kit, 0.8, 0.12, (ctx, w, h) => {
    ctx.font = `${h * 0.19}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.ink;
    ctx.fillText("update() takes one token per sampled entry, in that order", w * 0.02, h * 0.3);
    ctx.fillStyle = kit.palette.faint;
    ctx.fillText("a list can be empty and the call is still required", w * 0.02, h * 0.58);
    ctx.fillStyle = kit.palette.pink;
    ctx.fillText("first_token is stamped when P's token arrives", w * 0.02, h * 0.86);
  });
  update.group.position.set(0.5, 0, 0.14);
  const note = createCard(kit, 0.8, 0.035, (ctx, w, h) => {
    ctx.font = `${h * 0.55}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.faint;
    ctx.fillText("the sampled list of ChunkedStep; the sizes are a sketch", w * 0.01, h * 0.7);
  }, -0.6);
  note.group.position.set(0.5, 0, 0.32);
  root.add(update.group, note.group);

  let time = 0;
  const loop = 11.5;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 10.5 : (time + dt) % loop;
      const out = 1 - smooth(10.8, 11.4, time);
      for (const t of tokens) {
        const go = smooth(t.at, t.at + 0.9, time);
        t.mesh.visible = time >= t.at && out > 0.001;
        t.mesh.position.lerpVectors(t.from, t.to, go);
        t.mesh.position.y += Math.sin(go * Math.PI) * 0.08;
        t.mesh.scale.setScalar(Math.max(0.001, Math.min(out, smooth(t.at, t.at + 0.3, time))));
      }
    },
  };
}

export function createChunkedPrefillModel(kit: Kit, fonts: Fonts, numbers: ChunkedPrefillNumbers): BookModel {
  const stall = stallExhibit(kit, fonts, numbers);
  const budget = budgetExhibit(kit, fonts, numbers);
  const spread = spreadExhibit(kit, fonts, numbers);
  const preempt = preemptExhibit(kit, fonts);
  const sampled = sampledExhibit(kit, fonts, numbers);
  return {
    exhibits: [stall.root, budget.root, spread.root, preempt.root, sampled.root],
    handles: [spread.knob],
    dragStart: () => spread.setHeld(true),
    drag: (x) => spread.drag(x),
    dragEnd: () => spread.setHeld(false),
    enter: (stop) => {
      if (stop === 2) spread.enter();
    },
    update: (dt, exhibit, _stop, reduced) => {
      if (exhibit === 0) stall.update(dt, reduced);
      if (exhibit === 1) budget.update(dt, reduced);
      if (exhibit === 2) spread.update(dt, reduced);
      if (exhibit === 3) preempt.update(dt, reduced);
      if (exhibit === 4) sampled.update(dt, reduced);
    },
  };
}
