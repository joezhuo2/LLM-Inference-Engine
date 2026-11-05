import * as THREE from "three";
import type { SharedBytes } from "@/components/kernels/attention-v2/figures";
import { createCard, type Kit, liquidMaterial, ribbonTube, washPlane } from "../ink";
import type { BookModel } from "../book-scene";
import type { Fonts } from "../typeset";

export type AttentionV2Numbers = { v1At2048: number; sharedPoints: SharedBytes[]; cap: number; limit: number; scores: number[] };

const placed = <T extends THREE.Object3D>(object: T, x: number, y: number, z: number): T => {
  object.position.set(x, y, z);
  return object;
};
const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const fmt = (n: number) => Math.round(n).toLocaleString("en-US");
const valueOf = (token: number, dim: number) => 0.55 + 0.45 * Math.sin(token * 0.7 + dim * 1.3);

function textCard(kit: Kit, fonts: Fonts, width: number, height: number, lines: () => { text: string; color?: string; size?: number; serif?: boolean }[], tilt?: number) {
  return createCard(
    kit,
    width,
    height,
    (ctx, w, h) => {
      const rows = lines();
      const total = rows.reduce((sum, r) => sum + (r.size ?? 1), 0);
      let y = 0;
      for (const r of rows) {
        const share = (r.size ?? 1) / total;
        ctx.fillStyle = r.color ?? kit.palette.ink;
        const size = h * share * 0.74;
        const face = r.serif ? fonts.serif : fonts.sans;
        ctx.font = `${size}px ${face}`;
        const wide = ctx.measureText(r.text).width;
        if (wide > w * 0.94) ctx.font = `${(size * w * 0.94) / wide}px ${face}`;
        y += h * share;
        ctx.fillText(r.text, w * 0.03, y - h * share * 0.26);
      }
    },
    tilt,
  );
}

function threePassExhibit(kit: Kit, fonts: Fonts, numbers: AttentionV2Numbers) {
  const root = new THREE.Group();
  const scores = numbers.scores;
  root.add(placed(washPlane(kit.palette.blue, 1, 0.9, 0.12), 0.5, 0.002, -0.18));
  const trayZ = -0.42;
  const cellW = 0.8 / scores.length;
  root.add(placed(kit.solid(new THREE.BoxGeometry(0.82, 0.01, 0.07)), 0.5, 0.005, trayZ));
  const top = Math.max(...scores);
  const unit = new THREE.BoxGeometry(cellW - 0.004, 1, 0.05);
  const cells = scores.map((s, i) => {
    const mesh = new THREE.Mesh(unit, liquidMaterial(kit.palette.blue));
    mesh.castShadow = true;
    mesh.position.set(0.1 + (i + 0.5) * cellW, 0, trayZ);
    root.add(mesh);
    return { mesh, score: s * 0.13, prob: Math.exp(6 * (s - top)) * 0.13, isMax: s === top };
  });
  const scanner = new THREE.Mesh(new THREE.SphereGeometry(0.014, 18, 14), kit.pink);
  scanner.castShadow = true;
  root.add(scanner);
  const trayCard = textCard(kit, fonts, 0.8, 0.045, () => [{ text: `shared memory: one float per token, ${fmt(numbers.cap)} tokens (one cell = 64)`, color: kit.palette.faint }], -0.6);
  trayCard.group.position.set(0.5, 0, -0.32);

  const pegZ = -0.17;
  const pegs = Array.from({ length: 16 }, (_, t) => {
    const material = liquidMaterial(kit.palette.blue);
    const mesh = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, 1, 16), material);
    mesh.castShadow = true;
    mesh.position.set(0.1 + (t + 0.5) * 0.05, 0, pegZ);
    root.add(mesh);
    return { mesh, material };
  });
  const pegCard = textCard(kit, fonts, 0.8, 0.045, () => [{ text: "one thread block, 256 threads (one peg = 16)", color: kit.palette.faint }], -0.6);
  pegCard.group.position.set(0.5, 0, -0.1);

  const outZ = 0.04;
  const outW = 0.8 / 64;
  const outs = Array.from({ length: 64 }, (_, d) => {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(outW - 0.002, 1, 0.04), kit.blue);
    mesh.castShadow = true;
    mesh.position.set(0.1 + (d + 0.5) * outW, 0, outZ);
    root.add(mesh);
    return { mesh, value: valueOf(d, d) * 0.07 };
  });
  root.add(placed(kit.solid(new THREE.BoxGeometry(0.82, 0.008, 0.055)), 0.5, 0.004, outZ));
  const outCard = textCard(kit, fonts, 0.8, 0.045, () => [{ text: "output: Σ p · v, one value per dimension, 64 dims", color: kit.palette.faint }], -0.6);
  outCard.group.position.set(0.5, 0, 0.11);

  let pass = 0;
  const passText = [
    ["pass 1 of 3", "a whole dot product per thread, one score per token", "256 of 256 threads busy"],
    ["pass 2 of 3", "the maximum of every score, then exp and their sum", "256 of 256 threads busy"],
    ["pass 3 of 3", "probabilities × values, one thread per dimension", "64 of 256 threads busy, 192 wait"],
  ];
  const passCard = textCard(kit, fonts, 0.8, 0.12, () => [
    { text: passText[pass][0], color: kit.palette.faint, size: 0.8 },
    { text: passText[pass][1], serif: true, size: 1.1 },
    { text: passText[pass][2], color: pass === 2 ? kit.palette.pink : kit.palette.blue, size: 0.9 },
  ], -1.1);
  passCard.group.position.set(0.5, 0, 0.21);
  const timeCard = textCard(kit, fonts, 0.5, 0.05, () => [{ text: `one row at 2,048 tokens: ${numbers.v1At2048} µs`, serif: true }], -0.6);
  timeCard.group.position.set(0.5, 0, 0.36);
  root.add(trayCard.group, pegCard.group, outCard.group, passCard.group, timeCard.group);

  const blue = new THREE.Color(kit.palette.blue);
  const pink = new THREE.Color(kit.palette.pink);
  const paper = new THREE.Color(kit.palette.paper);
  let time = 0;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 11.5 : (time + dt) % 13;
      const p1 = (time - 0.6) / 4;
      const p2 = (time - 5) / 2.6;
      const p3 = smooth(8, 11, time);
      cells.forEach((c, i) => {
        const a = i / cells.length;
        const written = smooth(a, a + 1 / cells.length, p1);
        const scanned = smooth(a, a + 1 / cells.length, p2);
        const height = 0.004 + c.score * written + (c.prob - c.score) * scanned;
        c.mesh.scale.y = height;
        c.mesh.position.y = 0.01 + height / 2;
        const material = c.mesh.material as THREE.MeshLambertMaterial;
        material.color.copy(paper).lerp(blue, written);
        if (c.isMax) material.color.lerp(pink, scanned);
        material.emissive.copy(material.color);
      });
      const sweep = Math.min(1, Math.max(0, p2));
      scanner.visible = p2 > 0 && p2 < 1;
      scanner.position.set(0.1 + sweep * 0.8, 0.17, trayZ);
      const narrow = smooth(7.3, 8, time);
      pegs.forEach(({ mesh, material }, t) => {
        const idle = t >= 4 ? narrow : 0;
        const height = 0.05 - 0.036 * idle;
        mesh.scale.y = height;
        mesh.position.y = height / 2;
        material.color.copy(blue).lerp(paper, idle);
        material.emissive.copy(material.color);
      });
      outs.forEach((o, d) => {
        const height = 0.003 + o.value * smooth(d / 64, d / 64 + 0.25, p3 * 1.25);
        o.mesh.visible = height > 0.004;
        o.mesh.scale.y = height;
        o.mesh.position.y = 0.008 + height / 2;
      });
      const next = time < 4.8 ? 0 : time < 7.6 ? 1 : 2;
      if (next !== pass) {
        pass = next;
        passCard.redraw();
      }
    },
  };
}

function capExhibit(kit: Kit, fonts: Fonts, numbers: AttentionV2Numbers) {
  const root = new THREE.Group();
  const { sharedPoints, cap, limit } = numbers;
  const [, mid, atCap] = sharedPoints;
  const slope = ((atCap.v1 ?? 0) - (mid.v1 ?? 0)) / (atCap.context - mid.context);
  const v1Bytes = (context: number) => (atCap.v1 ?? 0) + slope * (context - atCap.context);
  const v2Bytes = sharedPoints[0].v2;
  const x0 = 0.08;
  const span = 0.82;
  const toX = (bytes: number) => (bytes / limit) * span;
  root.add(placed(washPlane(kit.palette.pink, 1, 0.8, 0.12), 0.5, 0.002, -0.2));

  const trays = [-0.42, -0.2].map((z) => {
    root.add(placed(kit.solid(new THREE.BoxGeometry(span + 0.02, 0.008, 0.07)), x0 + span / 2, 0.004, z));
    return z;
  });
  const v1Material = liquidMaterial(kit.palette.blue);
  const v1Bar = new THREE.Mesh(new THREE.BoxGeometry(1, 0.04, 0.05), v1Material);
  v1Bar.castShadow = true;
  const v2Bar = new THREE.Mesh(new THREE.BoxGeometry(toX(v2Bytes), 0.04, 0.05), kit.blue);
  v2Bar.castShadow = true;
  v2Bar.position.set(x0 + toX(v2Bytes) / 2, 0.028, trays[1]);
  root.add(v1Bar, v2Bar);
  const capX = x0 + toX(v1Bytes(cap));
  const capPost = placed(kit.solid(new THREE.BoxGeometry(0.006, 0.09, 0.09), { material: kit.pink }), capX, 0.045, trays[0]);
  const wall = placed(kit.solid(new THREE.BoxGeometry(0.006, 0.07, 0.36)), x0 + span, 0.035, -0.31);
  root.add(capPost, wall);
  const limitCard = textCard(kit, fonts, 0.34, 0.075, () => [
    { text: `${fmt(limit / 1024)} KB`, serif: true },
    { text: "CUDA's default limit per block", color: kit.palette.faint, size: 0.7 },
  ]);
  limitCard.group.position.set(0.74, 0, -0.53);

  let context = 512;
  const rejected = () => context > cap;
  const v1Card = textCard(kit, fonts, 0.62, 0.085, () => [
    { text: rejected() ? `v1: rejected, past the ${fmt(cap)}-token cap` : `v1: ${fmt(v1Bytes(context))} bytes`, color: rejected() ? kit.palette.pink : kit.palette.ink, serif: true },
    { text: rejected() ? `it would have needed ${fmt(v1Bytes(context))} bytes, still under the limit` : "every score kept, 4 bytes per token", color: kit.palette.faint, size: 0.7 },
  ]);
  v1Card.group.position.set(0.4, 0, trays[0] + 0.1);
  const v2Card = textCard(kit, fonts, 0.62, 0.085, () => [
    { text: `v2: ${fmt(v2Bytes)} bytes`, color: kit.palette.blue, serif: true },
    { text: "one chunk, the same at any length", color: kit.palette.faint, size: 0.7 },
  ]);
  v2Card.group.position.set(0.4, 0, trays[1] + 0.1);
  root.add(limitCard.group, v1Card.group, v2Card.group);

  const railZ = 0.16;
  const rail = kit.solid(new THREE.BoxGeometry(0.8, 0.018, 0.022));
  rail.position.set(0.5, 0.009, railZ);
  const marks = sharedPoints.map((p) => p.context);
  const fraction = (c: number) => Math.log2(c / marks[0]) / Math.log2(marks[marks.length - 1] / marks[0]);
  const ticks: number[] = [];
  for (const c of marks) {
    const x = 0.1 + fraction(c) * 0.8;
    ticks.push(x, 0.019, railZ - 0.02, x, 0.019, railZ + 0.02);
  }
  const tickGeometry = new THREE.BufferGeometry();
  tickGeometry.setAttribute("position", new THREE.Float32BufferAttribute(ticks, 3));
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.03, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(0.1, 0.05, railZ);
  const tickCard = createCard(
    kit,
    0.9,
    0.035,
    (ctx, w, h) => {
      ctx.fillStyle = kit.palette.faint;
      ctx.font = `${h * 0.6}px ${fonts.sans}`;
      ctx.textAlign = "center";
      for (const c of marks) ctx.fillText(fmt(c), w * ((0.05 + fraction(c) * 0.8) / 0.9), h * 0.72);
    },
    -0.7,
  );
  tickCard.group.position.set(0.5, 0, railZ + 0.035);
  const readout = textCard(kit, fonts, 0.24, 0.07, () => [
    { text: "context, tokens", color: kit.palette.faint, size: 0.6 },
    { text: fmt(context), serif: true, size: 1.2 },
  ]);
  readout.group.position.set(0.5, 0, railZ + 0.14);
  root.add(rail, new THREE.LineSegments(tickGeometry, new THREE.LineBasicMaterial({ color: kit.palette.ink })), knob, tickCard.group, readout.group);

  const blue = new THREE.Color(kit.palette.blue);
  const pink = new THREE.Color(kit.palette.pink);
  let u = fraction(context);
  let held = false;
  let time = 0;
  let shown = "";
  const apply = (dt: number) => {
    context = Math.round((marks[0] * Math.pow(marks[marks.length - 1] / marks[0], u)) / 64) * 64;
    knob.position.x = 0.1 + u * 0.8;
    const length = toX(v1Bytes(context));
    v1Bar.scale.x = length;
    v1Bar.position.set(x0 + length / 2, 0.028, trays[0]);
    v1Material.color.lerp(rejected() ? pink : blue, Math.min(1, dt * 4));
    v1Material.emissive.copy(v1Material.color);
    const key = `${context}`;
    if (key !== shown) {
      shown = key;
      readout.redraw();
      v1Card.redraw();
    }
  };
  apply(1);
  return {
    root,
    knob,
    setHeld: (value: boolean) => {
      held = value;
    },
    drag: (x: number) => {
      u = Math.min(1, Math.max(0, (x - 0.1) / 0.8));
      apply(1);
    },
    update: (dt: number, reduced: boolean) => {
      if (!held) {
        if (reduced) u = 1;
        else {
          time += dt;
          u = 0.5 - 0.5 * Math.cos((time * Math.PI * 2) / 18);
        }
      }
      apply(reduced ? 1 : dt);
    },
  };
}

function onlineExhibit(kit: Kit, fonts: Fonts, numbers: AttentionV2Numbers) {
  const root = new THREE.Group();
  const scores = numbers.scores;
  const per = 8;
  const chunks = scores.length / per;
  const k = 6;
  const yScale = 0.24;
  root.add(placed(washPlane(kit.palette.blue, 1, 0.9, 0.12), 0.5, 0.002, -0.16));
  const rowZ = -0.4;
  const colW = 0.022;
  const colX = (i: number) => 0.09 + i * colW + Math.floor(i / per) * 0.02 + colW / 2;
  const columns = scores.map((s, i) => {
    const material = liquidMaterial(kit.palette.paper);
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(colW - 0.004, s * yScale, 0.04), material);
    mesh.castShadow = true;
    mesh.position.set(colX(i), (s * yScale) / 2, rowZ);
    root.add(mesh);
    return material;
  });
  const chunkW = per * colW + 0.012;
  const windowBox = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(chunkW, 0.24, 0.07)), new THREE.LineBasicMaterial({ color: kit.palette.ink }));
  root.add(windowBox);
  const maxBar = new THREE.Mesh(new THREE.BoxGeometry(1, 0.006, 0.014), kit.pink);
  root.add(maxBar);
  const memCard = textCard(kit, fonts, 0.66, 0.075, () => [
    { text: "the box: the chunk of 256 scores in shared memory", color: kit.palette.faint },
    { text: `${fmt(numbers.sharedPoints[0].v2)} bytes at 128 tokens or at 8,192`, color: kit.palette.blue },
  ], -1.2);
  memCard.group.position.set(0.47, 0, -0.29);

  const ms: number[] = [];
  const ls: number[] = [0];
  const accs: number[][] = [Array(8).fill(0)];
  const rescaledL: number[] = [];
  const rescaledAcc: number[][] = [];
  let m = -Infinity;
  for (let c = 0; c < chunks; c++) {
    const part = scores.slice(c * per, (c + 1) * per);
    const next = Math.max(m, ...part);
    const factor = c === 0 ? 0 : Math.exp(k * (m - next));
    rescaledL.push(ls[c] * factor);
    rescaledAcc.push(accs[c].map((a) => a * factor));
    ls.push(rescaledL[c] + part.reduce((sum, s) => sum + Math.exp(k * (s - next)), 0));
    accs.push(rescaledAcc[c].map((a, d) => a + part.reduce((sum, s, t) => sum + Math.exp(k * (s - next)) * valueOf(c * per + t, d), 0)));
    ms.push(next);
    m = next;
  }
  const lMax = Math.max(...ls);
  const accMax = Math.max(...accs.flat());

  const stateZ = 0.02;
  const post = placed(kit.solid(new THREE.CylinderGeometry(0.004, 0.004, yScale, 10)), 0.13, yScale / 2, stateZ);
  const mBall = new THREE.Mesh(new THREE.SphereGeometry(0.018, 20, 16), kit.pink);
  mBall.castShadow = true;
  const lJar = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 1, 28), kit.blue);
  lJar.castShadow = true;
  const accBars = Array.from({ length: 8 }, (_, d) => {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.034, 1, 0.04), kit.blue);
    mesh.castShadow = true;
    mesh.position.set(0.5 + d * 0.052, 0, stateZ);
    root.add(mesh);
    return mesh;
  });
  root.add(post, mBall, lJar, placed(kit.solid(new THREE.BoxGeometry(0.44, 0.006, 0.06)), 0.682, 0.003, stateZ));
  const stateCard = (x: number, width: number, title: string, sub: string) => {
    const card = textCard(kit, fonts, width, 0.08, () => [
      { text: title, serif: true },
      { text: sub, color: kit.palette.faint, size: 0.75 },
    ]);
    card.group.position.set(x, 0, stateZ + 0.07);
    root.add(card.group);
  };
  stateCard(0.13, 0.17, "m", "running max");
  stateCard(0.31, 0.18, "l", "Σ exp(s − m)");
  stateCard(0.68, 0.44, "acc", "Σ exp(s − m) · v, 8 of 64 dims");

  let chunk = 0;
  let rose = false;
  const statusCard = textCard(kit, fonts, 0.82, 0.1, () => [
    { text: `chunk ${chunk + 1} of ${chunks}`, color: kit.palette.faint, size: 0.8 },
    chunk === 0
      ? { text: "the first chunk sets m", serif: true }
      : rose
        ? { text: "m rose: scale l and acc by exp(m_old − m_new), then add", color: kit.palette.pink, serif: true }
        : { text: "m held: add the chunk, nothing to rescale", serif: true },
  ], -1.1);
  statusCard.group.position.set(0.5, 0, 0.24);
  root.add(memCard.group, statusCard.group);

  const blue = new THREE.Color(kit.palette.blue);
  const paper = new THREE.Color(kit.palette.paper);
  const perChunk = 3.2;
  const loop = chunks * perChunk + 3;
  let time = 0;
  let shown = "";
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? loop - 0.5 : (time + dt) % loop;
      const c = Math.min(chunks - 1, Math.floor(time / perChunk));
      const local = time - c * perChunk;
      const light = smooth(0, 0.9, local);
      const rescale = smooth(0.9, 1.9, local);
      const add = smooth(1.9, 2.9, local);
      columns.forEach((material, i) => {
        const owner = Math.floor(i / per);
        const amount = owner < c ? 1 : owner === c ? light : 0;
        material.color.copy(paper).lerp(blue, amount);
        material.emissive.copy(material.color);
      });
      const windowFrom = colX(Math.max(0, c - 1) * per) + (per - 1) * colW * 0.5;
      const windowTo = colX(c * per) + (per - 1) * colW * 0.5;
      windowBox.position.set(c === 0 ? windowTo : windowFrom + (windowTo - windowFrom) * light, 0.12, rowZ);
      const mBefore = c === 0 ? ms[0] : ms[c - 1];
      const mNow = mBefore + (ms[c] - mBefore) * rescale;
      const shownM = c === 0 ? mNow * rescale : mNow;
      const reach = colX(c * per + per - 1) + colW / 2 + 0.01;
      maxBar.visible = c > 0 || rescale > 0;
      maxBar.scale.x = reach - 0.08;
      maxBar.position.set(0.08 + (reach - 0.08) / 2, shownM * yScale, rowZ);
      mBall.position.set(0.13, shownM * yScale, stateZ);
      const lNow = ls[c] + (rescaledL[c] - ls[c]) * rescale + (ls[c + 1] - rescaledL[c]) * add;
      const lHeight = 0.003 + (lNow / lMax) * 0.13;
      lJar.scale.y = lHeight;
      lJar.position.set(0.31, lHeight / 2 + 0.002, stateZ);
      accBars.forEach((mesh, d) => {
        const a = accs[c][d] + (rescaledAcc[c][d] - accs[c][d]) * rescale + (accs[c + 1][d] - rescaledAcc[c][d]) * add;
        const height = 0.003 + (a / accMax) * 0.13;
        mesh.scale.y = height;
        mesh.position.y = height / 2 + 0.006;
      });
      const nowRose = c > 0 && ms[c] > ms[c - 1];
      const key = `${c}${nowRose}`;
      if (key !== shown) {
        shown = key;
        chunk = c;
        rose = nowRose;
        statusCard.redraw();
      }
    },
  };
}

function lanesExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 0.9, 0.9, 0.12), 0.42, 0.002, -0.15));
  const colX = (n: number, j: number) => 0.12 + ((j + 0.5) * 0.48) / n;
  const rowZ = (s: number) => -0.45 + s * 0.042;
  root.add(placed(kit.solid(new THREE.BoxGeometry(0.5, 0.008, 0.35)), 0.36, 0.004, rowZ(3.5)));
  const cells = Array.from({ length: 64 }, (_, i) => {
    const material = liquidMaterial(kit.palette.paper);
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.052, 1, 0.034), material);
    mesh.castShadow = true;
    mesh.position.set(colX(8, i % 8), 0, rowZ(Math.floor(i / 8)));
    root.add(mesh);
    return { mesh, material, step: Math.floor(i / 8) };
  });
  for (let j = 0; j < 8; j++) root.add(placed(new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.03, 14), kit.blue), colX(8, j), 0.015, -0.515));
  const laneCard = createCard(
    kit,
    0.5,
    0.03,
    (ctx, w, h) => {
      ctx.fillStyle = kit.palette.faint;
      ctx.font = `${h * 0.66}px ${fonts.sans}`;
      ctx.textAlign = "center";
      for (let j = 0; j < 8; j++) ctx.fillText(`lane ${j}`, w * ((j + 0.5) / 8) * 0.96 + w * 0.02, h * 0.74);
    },
    -0.7,
  );
  laneCard.group.position.set(0.36, 0, -0.565);

  let step = 0;
  let phase: "read" | "shuffle" = "read";
  const rowCard = textCard(kit, fonts, 0.32, 0.13, () =>
    phase === "read"
      ? [
          { text: "one key, 64 dims", color: kit.palette.faint, size: 0.75 },
          { text: `dims ${step * 8} to ${step * 8 + 7}`, serif: true, size: 1.1 },
          { text: "8 lanes, one 16-byte read", color: kit.palette.blue, size: 0.8 },
        ]
      : [
          { text: "every row read", color: kit.palette.faint, size: 0.75 },
          { text: "8 partial sums", serif: true, size: 1.1 },
          { text: "three shuffles add them", color: kit.palette.pink, size: 0.8 },
        ],
  );
  rowCard.group.position.set(0.8, 0, -0.32);
  const v1Card = textCard(kit, fonts, 0.32, 0.1, () => [
    { text: "v1: one thread walks", serif: true },
    { text: "all 64 dims of its key alone,", color: kit.palette.faint, size: 0.8 },
    { text: "neighbours 128 bytes apart", color: kit.palette.faint, size: 0.8 },
  ]);
  v1Card.group.position.set(0.8, 0, 0.39);

  const levelZ = [0.0, 0.09, 0.18, 0.27];
  const levels = [8, 4, 2, 1].map((n, li) =>
    Array.from({ length: n }, (_, j) => {
      const ball = new THREE.Mesh(new THREE.SphereGeometry(li === 3 ? 0.02 : 0.013, 18, 14), li === 3 ? kit.pink : kit.blue);
      ball.castShadow = true;
      ball.position.set(colX(n, j), 0.02, levelZ[li]);
      root.add(ball);
      return ball;
    }),
  );
  const links = levels.slice(1).map((level, li) =>
    level.flatMap((ball, j) =>
      [0, 1].map((side) => {
        const from = levels[li][j * 2 + side].position;
        const tube = ribbonTube([from.clone(), new THREE.Vector3((from.x + ball.position.x) / 2, 0.035, (from.z + ball.position.z) / 2), ball.position.clone()], 0.0022, kit.blue);
        root.add(tube);
        return tube;
      }),
    ),
  );
  ["shuffle xor 4", "shuffle xor 2", "shuffle xor 1"].forEach((text, i) => {
    const card = textCard(kit, fonts, 0.2, 0.035, () => [{ text, color: kit.palette.faint }], -0.6);
    card.group.position.set(0.76, 0, levelZ[i + 1] + 0.015);
    root.add(card.group);
  });
  const scoreCard = textCard(kit, fonts, 0.5, 0.08, () => [
    { text: "one score for the key", color: kit.palette.pink, serif: true },
    { text: "a warp scores 4 keys at once, a block 32", color: kit.palette.faint, size: 0.75 },
  ]);
  scoreCard.group.position.set(0.36, 0, 0.36);
  root.add(laneCard.group, rowCard.group, v1Card.group, scoreCard.group);

  const blue = new THREE.Color(kit.palette.blue);
  const pink = new THREE.Color(kit.palette.pink);
  const paper = new THREE.Color(kit.palette.paper);
  const perRow = 0.75;
  const treeAt = 0.5 + 8 * perRow;
  const loop = treeAt + 3 * 0.9 + 2.5;
  let time = 0;
  let shown = "";
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? loop - 0.1 : (time + dt) % loop;
      cells.forEach(({ mesh, material, step: s }) => {
        const start = 0.5 + s * perRow;
        const done = smooth(start, start + perRow * 0.9, time);
        const hot = time >= start && time < start + perRow ? Math.sin(((time - start) / perRow) * Math.PI) : 0;
        const height = 0.006 + 0.02 * done;
        mesh.scale.y = height;
        mesh.position.y = 0.008 + height / 2;
        material.color.copy(paper).lerp(blue, done).lerp(pink, hot * 0.7);
        material.emissive.copy(material.color);
      });
      const first = smooth(treeAt - 0.5, treeAt, time);
      levels[0].forEach((ball) => ball.scale.setScalar(0.4 + 0.6 * first));
      links.forEach((tubes, li) => {
        const grow = smooth(treeAt + li * 0.9, treeAt + li * 0.9 + 0.8, time);
        tubes.forEach((tube) => {
          tube.material = grow > 0.5 ? kit.blue : kit.ink;
        });
        levels[li + 1].forEach((ball) => ball.scale.setScalar(0.4 + 0.6 * grow));
      });
      const nextStep = Math.min(7, Math.max(0, Math.floor((time - 0.5) / perRow)));
      const nextPhase = time >= treeAt - 0.5 ? "shuffle" : "read";
      const key = `${nextStep}${nextPhase}`;
      if (key !== shown) {
        shown = key;
        step = nextStep;
        phase = nextPhase;
        rowCard.redraw();
      }
    },
  };
}

export function createAttentionV2Model(kit: Kit, fonts: Fonts, numbers: AttentionV2Numbers): BookModel {
  const passes = threePassExhibit(kit, fonts, numbers);
  const cap = capExhibit(kit, fonts, numbers);
  const online = onlineExhibit(kit, fonts, numbers);
  const lanes = lanesExhibit(kit, fonts);
  return {
    exhibits: [passes.root, cap.root, online.root, lanes.root],
    handles: [cap.knob],
    dragStart: () => cap.setHeld(true),
    drag: (x) => cap.drag(x),
    dragEnd: () => cap.setHeld(false),
    update: (dt, exhibit, _stop, reduced) => {
      if (exhibit === 0) passes.update(dt, reduced);
      if (exhibit === 1) cap.update(dt, reduced);
      if (exhibit === 2) online.update(dt, reduced);
      if (exhibit === 3) lanes.update(dt, reduced);
    },
  };
}
