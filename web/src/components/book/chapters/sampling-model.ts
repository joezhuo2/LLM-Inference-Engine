import * as THREE from "three";
import { gumbel, keys, logits, mulberry, searchSteps } from "@/components/kernels/sampling/figures";
import { createCard, inkEdges, inkSegments, type Kit, washPlane } from "../ink";
import type { BookModel } from "../book-scene";
import type { Fonts } from "../typeset";

export type SamplingNumbers = {
  modes: { column: string; label: string; sort: boolean; us: { "1": number; "16": number } }[];
  greedy: number;
  temperature: number;
  topK: number;
  sort: number;
  busy: number;
  share: number;
  float: [number, number];
};

type Line = { text: string; y: number; x?: number; size?: number; color?: string; serif?: boolean };

const placed = <T extends THREE.Object3D>(object: T, x: number, y: number, z: number): T => {
  object.position.set(x, y, z);
  return object;
};
const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const fmt = (v: number) => v.toLocaleString("en-US");

function label(kit: Kit, fonts: Fonts, w: number, h: number, lines: () => Line[], tilt?: number) {
  return createCard(
    kit,
    w,
    h,
    (ctx, W, H) => {
      for (const l of lines()) {
        ctx.fillStyle = l.color ?? kit.palette.ink;
        ctx.font = `${H * (l.size ?? 0.3)}px ${l.serif ? fonts.serif : fonts.sans}`;
        ctx.fillText(l.text, W * (l.x ?? 0.03), H * l.y);
      }
    },
    tilt,
  );
}

const COLS = logits.length;
const colX = (i: number) => 0.1 + i * 0.058;
const colH = (v: number) => 0.02 + v * 0.045;
const best = logits.indexOf(Math.max(...logits));

function greedyExhibit(kit: Kit, fonts: Fonts, us: number) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 0.95, 0.85, 0.14), 0.45, 0.002, -0.15));
  const colZ = -0.42;
  logits.forEach((v, i) => root.add(placed(kit.solid(new THREE.BoxGeometry(0.042, colH(v), 0.05)), colX(i), colH(v) / 2, colZ)));
  const cap = new THREE.Mesh(new THREE.BoxGeometry(0.046, colH(logits[best]) + 0.002, 0.054), kit.blue);
  cap.position.set(colX(best), colH(logits[best]) / 2, colZ);
  root.add(cap);

  const groups: number[][][] = [];
  let level = Array.from({ length: COLS / 2 }, (_, j) => [2 * j, 2 * j + 1]);
  while (level.length) {
    groups.push(level);
    if (level.length === 1) break;
    const next: number[][] = [];
    for (let j = 0; j < level.length; j += 2) next.push(level.slice(j, j + 2).flat());
    level = next;
  }
  const levelZ = groups.map((_, li) => -0.25 + li * 0.11);
  const winnerOf = (cover: number[]) => cover.reduce((a, b) => (logits[b] > logits[a] ? b : a));
  const meanX = (cover: number[]) => cover.reduce((a, i) => a + colX(i), 0) / cover.length;
  const balls: { group: THREE.Group; height: number }[][] = [];
  const wires: THREE.Object3D[] = [];
  groups.forEach((g, li) => {
    const row: { group: THREE.Group; height: number }[] = [];
    const points: number[] = [];
    g.forEach((cover) => {
      const height = colH(logits[winnerOf(cover)]);
      const thread = new THREE.Group();
      thread.position.set(meanX(cover), 0, levelZ[li]);
      const post = kit.solid(new THREE.CylinderGeometry(0.004, 0.004, 1, 8), { edges: false });
      const ball = new THREE.Mesh(new THREE.SphereGeometry(li === groups.length - 1 ? 0.017 : 0.012, 18, 14), kit.blue);
      ball.castShadow = true;
      thread.add(post, ball);
      root.add(thread);
      row.push({ group: thread, height });
      if (li === 0) for (const i of cover) points.push(colX(i), colH(logits[i]), colZ + 0.025, meanX(cover), height, levelZ[li]);
      else {
        const below = groups[li - 1].filter((c) => c.every((i) => cover.includes(i)));
        for (const c of below) points.push(meanX(c), colH(logits[winnerOf(c)]), levelZ[li - 1], meanX(cover), height, levelZ[li]);
      }
    });
    const wire = inkSegments(points, kit.thin);
    wires.push(wire);
    root.add(wire);
    balls.push(row);
    const card = label(kit, fonts, 0.14, 0.04, () => [{ text: li === groups.length - 1 ? "1 id" : `${g.length} threads`, y: 0.74, size: 0.62, color: kit.palette.faint }]);
    root.add(placed(card.group, 0.87, 0, levelZ[li]));
  });

  const result = label(kit, fonts, 0.84, 0.13, () => [
    { text: `token ${best}: the largest logit, lowest index on ties`, y: 0.32, size: 0.24, serif: true },
    { text: `12 of 32,000 logits; 1,024 threads each keep the best (value, index) they saw`, y: 0.6, size: 0.15, color: kit.palette.faint },
    { text: `then a tree halves them to one id: ${us} µs per call, one block per row`, y: 0.84, size: 0.15, color: kit.palette.blue },
  ]);
  root.add(placed(result.group, 0.47, 0, 0.24));

  let time = 0;
  return {
    root,
    reset: () => {
      time = 0;
    },
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 9.5 : (time + dt) % 11;
      balls.forEach((row, li) => {
        const rise = smooth(0.8 + li * 1.7, 1.8 + li * 1.7, time);
        wires[li].visible = rise > 0.05;
        for (const { group, height } of row) {
          const post = group.children[0];
          post.scale.y = Math.max(0.0001, height * rise);
          post.position.y = (height * rise) / 2;
          group.children[1].position.y = height * rise + 0.01;
          group.visible = rise > 0;
        }
      });
      const win = smooth(7.8, 8.8, time);
      cap.scale.y = Math.max(0.001, win);
      cap.position.y = (colH(logits[best]) * win) / 2;
      cap.visible = win > 0;
    },
  };
}

const softmax = (() => {
  const e = logits.map((l) => Math.exp(l - logits[best]));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
})();
const draw = (rand: () => number) => logits.map(() => gumbel(Math.min(Math.max(rand(), 1e-6), 1 - 1e-6)));
const argmax = (v: number[]) => v.indexOf(Math.max(...v));

function gumbelExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.pink, 0.9, 0.8, 0.12), 0.45, 0.002, -0.2));
  const colZ = -0.4;
  const histZ = -0.08;
  const unit = new THREE.BoxGeometry(0.042, 1, 0.05);
  const noise = logits.map((v, i) => {
    root.add(placed(kit.solid(new THREE.BoxGeometry(0.042, colH(v), 0.05)), colX(i), colH(v) / 2, colZ));
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.046, 1, 0.054), new THREE.MeshLambertMaterial({ color: kit.palette.pink, emissive: kit.palette.pink, emissiveIntensity: 0.45, transparent: true, opacity: 0.7 }));
    mesh.position.set(colX(i), 0, colZ);
    root.add(mesh);
    return mesh;
  });
  const marker = new THREE.Mesh(new THREE.SphereGeometry(0.016, 18, 14), kit.blue);
  marker.castShadow = true;
  root.add(marker);
  const histScale = 0.6;
  const bars = logits.map((_, i) => {
    const bar = new THREE.Mesh(unit, kit.blue);
    bar.castShadow = true;
    bar.position.set(colX(i), 0, histZ);
    root.add(bar);
    const outline = inkEdges(new THREE.BoxGeometry(0.046, softmax[i] * histScale, 0.054), kit.line);
    root.add(placed(outline, colX(i), (softmax[i] * histScale) / 2, histZ));
    return bar;
  });
  const floor = kit.solid(new THREE.BoxGeometry(0.72, 0.004, 0.08));
  root.add(placed(floor, 0.42, 0.002, histZ));
  root.add(placed(label(kit, fonts, 0.15, 0.08, () => [
    { text: "logit / T", y: 0.42, size: 0.34 },
    { text: "+ noise", y: 0.86, size: 0.34, color: kit.palette.pink },
  ]).group, 0.87, 0, colZ));
  root.add(placed(label(kit, fonts, 0.15, 0.08, () => [
    { text: "picks", y: 0.42, size: 0.34, color: kit.palette.blue },
    { text: "softmax", y: 0.86, size: 0.34, color: kit.palette.faint },
  ]).group, 0.87, 0, histZ));

  const still = (() => {
    const rand = mulberry(7);
    const counts = logits.map(() => 0);
    for (let n = 0; n < 2000; n++) counts[argmax(draw(rand).map((g, i) => logits[i] + g))] += 1;
    return counts;
  })();
  let rand = mulberry(7);
  let counts = logits.map(() => 0);
  let from = logits.map(() => 0);
  let to = logits.map(() => 0);
  let pick = best;
  let lastPick = best;
  let clock = 0;
  let reducedShown = false;
  const total = () => counts.reduce((a, b) => a + b, 0);
  const status = label(kit, fonts, 0.84, 0.12, () => [
    { text: reducedShown ? "2,000 draws, simulated in your browser" : `this draw: token ${pick} · ${fmt(total())} draws so far`, y: 0.36, size: 0.26, serif: true },
    { text: "the winner of logit / T + noise is distributed exactly as the softmax:", y: 0.62, size: 0.15, color: kit.palette.faint },
    { text: "no exp, no sum, no prefix scan, just one more argmax", y: 0.86, size: 0.15, color: kit.palette.faint },
  ]);
  root.add(placed(status.group, 0.47, 0, 0.18));
  const period = 0.7;

  const showCounts = (c: number[]) => {
    const n = c.reduce((a, b) => a + b, 0);
    bars.forEach((bar, i) => {
      const h = Math.max(0.0005, n ? (c[i] / n) * histScale : 0);
      bar.scale.y = h;
      bar.position.y = h / 2;
    });
  };
  const showNoise = (g: number[]) => {
    noise.forEach((mesh, i) => {
      const a = colH(logits[i]);
      const b = Math.max(0.003, a + g[i] * 0.045);
      mesh.scale.y = Math.max(0.0005, Math.abs(b - a));
      mesh.position.y = (a + b) / 2;
    });
  };
  return {
    root,
    reset: () => {
      rand = mulberry(7);
      counts = logits.map(() => 0);
      from = logits.map(() => 0);
      to = logits.map(() => 0);
      clock = 0;
      pick = lastPick = best;
      status.redraw();
    },
    update: (dt: number, reduced: boolean) => {
      if (reduced) {
        if (!reducedShown) {
          reducedShown = true;
          status.redraw();
        }
        showNoise(logits.map(() => 0));
        showCounts(still);
        marker.position.set(colX(best), colH(logits[best]) + 0.03, colZ);
        return;
      }
      if (reducedShown) {
        reducedShown = false;
        status.redraw();
      }
      clock += dt;
      if (clock >= period) {
        clock -= period;
        from = to;
        to = draw(rand);
        lastPick = pick;
        pick = argmax(to.map((g, i) => logits[i] + g));
        counts[pick] += 1;
        status.redraw();
      }
      const e = smooth(0, period, clock);
      const g = from.map((f, i) => f + (to[i] - f) * e);
      showNoise(g);
      showCounts(counts);
      const top = (i: number) => Math.max(colH(logits[i]), colH(logits[i]) + g[i] * 0.045) + 0.03;
      const x = colX(lastPick) + (colX(pick) - colX(lastPick)) * e;
      marker.position.set(x, top(e < 0.5 ? lastPick : pick), colZ);
    },
  };
}

function philoxExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 0.95, 0.85, 0.12), 0.5, 0.002, -0.18));
  const STEPS = 4;
  const TOKENS = 10;
  const VOCAB = 32000;
  const cellX = (i: number) => 0.27 + i * 0.066;
  const grids = [-0.43, -0.08];
  const u = (s: number, i: number) => mulberry(7 * 131071 + s * VOCAB + i)();
  const cells: { mesh: THREE.Mesh; height: number; at: number }[] = [];
  grids.forEach((z0, gi) => {
    for (let s = 0; s < STEPS; s++) {
      const z = z0 + s * 0.055;
      for (let i = 0; i < TOKENS; i++) {
        root.add(placed(kit.solid(new THREE.BoxGeometry(0.056, 0.006, 0.045)), cellX(i), 0.003, z));
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.044, 1, 0.034), kit.blue);
        mesh.castShadow = true;
        mesh.position.set(cellX(i), 0, z);
        root.add(mesh);
        cells.push({ mesh, height: 0.008 + u(s, i) * 0.05, at: 0.6 + gi * 4.8 + (s * TOKENS + i) * 0.1 });
      }
      root.add(placed(label(kit, fonts, 0.12, 0.04, () => [{ text: `step ${s}`, y: 0.76, size: 0.66, color: kit.palette.faint }]).group, 0.14, 0, z));
    }
  });
  root.add(placed(label(kit, fonts, 0.66, 0.042, () => [{ text: "request alone, seed 7", y: 0.72, size: 0.56 }]).group, 0.55, 0, -0.52));
  root.add(placed(label(kit, fonts, 0.66, 0.042, () => [{ text: "the same request as row 5 of a batch of 16, seed 7", y: 0.72, size: 0.56 }]).group, 0.55, 0, -0.17));
  const pointer = new THREE.Mesh(new THREE.SphereGeometry(0.012, 16, 12), kit.pink);
  pointer.castShadow = true;
  root.add(pointer);
  let offset = 0;
  const readout = label(kit, fonts, 0.84, 0.13, () => [
    { text: `offset = s × 32,000 + i = ${fmt(offset)}`, y: 0.34, size: 0.26, serif: true },
    { text: "noise = Philox(seed, offset): nothing is stored between calls,", y: 0.62, size: 0.15, color: kit.palette.faint },
    { text: "so the same seed and step give the same noise, alone or in a batch", y: 0.86, size: 0.15, color: kit.palette.faint },
  ]);
  root.add(placed(readout.group, 0.5, 0, 0.2));
  let time = 0;
  return {
    root,
    reset: () => {
      time = 0;
    },
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 11 : (time + dt) % 12.5;
      let current = -1;
      cells.forEach(({ mesh, height, at }, k) => {
        const rise = smooth(at, at + 0.7, time);
        mesh.scale.y = Math.max(0.0001, height * rise);
        mesh.position.y = 0.006 + (height * rise) / 2;
        mesh.visible = rise > 0;
        if (time >= at) current = k;
      });
      const cell = current < 0 ? 0 : current % (STEPS * TOKENS);
      const s = Math.floor(cell / TOKENS);
      const i = cell % TOKENS;
      const target = cells[Math.max(0, current)];
      pointer.visible = !reduced && current >= 0 && current < cells.length - 1;
      pointer.position.set(target.mesh.position.x, 0.09, target.mesh.position.z);
      const next = s * VOCAB + i;
      if (next !== offset) {
        offset = next;
        readout.redraw();
      }
    },
  };
}

function searchExhibit(kit: Kit, fonts: Fonts, mode: "k" | "p") {
  const root = new THREE.Group();
  root.add(placed(washPlane(mode === "k" ? kit.palette.blue : kit.palette.pink, 0.95, 0.85, 0.12), 0.5, 0.002, -0.18));
  const min = Math.min(...keys) - 0.2;
  const max = Math.max(...keys) + 0.2;
  const keyH = (v: number) => 0.01 + (v - min) * 0.05;
  const pegZ = -0.36;
  const pegs = keys.map((v, i) => {
    const peg = kit.solid(new THREE.CylinderGeometry(0.011, 0.011, keyH(v), 16), { threshold: 40 });
    root.add(placed(peg, 0.13 + (i % 10) * 0.077, keyH(v) / 2, -0.48 + Math.floor(i / 10) * 0.08));
    return peg.children[0] as THREE.Mesh;
  });
  const plane = new THREE.Group();
  const pw = 0.41;
  const pd = 0.16;
  const sheet = new THREE.Mesh(new THREE.PlaneGeometry(pw * 2, pd * 2), new THREE.MeshBasicMaterial({ color: kit.palette.pink, transparent: true, opacity: 0.3, side: THREE.DoubleSide, depthWrite: false }));
  sheet.rotation.x = -Math.PI / 2;
  const rim = new THREE.BufferGeometry();
  rim.setAttribute("position", new THREE.Float32BufferAttribute([-pw, 0, -pd, pw, 0, -pd, pw, 0, -pd, pw, 0, pd, pw, 0, pd, -pw, 0, pd, -pw, 0, pd, -pw, 0, -pd], 3));
  plane.add(sheet, new THREE.LineSegments(rim, new THREE.LineBasicMaterial({ color: kit.palette.pink })));
  plane.position.set(0.477, 0, pegZ);
  root.add(plane);

  const top = Math.max(...keys);
  const massAt = (cut: number) => keys.filter((v) => v >= cut).reduce((a, v) => a + Math.exp(v - top), 0);
  const all = massAt(-Infinity);
  let k = 6;
  let p = 0.9;
  const keep = (cut: number) => (mode === "k" ? keys.filter((v) => v >= cut).length >= k : massAt(cut) >= p * all);
  let cuts: number[] = [];
  const PASSES = 7;
  const plan = () => {
    cuts = searchSteps(keep, min, max, PASSES);
  };
  plan();
  let cutNow = min;
  let cutFrom = min;

  const troughZ = -0.12;
  const troughX = 0.1;
  const troughL = 0.8;
  let fill: THREE.Mesh | null = null;
  let post: THREE.Mesh | null = null;
  if (mode === "p") {
    const glass = new THREE.Mesh(new THREE.BoxGeometry(troughL, 0.04, 0.05), kit.glass);
    root.add(placed(glass, troughX + troughL / 2, 0.02, troughZ));
    root.add(placed(inkEdges(new THREE.BoxGeometry(troughL, 0.04, 0.05), kit.thin), troughX + troughL / 2, 0.02, troughZ));
    fill = new THREE.Mesh(new THREE.BoxGeometry(1, 0.034, 0.042), kit.blue);
    fill.castShadow = true;
    root.add(fill);
    post = new THREE.Mesh(new THREE.BoxGeometry(0.006, 0.07, 0.07), kit.pink);
    post.castShadow = true;
    root.add(post);
  }

  let pass = 0;
  const countAt = (cut: number) => keys.filter((v) => v >= cut).length;
  const passLine = () => {
    const cut = cuts[pass];
    const up = keep(cut);
    if (mode === "k") return `pass ${pass + 1} of ${PASSES}: ${countAt(cut)} logits at or above the cut, ${up ? "move it up" : "move it down"}`;
    return `pass ${pass + 1} of ${PASSES}: kept mass ${Math.round((massAt(cut) / all) * 100)}% ${up ? "≥" : "<"} ${Math.round(p * 100)}%, ${up ? "move it up" : "move it down"}`;
  };
  const status = label(kit, fonts, 0.84, 0.13, () => [
    { text: passLine(), y: 0.34, size: 0.22, serif: true },
    {
      text: mode === "k" ? "each pass: every thread counts its share of the row, then one block sum" : "each pass: an exp and a block sum of exp(s − max) at or above the cut",
      y: 0.62,
      size: 0.15,
      color: kit.palette.faint,
    },
    {
      text: mode === "k" ? `the real cut is a 32-bit key: 32 passes, and ties with the k-th largest stay in` : "with top-k also set, the search starts from top-k's cut",
      y: 0.86,
      size: 0.15,
      color: kit.palette.faint,
    },
  ]);
  root.add(placed(status.group, 0.5, 0, mode === "k" ? -0.08 : 0.0));

  const railZ = 0.24;
  const rail = kit.solid(new THREE.BoxGeometry(0.62, 0.016, 0.02));
  root.add(placed(rail, 0.41, 0.008, railZ));
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.026, 26, 20), kit.pink);
  knob.castShadow = true;
  knob.position.set(0.1, 0.044, railZ);
  root.add(knob);
  const KMAX = 20;
  const railX = () => (mode === "k" ? 0.1 + ((k - 1) / (KMAX - 1)) * 0.62 : 0.1 + ((p - 0.05) / 0.95) * 0.62);
  const readout = label(kit, fonts, 0.17, 0.09, () => [
    { text: mode === "k" ? "top-k" : "top-p", y: 0.34, size: 0.28, color: kit.palette.faint },
    { text: mode === "k" ? String(k) : p.toFixed(2), y: 0.86, size: 0.5, serif: true },
  ]);
  root.add(placed(readout.group, 0.84, 0, railZ - 0.03));

  const step = 1.3;
  const start = 0.6;
  let time = 0;
  let held = false;
  const restart = () => {
    plan();
    cutFrom = cutNow;
    time = 0;
    pass = 0;
    status.redraw();
  };
  return {
    root,
    knob,
    reset: () => {
      cutNow = min;
      restart();
    },
    setHeld: (value: boolean) => {
      held = value;
    },
    drag: (x: number) => {
      const t = Math.min(1, Math.max(0, (x - 0.1) / 0.62));
      if (mode === "k") {
        const next = Math.round(1 + t * (KMAX - 1));
        if (next === k) return;
        k = next;
      } else {
        const next = Math.round((0.05 + t * 0.95) * 20) / 20;
        if (next === p) return;
        p = next;
      }
      readout.redraw();
      restart();
    },
    update: (dt: number, reduced: boolean) => {
      time = reduced ? start + PASSES * step : held ? 0 : Math.min(start + PASSES * step, time + dt);
      const j = Math.min(PASSES - 1, Math.max(0, Math.floor((time - start) / step)));
      const prev = j === 0 ? cutFrom : cuts[j - 1];
      const e = reduced ? 1 : smooth(start + j * step, start + j * step + 0.9, time);
      cutNow = time < start ? cutFrom : prev + (cuts[j] - prev) * e;
      plane.position.y = keyH(cutNow);
      if (j !== pass) {
        pass = j;
        status.redraw();
      }
      const settled = cuts[j];
      pegs.forEach((mesh, i) => {
        mesh.material = time >= start && keys[i] >= (e > 0.6 ? settled : prev) ? kit.blue : kit.ink;
      });
      knob.position.x = railX();
      if (fill && post) {
        const share = massAt(cutNow) / all;
        fill.scale.x = Math.max(0.001, share * troughL);
        fill.position.set(troughX + (share * troughL) / 2, 0.02, troughZ);
        post.position.set(troughX + p * troughL, 0.035, troughZ);
      }
    },
  };
}

function costExhibit(kit: Kit, fonts: Fonts, numbers: SamplingNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 0.95, 0.85, 0.12), 0.5, 0.002, -0.15));
  const modes = numbers.modes;
  const max = Math.max(...modes.map((m) => m.us["1"]));
  const x0 = 0.36;
  const span = 0.56;
  const rowZ = (i: number) => -0.44 + i * 0.105;
  const bars = modes.map((m, i) => {
    const us = m.us["1"];
    const material = m.sort ? kit.pink : us < numbers.sort ? kit.blue : kit.ink;
    const bar = kit.solid(new THREE.BoxGeometry(1, 0.04, 0.05), { material, edges: false });
    root.add(placed(bar, x0, 0.02, rowZ(i)));
    root.add(
      placed(
        label(kit, fonts, 0.25, 0.075, () => [
          { text: m.label, y: 0.44, size: 0.38, color: m.sort ? kit.palette.pink : us < numbers.sort ? kit.palette.blue : kit.palette.ink },
          { text: `${us.toFixed(1)} µs`, y: 0.88, size: 0.34, color: kit.palette.faint },
        ]).group,
        0.2,
        0,
        rowZ(i) - 0.03,
      ),
    );
    return { bar, length: (us / max) * span };
  });
  const wallX = x0 + (numbers.sort / max) * span;
  const depth = rowZ(modes.length - 1) - rowZ(0) + 0.1;
  const wall = new THREE.Mesh(new THREE.PlaneGeometry(depth, 0.09), new THREE.MeshBasicMaterial({ color: kit.palette.pink, transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false }));
  wall.rotation.y = Math.PI / 2;
  root.add(placed(wall, wallX, 0.045, (rowZ(0) + rowZ(modes.length - 1)) / 2));
  const axis = new THREE.BufferGeometry();
  axis.setAttribute("position", new THREE.Float32BufferAttribute([x0, 0.001, rowZ(0) - 0.05, x0, 0.001, rowZ(modes.length - 1) + 0.05], 3));
  root.add(new THREE.LineSegments(axis, new THREE.LineBasicMaterial({ color: kit.palette.ink })));
  const legend = label(kit, fonts, 0.84, 0.11, () => [
    { text: `top-k 50 takes ${numbers.topK} µs; sorting the row alone takes ${numbers.sort} µs`, y: 0.36, size: 0.22, serif: true },
    { text: "µs per call, one row, vocab 32,000, temperature 0.7; the pink wall is the sort alone", y: 0.66, size: 0.15, color: kit.palette.faint },
    { text: "blue bars finish before the sort would; temperature only is the slowest of all", y: 0.9, size: 0.15, color: kit.palette.faint },
  ]);
  root.add(placed(legend.group, 0.5, 0, 0.23));
  let time = 0;
  return {
    root,
    reset: () => {
      time = 0;
    },
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 9 : Math.min(9, time + dt);
      bars.forEach(({ bar, length }, i) => {
        const l = Math.max(0.001, length * smooth(0.5 + i * 0.25, 1.7 + i * 0.25, time));
        bar.scale.x = l;
        bar.position.x = x0 + l / 2;
      });
    },
  };
}

function fp64Exhibit(kit: Kit, fonts: Fonts, numbers: SamplingNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.pink, 0.95, 0.85, 0.12), 0.5, 0.002, -0.15));
  const jarR = 0.055;
  const jarH = 0.2;
  const jarZ = -0.36;
  const jars = [
    { x: 0.16, value: numbers.share, material: kit.blue, name: "FP64 share of instructions" },
    { x: 0.37, value: numbers.busy, material: kit.pink, name: "FP64 pipe busy" },
  ].map(({ x, value, material, name }) => {
    const glass = new THREE.Group();
    const wall = new THREE.Mesh(new THREE.CylinderGeometry(jarR, jarR, jarH, 40, 1, true), kit.glass);
    wall.position.y = jarH / 2;
    const rims: number[] = [];
    for (const y of [0, jarH]) {
      for (let k = 0; k < 40; k++) {
        const a = (k / 40) * Math.PI * 2;
        const b = ((k + 1) / 40) * Math.PI * 2;
        rims.push(Math.cos(a) * jarR, y, Math.sin(a) * jarR, Math.cos(b) * jarR, y, Math.sin(b) * jarR);
      }
    }
    rims.push(-jarR, 0, 0, -jarR, jarH, 0, jarR, 0, 0, jarR, jarH, 0);
    const rimGeometry = new THREE.BufferGeometry();
    rimGeometry.setAttribute("position", new THREE.Float32BufferAttribute(rims, 3));
    const fill = new THREE.Mesh(new THREE.CylinderGeometry(jarR * 0.94, jarR * 0.94, 1, 32), material);
    fill.castShadow = true;
    glass.add(fill, wall, new THREE.LineSegments(rimGeometry, new THREE.LineBasicMaterial({ color: kit.palette.ink })));
    root.add(placed(glass, x, 0, jarZ));
    root.add(
      placed(
        label(kit, fonts, 0.19, 0.07, () => [
          { text: name, y: 0.32, size: 0.2, color: kit.palette.faint },
          { text: `${value}%`, y: 0.84, size: 0.5, serif: true },
        ]).group,
        x,
        0,
        jarZ + jarR + 0.03,
      ),
    );
    return { fill, value };
  });
  root.add(
    placed(
      label(kit, fonts, 0.42, 0.2, () => [
        { text: "gumbel(u) = −log(−log u) in double", y: 0.16, size: 0.12, serif: true },
        { text: "temperature only, as written", y: 0.34, size: 0.1, color: kit.palette.faint },
        { text: `${numbers.temperature} µs`, y: 0.54, size: 0.17, serif: true, color: kit.palette.pink },
        { text: "float u and __logf, a throwaway copy", y: 0.72, size: 0.1, color: kit.palette.faint },
        { text: `${numbers.float[0].toFixed(0)} to ${numbers.float[1].toFixed(0)} µs`, y: 0.92, size: 0.17, serif: true, color: kit.palette.blue },
      ]).group,
      0.71,
      0,
      -0.4,
    ),
  );

  const CELLS = 40;
  const cellX = (i: number) => 0.1 + i * 0.0205;
  const strips = [
    { z: -0.08, count: CELLS, text: "temperature only: noise for all 32,000 tokens", color: kit.palette.pink },
    { z: 0.14, count: 1, text: `top-k 50: noise for the 50 kept tokens only, ${numbers.topK} µs`, color: kit.palette.blue },
  ];
  const drops: { mesh: THREE.Mesh; at: number }[] = [];
  strips.forEach(({ z, count, text, color }) => {
    root.add(placed(kit.solid(new THREE.BoxGeometry(CELLS * 0.0205, 0.008, 0.04)), cellX(19.5), 0.004, z));
    for (let i = 0; i < CELLS; i++) {
      if (i < count) {
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.014, 0.03, 0.03), kit.pink);
        mesh.castShadow = true;
        mesh.position.set(cellX(i), 0.023, z);
        root.add(mesh);
        drops.push({ mesh, at: 1.6 + i * 0.15 });
      }
    }
    root.add(placed(label(kit, fonts, 0.84, 0.04, () => [{ text, y: 0.72, size: 0.52, color }]).group, 0.5, 0, z + 0.05));
  });
  let time = 0;
  return {
    root,
    reset: () => {
      time = 0;
    },
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 9.5 : (time + dt) % 11;
      const level = smooth(0.4, 1.6, time);
      for (const { fill, value } of jars) {
        const h = Math.max(0.0005, (value / 100) * jarH * 0.95 * level);
        fill.scale.y = h;
        fill.position.y = h / 2 + 0.002;
      }
      for (const { mesh, at } of drops) {
        const d = smooth(at, at + 0.7, time);
        mesh.visible = d > 0;
        mesh.position.y = 0.023 + (1 - d) * 0.08;
      }
    },
  };
}

export function createSamplingModel(kit: Kit, fonts: Fonts, numbers: SamplingNumbers): BookModel {
  const exhibits = [greedyExhibit(kit, fonts, numbers.greedy), gumbelExhibit(kit, fonts), philoxExhibit(kit, fonts), searchExhibit(kit, fonts, "k"), searchExhibit(kit, fonts, "p"), costExhibit(kit, fonts, numbers), fp64Exhibit(kit, fonts, numbers)];
  const topK = exhibits[3] as ReturnType<typeof searchExhibit>;
  const topP = exhibits[4] as ReturnType<typeof searchExhibit>;
  let active: ReturnType<typeof searchExhibit> | null = null;
  return {
    exhibits: exhibits.map((e) => e.root),
    handles: [topK.knob, topP.knob],
    dragStart: (handle) => {
      active = handle === topK.knob ? topK : handle === topP.knob ? topP : null;
      active?.setHeld(true);
    },
    drag: (x) => active?.drag(x),
    dragEnd: () => {
      active?.setHeld(false);
      active = null;
    },
    enter: (stop) => exhibits[stop]?.reset(),
    update: (dt, exhibit, _stop, reduced) => exhibits[exhibit]?.update(dt, reduced),
  };
}
