import * as THREE from "three";
import { others, scores, table } from "@/components/kernels/attention/figures";
import { words } from "@/components/kernels/attention/prologue-figures";
import { createCard, type Kit, ribbonTube, washPlane } from "../ink";
import type { BookModel } from "../book-scene";
import type { Fonts } from "../typeset";

export type PagedNumbers = {
  ttft128: number;
  ttft2048: number;
  perToken128: number;
  perToken2048: number;
  gbs: number;
  percent: number;
  stepMs: number;
  ratio: number;
  stepV1: number;
  stepV2: number;
  v2Loop: number;
  split8: number;
};

type Row = { text: string; y: number; size: number; color?: string; serif?: boolean; x?: number; align?: CanvasTextAlign };

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const fmt = (n: number) => n.toLocaleString("en-US");

function tools(kit: Kit, fonts: Fonts) {
  const ink = kit.palette.ink;
  const faint = kit.palette.faint;
  const label = (w: number, h: number, rows: () => Row[], tilt?: number) =>
    createCard(
      kit,
      w,
      h,
      (ctx, W, H) => {
        for (const r of rows()) {
          ctx.fillStyle = r.color ?? ink;
          ctx.font = `${H * r.size}px ${r.serif ? fonts.serif : fonts.sans}`;
          ctx.textAlign = r.align ?? "left";
          ctx.fillText(r.text, W * (r.x ?? (r.align === "center" ? 0.5 : 0.04)), H * r.y);
        }
        ctx.textAlign = "left";
      },
      tilt,
    );
  const put = <T extends THREE.Object3D>(root: THREE.Group, object: T, x: number, z: number, y = 0): T => {
    object.position.set(x, y, z);
    root.add(object);
    return object;
  };
  const card = (root: THREE.Group, x: number, z: number, w: number, h: number, rows: () => Row[], tilt?: number) => {
    const c = label(w, h, rows, tilt);
    put(root, c.group, x, z);
    return c;
  };
  const flat = (root: THREE.Group, x: number, z: number, y: number, w: number, h: number, rows: () => Row[]) => {
    const c = label(w, h, rows, -Math.PI / 2);
    put(root, c.group, x, z - h / 2, y + 0.002);
    return c;
  };
  const box = (root: THREE.Group, x: number, z: number, w: number, h: number, d: number, material?: THREE.Material, y = 0) => {
    const b = kit.solid(new THREE.BoxGeometry(w, h, d), material ? { material, threshold: 40 } : { threshold: 40 });
    return put(root, b, x, z, y + h / 2);
  };
  const liquid = (root: THREE.Group, x: number, z: number, w: number, d: number, material: THREE.Material) => {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, 1, d), material);
    mesh.castShadow = true;
    mesh.scale.y = 0.0001;
    return put(root, mesh, x, z);
  };
  const level = (mesh: THREE.Object3D, height: number, base = 0) => {
    mesh.scale.y = Math.max(0.0001, height);
    mesh.position.y = base + Math.max(0.0001, height) / 2;
  };
  const bead = (root: THREE.Group, material: THREE.Material, r = 0.012) => {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 18, 14), material);
    mesh.castShadow = true;
    mesh.visible = false;
    root.add(mesh);
    return mesh;
  };
  const fly = (mesh: THREE.Object3D, from: THREE.Vector3Like, to: THREE.Vector3Like, p: number, lift: number) => {
    mesh.visible = p > 0 && p < 1;
    mesh.position.set(lerp(from.x, to.x, p), lerp(from.y, to.y, p) + Math.sin(p * Math.PI) * lift, lerp(from.z, to.z, p));
  };
  const outline = (root: THREE.Group, w: number, d: number, material: THREE.Material, t = 0.006) => {
    const group = new THREE.Group();
    const bar = (bw: number, bd: number, x: number, z: number) => {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(bw, 0.006, bd), material);
      mesh.position.set(x, 0, z);
      group.add(mesh);
    };
    bar(w + t, t, 0, -d / 2);
    bar(w + t, t, 0, d / 2);
    bar(t, d, -w / 2, 0);
    bar(t, d, w / 2, 0);
    root.add(group);
    return group;
  };
  const trough = (root: THREE.Group, x0: number, z: number, length: number, material: THREE.Material, fraction: number) => {
    box(root, x0 + length / 2, z, length, 0.008, 0.05);
    const fill = new THREE.Mesh(new THREE.BoxGeometry(1, 0.022, 0.036), material);
    fill.castShadow = true;
    put(root, fill, x0, z, 0.019);
    const set = (f: number) => {
      fill.visible = f > 0.0005;
      fill.scale.x = Math.max(0.0001, length * f);
      fill.position.x = x0 + (length * f) / 2;
    };
    set(fraction);
    return set;
  };
  return { ink, faint, blue: kit.palette.blue, pink: kit.palette.pink, card, flat, box, liquid, level, bead, fly, outline, trough, put };
}

type Tools = ReturnType<typeof tools>;
type Exhibit = { root: THREE.Group; update: (dt: number, reduced: boolean) => void; reset: () => void };

function looped(period: number, still: number, run: (t: number) => void): Pick<Exhibit, "update" | "reset"> {
  let time = 0;
  return {
    update: (dt, reduced) => {
      time = reduced ? still : (time + dt) % period;
      run(time);
    },
    reset: () => {
      time = 0;
    },
  };
}

function prefillExhibit(kit: Kit, t: Tools, n: PagedNumbers): Exhibit {
  const root = new THREE.Group();
  t.put(root, washPlane(t.blue, 1, 0.9, 0.14), 0.52, -0.2, 0.002);
  const xs = Array.from({ length: 8 }, (_, i) => 0.2 + i * 0.095);
  t.card(root, 0.54, -0.61, 0.76, 0.04, () => [{ text: "the prompt: 8 tokens, all known up front (illustrative)", y: 0.7, size: 0.5, color: t.faint }], -0.5);
  xs.forEach((x, i) => t.card(root, x, -0.5, 0.085, 0.042, () => [{ text: words[i], y: 0.7, size: 0.5, serif: true, align: "center" }]));
  t.box(root, 0.533, -0.31, 0.78, 0.04, 0.08);
  const glow = t.liquid(root, 0.533, -0.31, 0.76, 0.06, kit.blue);
  t.card(root, 0.533, -0.255, 0.7, 0.036, () => [{ text: "model: one wide pass, every position side by side", y: 0.7, size: 0.52, align: "center" }], -0.5);
  const kz = -0.13;
  const vz = -0.04;
  t.card(root, 0.1, kz - 0.01, 0.05, 0.036, () => [{ text: "K", y: 0.75, size: 0.62, serif: true, align: "center" }]);
  t.card(root, 0.1, vz - 0.01, 0.05, 0.036, () => [{ text: "V", y: 0.75, size: 0.62, serif: true, align: "center" }]);
  const fills = xs.flatMap((x) => {
    t.box(root, x, kz, 0.08, 0.008, 0.06);
    t.box(root, x, vz, 0.08, 0.008, 0.06);
    return [t.liquid(root, x, kz, 0.066, 0.046, kit.blue), t.liquid(root, x, vz, 0.066, 0.046, kit.pink)];
  });
  const ups = xs.map(() => t.bead(root, kit.blue));
  const downK = xs.map(() => t.bead(root, kit.blue));
  const downV = xs.map(() => t.bead(root, kit.pink));
  t.card(root, 0.54, 0.02, 0.8, 0.038, () => [{ text: "8 positions in, 8 keys and 8 values out, in one pass", y: 0.7, size: 0.52 }], -0.5);
  t.card(root, 0.54, 0.1, 0.8, 0.07, () => [
    { text: "time to first token is this pass, measured", y: 0.4, size: 0.34, color: t.faint },
    { text: `per prompt token: ${n.perToken128} ms at 128 tokens, ${n.perToken2048} ms at 2,048`, y: 0.84, size: 0.34, color: t.faint },
  ]);
  const bars: [string, number, number][] = [
    ["128 tokens", n.ttft128, 0.22],
    ["2,048 tokens", n.ttft2048, 0.33],
  ];
  for (const [name, ms, z] of bars) {
    t.card(root, 0.17, z - 0.035, 0.17, 0.055, () => [
      { text: name, y: 0.42, size: 0.34, color: t.faint },
      { text: `${fmt(ms)} ms`, y: 0.88, size: 0.42, serif: true },
    ]);
    t.trough(root, 0.29, z, 0.64, kit.blue, ms / n.ttft2048);
  }
  const looper = looped(8, 5, (time) => {
    const up = smooth(0.6, 1.6, time);
    const down = smooth(2.4, 3.4, time);
    const fade = 1 - smooth(7.3, 8, time);
    xs.forEach((x, i) => {
      t.fly(ups[i], { x, y: 0.03, z: -0.48 }, { x, y: 0.05, z: -0.31 }, up, 0.06);
      t.fly(downK[i], { x, y: 0.05, z: -0.31 }, { x, y: 0.02, z: kz }, down, 0.05);
      t.fly(downV[i], { x, y: 0.05, z: -0.31 }, { x, y: 0.02, z: vz }, down, 0.08);
    });
    t.level(glow, 0.012 * smooth(1.5, 2.3, time) * fade, 0.04);
    for (const f of fills) t.level(f, 0.026 * smooth(3.2, 3.9, time) * fade, 0.008);
  });
  return { root, ...looper };
}

function decodeExhibit(kit: Kit, t: Tools, n: PagedNumbers): Exhibit {
  const root = new THREE.Group();
  t.put(root, washPlane(t.pink, 1, 0.9, 0.12), 0.52, -0.22, 0.002);
  const prompt = 8;
  const xs = Array.from({ length: 12 }, (_, i) => 0.16 + i * 0.066);
  t.card(root, 0.54, -0.62, 0.8, 0.06, () => [
    { text: "8 prompt tokens, then one new token per step (illustrative)", y: 0.42, size: 0.34, color: t.faint },
    { text: "each step needs the token the last one produced, so steps cannot overlap", y: 0.86, size: 0.3, color: t.faint },
  ]);
  const tokens = xs.map((x, i) => t.card(root, x, -0.5, 0.058, 0.04, () => [{ text: words[i].slice(0, 5), y: 0.7, size: 0.44, serif: true, align: "center" }]).group);
  const dash: number[] = [];
  for (const x of xs.slice(prompt)) {
    const a = x - 0.029;
    const b = x + 0.029;
    dash.push(a, 0.002, -0.5, b, 0.002, -0.5, b, 0.002, -0.5, b, 0.002, -0.47, b, 0.002, -0.47, a, 0.002, -0.47, a, 0.002, -0.47, a, 0.002, -0.5);
  }
  const dashGeometry = new THREE.BufferGeometry();
  dashGeometry.setAttribute("position", new THREE.Float32BufferAttribute(dash, 3));
  root.add(new THREE.LineSegments(dashGeometry, new THREE.LineDashedMaterial({ color: t.faint, dashSize: 0.006, gapSize: 0.005 })).computeLineDistances());
  const head = t.box(root, xs[prompt - 1], -0.34, 0.058, 0.03, 0.07);
  const headGlow = t.liquid(root, xs[prompt - 1], -0.34, 0.046, 0.056, kit.blue);
  t.card(root, 0.3, -0.31, 0.32, 0.034, () => [{ text: "model: one position per step", y: 0.7, size: 0.54 }], -0.5);
  const kz = -0.2;
  const vz = -0.115;
  const cells = xs.map((x) => {
    t.box(root, x, kz, 0.058, 0.007, 0.06);
    t.box(root, x, vz, 0.058, 0.007, 0.06);
    return [t.liquid(root, x, kz, 0.046, 0.046, kit.blue), t.liquid(root, x, vz, 0.046, 0.046, kit.pink)];
  });
  const reads = xs.map(() => t.bead(root, kit.pink, 0.009));
  let shownStep = 0;
  const stepCard = t.card(root, 0.54, -0.05, 0.8, 0.04, () => [
    { text: `step ${shownStep + 1}: reads ${prompt + shownStep} cached K/V pairs, adds 1`, y: 0.7, size: 0.5, color: t.pink },
  ], -0.5);
  t.card(root, 0.54, 0.06, 0.8, 0.07, () => [
    { text: `batch 1: ${fmt(n.gbs)} GB/s moved, ${n.percent}% of the GPU's measured peak`, y: 0.42, size: 0.36 },
    { text: "the step waits on memory, not on arithmetic", y: 0.86, size: 0.32, color: t.faint },
  ]);
  box100(t, root, kit, n.percent);
  const per = 2.4;
  const looper = looped(4 * per + 1.8, 4 * per + 0.5, (time) => {
    const s = Math.min(3, Math.floor(time / per));
    const local = time - s * per;
    const done = time >= 4 * per;
    const fade = 1 - smooth(4 * per + 1.2, 4 * per + 1.8, time);
    const current = done ? 3 : s;
    xs.forEach((x, i) => {
      const added = i < prompt ? 1 : i - prompt < current ? 1 : i - prompt === current ? smooth(1.2, 1.8, done ? 9 : local) : 0;
      const shown = i < prompt ? 1 : added * fade;
      t.level(cells[i][0], 0.024 * shown, 0.007);
      t.level(cells[i][1], 0.024 * shown, 0.007);
      tokens[i].scale.setScalar(Math.max(0.001, shown));
      const read = done ? 1 : smooth(0, 1.0, local);
      t.fly(reads[i], { x, y: 0.03, z: kz }, { x: xs[prompt - 1 + current], y: 0.04, z: -0.34 }, i < prompt + current ? read : 0, 0.05);
    });
    const hx = xs[prompt - 1 + current];
    head.position.x = hx;
    headGlow.position.x = hx;
    t.level(headGlow, 0.008 * (done ? 0 : smooth(0.8, 1.1, local) * (1 - smooth(1.6, 2.0, local))), 0.03);
    if (current !== shownStep) {
      shownStep = current;
      stepCard.redraw();
    }
  });
  return { root, ...looper };
}

function box100(t: Tools, root: THREE.Group, kit: Kit, percent: number) {
  t.trough(root, 0.16, 0.2, 0.74, kit.blue, percent / 100);
  t.card(root, 0.53, 0.25, 0.76, 0.032, () => [
    { text: "0%", y: 0.72, size: 0.56, color: t.faint, x: 0.01 },
    { text: `${percent}%`, y: 0.72, size: 0.56, color: t.blue, x: 0.01 + (0.97 * percent) / 100, align: "center" },
    { text: "100% of measured peak", y: 0.72, size: 0.56, color: t.faint, x: 0.99, align: "right" },
  ], -0.6);
}

function cacheExhibit(kit: Kit, t: Tools, n: PagedNumbers): Exhibit {
  const root = new THREE.Group();
  t.put(root, washPlane(t.blue, 1, 0.9, 0.12), 0.52, -0.2, 0.002);
  t.card(root, 0.53, -0.62, 0.8, 0.04, () => [{ text: "the KV cache exists so decode does not recompute the prefill", y: 0.7, size: 0.5 }], -0.5);
  const centers = [0.2, 0.53, 0.86];
  const z = -0.47;
  const names: [string, string][] = [
    ["prefill", "all positions, once"],
    ["KV cache", "K and V per position"],
    ["decode", "one token per step"],
  ];
  centers.forEach((x, i) => {
    t.box(root, x, z, 0.2, i === 1 ? 0.02 : 0.05, 0.08);
    t.card(root, x, z + 0.06, 0.2, 0.06, () => [
      { text: names[i][0], y: 0.45, size: 0.4, serif: true, align: "center" },
      { text: names[i][1], y: 0.85, size: 0.27, color: t.faint, align: "center" },
    ]);
  });
  for (let k = 0; k < 6; k++) {
    t.box(root, 0.455 + k * 0.03, z - 0.02, 0.024, 0.012, 0.03, kit.blue, 0.02);
    t.box(root, 0.455 + k * 0.03, z + 0.02, 0.024, 0.012, 0.03, kit.pink, 0.02);
  }
  for (const [a, b] of [
    [0.3, 0.43],
    [0.63, 0.76],
  ]) {
    root.add(ribbonTube([new THREE.Vector3(a, 0.03, z), new THREE.Vector3((a + b) / 2, 0.05, z), new THREE.Vector3(b, 0.03, z)], 0.003, kit.ink));
  }
  t.card(root, 0.365, z + 0.06, 0.08, 0.03, () => [{ text: "write", y: 0.72, size: 0.6, color: t.faint, align: "center" }]);
  t.card(root, 0.695, z + 0.06, 0.08, 0.03, () => [{ text: "read", y: 0.72, size: 0.6, color: t.faint, align: "center" }]);
  const written = t.bead(root, kit.blue, 0.014);
  const read = [0, 1, 2].map(() => t.bead(root, kit.pink, 0.011));
  const bars: [string, string, number, THREE.Material, number][] = [
    ["without a cache, rerun the prefill for each new token", `${fmt(n.ttft2048)} ms, a 2,048-token prefill`, n.ttft2048, kit.pink, 0.0],
    ["with the cache, one decode step at 1,984 tokens", `${n.stepMs} ms, about ${n.ratio}x less`, n.stepMs, kit.blue, 0.24],
  ];
  const growers = bars.map(([title, value, ms, material, bz]) => {
    t.card(root, 0.53, bz - 0.13, 0.8, 0.075, () => [
      { text: title, y: 0.38, size: 0.3, color: t.faint },
      { text: value, y: 0.84, size: 0.4, serif: true, color: material === kit.pink ? t.pink : t.blue },
    ]);
    return { set: t.trough(root, 0.13, bz, 0.8, material, 1), share: ms / n.ttft2048 };
  });
  const grow = 3;
  const looper = looped(9, 8, (time) => {
    t.fly(written, { x: 0.3, y: 0.04, z }, { x: 0.43, y: 0.04, z }, smooth(0.3, 1.2, time), 0.03);
    read.forEach((b, i) => t.fly(b, { x: 0.63, y: 0.04, z }, { x: 0.76, y: 0.04, z }, smooth(1.4 + i * 1.6, 2.3 + i * 1.6, time), 0.03));
    for (const g of growers) g.set(Math.min(g.share, smooth(1.4, 1.4 + grow, time)) * (1 - smooth(8.4, 9, time)));
  });
  return { root, ...looper };
}

function homeExhibit(kit: Kit, t: Tools): Exhibit {
  const root = new THREE.Group();
  t.put(root, washPlane(t.blue, 1, 0.9, 0.12), 0.52, -0.15, 0.002);
  t.card(root, 0.53, -0.62, 0.8, 0.04, () => [{ text: "one sequence's cache, growing a token per step", y: 0.7, size: 0.5 }], -0.5);
  const slotX = (k: number) => 0.18 + k * 0.135;
  const sz = -0.5;
  const fills = table.map((_, k) => {
    t.box(root, slotX(k), sz, 0.125, 0.01, 0.06);
    const fill = new THREE.Mesh(new THREE.BoxGeometry(1, 0.018, 0.044), kit.blue);
    fill.castShadow = true;
    return t.put(root, fill, slotX(k), sz, 0.019);
  });
  t.card(root, 0.53, -0.44, 0.8, 0.034, () => [{ text: "16 tokens per block, 6 blocks = 96 tokens", y: 0.72, size: 0.56, color: t.faint }], -0.5);
  t.card(root, 0.53, -0.34, 0.8, 0.034, () => [{ text: "KV pool on the GPU, shared by every request", y: 0.72, size: 0.56 }], -0.5);
  const tile = (id: number) => ({ x: 0.22 + (id % 4) * 0.2, z: -0.22 + Math.floor(id / 4) * 0.095 });
  for (let id = 0; id < 16; id++) {
    const { x, z } = tile(id);
    t.box(root, x, z, 0.18, others.has(id) ? 0.035 : 0.008, 0.075);
  }
  const movers = table.map(() => {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 0.018, 0.044), kit.blue);
    mesh.castShadow = true;
    mesh.visible = false;
    root.add(mesh);
    return mesh;
  });
  t.card(root, 0.53, 0.18, 0.8, 0.075, () => [
    { text: "blue: this sequence", y: 0.4, size: 0.34, color: t.blue },
    { text: "raised: other requests", y: 0.4, size: 0.34, color: t.ink, x: 0.38 },
    { text: "flat: free", y: 0.4, size: 0.34, color: t.faint, x: 0.75 },
    { text: "placed in whichever blocks are free, in any order", y: 0.86, size: 0.34, color: t.faint },
  ]);
  const per = 1.1;
  const looper = looped(10, 9, (time) => {
    const fade = 1 - smooth(9.3, 10, time);
    table.forEach((phys, k) => {
      const start = 0.3 + k * per;
      const f = smooth(start, start + 0.8, time) * fade;
      const x0 = slotX(k) - 0.055;
      fills[k].visible = f > 0.001;
      fills[k].scale.x = Math.max(0.0001, 0.11 * f);
      fills[k].position.x = x0 + 0.055 * f;
      const p = smooth(start + 0.9, start + 1.8, time);
      const target = tile(phys);
      const m = movers[k];
      m.visible = p > 0 && fade > 0.01;
      m.scale.set(lerp(0.11, 0.16, p), fade, lerp(1, 1.4, p));
      m.position.set(lerp(slotX(k), target.x, p), lerp(0.019, 0.017, p) + Math.sin(p * Math.PI) * 0.12, lerp(sz, target.z, p));
    });
  });
  return { root, ...looper };
}

function jobExhibit(kit: Kit, t: Tools): Exhibit {
  const root = new THREE.Group();
  t.put(root, washPlane(t.blue, 1, 0.9, 0.12), 0.52, -0.15, 0.002);
  const cx = (i: number) => 0.3 + i * 0.042;
  const qz = -0.52;
  for (let i = 0; i < 8; i++) t.box(root, cx(i), qz, 0.036, 0.03, 0.04, kit.blue);
  t.card(root, 0.13, qz - 0.01, 0.15, 0.05, () => [
    { text: "q", y: 0.45, size: 0.42, serif: true },
    { text: "the new token", y: 0.86, size: 0.28, color: t.faint },
  ]);
  t.card(root, 0.73, qz - 0.01, 0.11, 0.034, () => [{ text: "q · k", y: 0.74, size: 0.62, serif: true }]);
  t.card(root, 0.88, qz - 0.01, 0.12, 0.034, () => [{ text: "softmax", y: 0.74, size: 0.55 }]);
  const probs = (() => {
    const e = scores.map((s) => Math.exp(4 * s));
    const sum = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / sum);
  })();
  const maxP = Math.max(...probs);
  const rz = (r: number) => -0.4 + r * 0.055;
  const bars = scores.map((s, r) => {
    for (let i = 0; i < 8; i++) t.box(root, cx(i), rz(r), 0.036, 0.01, 0.04);
    const score = new THREE.Mesh(new THREE.BoxGeometry(1, 0.014, 0.03), kit.ink);
    score.castShadow = true;
    t.put(root, score, 0.67, rz(r), 0.007);
    const weight = new THREE.Mesh(new THREE.BoxGeometry(1, 0.016, 0.03), kit.blue);
    weight.castShadow = true;
    t.put(root, weight, 0.82, rz(r), 0.008);
    return { score, weight, s: s * 0.13, p: (probs[r] / maxP) * 0.13 };
  });
  t.card(root, 0.13, rz(4), 0.15, 0.05, () => [
    { text: "K", y: 0.45, size: 0.42, serif: true },
    { text: "cached keys", y: 0.86, size: 0.28, color: t.faint },
  ]);
  const sweep = t.outline(root, 0.36, 0.048, kit.pink);
  const oz = 0.2;
  const out = Array.from({ length: 8 }, (_, i) => {
    t.box(root, cx(i), oz, 0.036, 0.006, 0.04);
    return t.liquid(root, cx(i), oz, 0.03, 0.034, kit.pink);
  });
  t.card(root, 0.13, oz - 0.01, 0.15, 0.05, () => [
    { text: "out = Σ p · v", y: 0.5, size: 0.34, serif: true },
    { text: "weighted values", y: 0.88, size: 0.26, color: t.faint },
  ]);
  t.card(root, 0.53, 0.3, 0.84, 0.07, () => [
    { text: "32 query heads of 64 dimensions; 8 dims and 10 keys shown", y: 0.4, size: 0.32, color: t.faint },
    { text: "one query per sequence per step; the context grows by one token every step", y: 0.84, size: 0.32, color: t.faint },
  ]);
  const setBar = (mesh: THREE.Mesh, x0: number, length: number) => {
    mesh.visible = length > 0.0005;
    mesh.scale.x = Math.max(0.0001, length);
    mesh.position.x = x0 + length / 2;
  };
  const looper = looped(9, 7, (time) => {
    const fade = 1 - smooth(8.4, 9, time);
    const row = (time - 0.3) / 0.35;
    sweep.visible = row >= 0 && row < scores.length;
    sweep.position.set(0.3 + 3.5 * 0.042, 0.016, rz(Math.min(scores.length - 1, Math.max(0, Math.floor(row)))));
    bars.forEach((b, r) => {
      setBar(b.score, 0.645, b.s * smooth(0, 1, row - r) * fade);
      setBar(b.weight, 0.81, b.p * smooth(4.0, 5.0, time) * fade);
    });
    for (const o of out) t.level(o, 0.026 * smooth(5.2, 6.0, time) * fade, 0.006);
  });
  return { root, ...looper };
}

function blocksExhibit(kit: Kit, t: Tools) {
  const root = new THREE.Group();
  t.put(root, washPlane(t.blue, 1, 0.9, 0.12), 0.52, -0.2, 0.002);
  const lx = (k: number) => 0.18 + k * 0.135;
  const lz = -0.5;
  const tz = -0.33;
  t.card(root, 0.53, -0.62, 0.8, 0.036, () => [{ text: "this sequence: 96 tokens, 6 blocks of 16", y: 0.72, size: 0.56 }], -0.5);
  table.forEach((phys, k) => {
    t.box(root, lx(k), lz, 0.125, 0.018, 0.06);
    t.flat(root, lx(k), lz, 0.018, 0.1, 0.04, () => [{ text: `${k * 16}–${k * 16 + 15}`, y: 0.72, size: 0.56, serif: true, align: "center" }]);
    t.box(root, lx(k), tz, 0.125, 0.012, 0.055);
    t.flat(root, lx(k), tz, 0.012, 0.06, 0.04, () => [{ text: String(phys), y: 0.74, size: 0.62, serif: true, align: "center" }]);
  });
  t.card(root, 0.53, -0.44, 0.8, 0.03, () => [{ text: "block table: logical block → physical block", y: 0.74, size: 0.6, color: t.faint }], -0.5);
  t.card(root, 0.53, -0.28, 0.8, 0.03, () => [{ text: "KV pool on the GPU, shared by every request", y: 0.74, size: 0.6, color: t.faint }], -0.5);
  const tile = (id: number) => ({ x: 0.22 + (id % 4) * 0.2, z: -0.19 + Math.floor(id / 4) * 0.09 });
  for (let id = 0; id < 16; id++) {
    const { x, z } = tile(id);
    const h = others.has(id) ? 0.03 : 0.008;
    t.box(root, x, z, 0.18, h, 0.07);
    t.flat(root, x, z - 0.008, h, 0.1, 0.034, () => [{ text: `block ${id}`, y: 0.74, size: 0.58, color: table.includes(id) ? t.ink : t.faint, align: "center" }]);
  }
  const logicalRing = t.outline(root, 0.135, 0.07, kit.blue);
  const tableRing = t.outline(root, 0.135, 0.065, kit.blue);
  const poolRing = t.outline(root, 0.19, 0.08, kit.blue);
  const links = table.map((phys, k) => {
    const end = tile(phys);
    const from = new THREE.Vector3(lx(k), 0.014, tz + 0.03);
    const to = new THREE.Vector3(end.x, 0.012, end.z - 0.036);
    const mid = new THREE.Vector3((from.x + to.x) / 2, 0.16, (from.z + to.z) / 2);
    const tube = ribbonTube([from, mid, to], 0.0025, kit.blue);
    tube.visible = false;
    root.add(tube);
    return tube;
  });
  const row = new THREE.Mesh(new THREE.BoxGeometry(0.009, 0.012, 0.016), kit.pink);
  root.add(row);
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.022, 28, 22), kit.pink);
  knob.castShadow = true;
  root.add(knob);
  const xOf = (pos: number) => lx(0) - 0.0625 + ((pos + 0.5) / 96) * 0.81;
  let position = 37;
  let held = false;
  let shown = -1;
  const readout = t.card(root, 0.53, 0.2, 0.82, 0.11, () => {
    const p = Math.floor(position);
    const k = Math.floor(p / 16);
    return [
      { text: `token ${p} → table[${k}] = block ${table[k]}, row ${p % 16}`, y: 0.34, size: 0.27, serif: true, color: t.pink },
      { text: "key of token pos = pool[table[pos / 16]], row pos % 16", y: 0.62, size: 0.21 },
      { text: "raised blocks belong to other requests; blocks are handed out as a sequence grows", y: 0.88, size: 0.18, color: t.faint },
    ];
  });
  const apply = () => {
    const p = Math.floor(position);
    const k = Math.floor(p / 16);
    const end = tile(table[k]);
    knob.position.set(xOf(position), 0.045, lz);
    logicalRing.position.set(lx(k), 0.02, lz);
    tableRing.position.set(lx(k), 0.014, tz);
    const h = others.has(table[k]) ? 0.03 : 0.008;
    poolRing.position.set(end.x, h + 0.002, end.z);
    links.forEach((l, i) => {
      l.visible = i === k;
    });
    row.position.set(end.x - 0.08 + ((p % 16) + 0.5) * 0.01, h + 0.006, end.z + 0.024);
    if (p !== shown) {
      shown = p;
      readout.redraw();
    }
  };
  apply();
  return {
    root,
    knob,
    setHeld: (value: boolean) => {
      held = value;
    },
    drag: (x: number) => {
      position = Math.min(95, Math.max(0, ((x - xOf(0)) / 0.81) * 96 + 0.5));
      apply();
    },
    update: (dt: number, reduced: boolean) => {
      if (!held && !reduced) position = (position + dt * 4) % 96;
      apply();
    },
    reset: () => undefined,
  };
}

function versionsExhibit(kit: Kit, t: Tools, n: PagedNumbers): Exhibit {
  const root = new THREE.Group();
  t.put(root, washPlane(t.pink, 1, 0.5, 0.1), 0.52, -0.38, 0.002);
  t.put(root, washPlane(t.blue, 1, 0.5, 0.1), 0.52, -0.05, 0.002);
  t.card(root, 0.53, -0.62, 0.8, 0.06, () => [
    { text: "v1: every score of the context in shared memory, three passes", y: 0.42, size: 0.32 },
    { text: "the scores must fit, so the context stops at 2,048 tokens", y: 0.86, size: 0.29, color: t.faint },
  ]);
  const slots = 24;
  const sx = (i: number) => 0.14 + (i + 0.5) * (0.74 / slots);
  const az = -0.47;
  t.box(root, 0.51, az, 0.76, 0.008, 0.06);
  t.box(root, 0.895, az, 0.012, 0.05, 0.06);
  const cells = Array.from({ length: slots }, (_, i) => t.liquid(root, sx(i), az, 0.74 / slots - 0.006, 0.042, kit.blue));
  const sweeper = t.outline(root, 0.74 / slots, 0.056, kit.pink);
  let pass = 0;
  const passCard = t.card(root, 0.53, -0.41, 0.8, 0.032, () =>
    ["pass 1: scores", "pass 2: softmax", "pass 3: weighted values"].map((text, i) => ({ text, y: 0.74, size: 0.56, x: 0.02 + i * 0.33, color: pass === i + 1 ? t.pink : t.faint })),
    -0.5,
  );
  t.card(root, 0.53, -0.33, 0.8, 0.06, () => [
    { text: "v2: keys stream through in chunks, with an online softmax", y: 0.42, size: 0.32 },
    { text: "shared memory holds one chunk, so 8,192 tokens run", y: 0.86, size: 0.29, color: t.faint },
  ]);
  const bz = -0.2;
  t.box(root, 0.24, bz, 0.18, 0.008, 0.06);
  t.box(root, 0.145, bz, 0.01, 0.04, 0.06);
  t.box(root, 0.335, bz, 0.01, 0.04, 0.06);
  const stream = Array.from({ length: 14 }, () => {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.02, 0.04), kit.blue);
    mesh.castShadow = true;
    root.add(mesh);
    return mesh;
  });
  const jarR = 0.035;
  const jarH = 0.11;
  const jar = new THREE.Mesh(new THREE.CylinderGeometry(jarR, jarR, jarH, 32, 1, true), kit.glass);
  t.put(root, jar, 0.5, bz, jarH / 2);
  const rim: number[] = [];
  for (const y of [0, jarH]) {
    for (let k = 0; k < 32; k++) {
      const a = (k / 32) * Math.PI * 2;
      const b = ((k + 1) / 32) * Math.PI * 2;
      rim.push(0.5 + Math.cos(a) * jarR, y, bz + Math.sin(a) * jarR, 0.5 + Math.cos(b) * jarR, y, bz + Math.sin(b) * jarR);
    }
  }
  const rimGeometry = new THREE.BufferGeometry();
  rimGeometry.setAttribute("position", new THREE.Float32BufferAttribute(rim, 3));
  root.add(new THREE.LineSegments(rimGeometry, new THREE.LineBasicMaterial({ color: t.ink })));
  const sum = new THREE.Mesh(new THREE.CylinderGeometry(jarR * 0.92, jarR * 0.92, 1, 24), kit.pink);
  t.put(root, sum, 0.5, bz);
  t.card(root, 0.24, bz + 0.045, 0.2, 0.03, () => [{ text: "one chunk", y: 0.74, size: 0.6, color: t.faint, align: "center" }]);
  t.card(root, 0.5, bz + 0.045, 0.2, 0.03, () => [{ text: "running max m, sum l", y: 0.74, size: 0.5, color: t.faint, align: "center" }]);
  t.card(root, 0.8, bz + 0.045, 0.2, 0.03, () => [{ text: "keys, block by block", y: 0.74, size: 0.5, color: t.faint, align: "center" }]);
  const timeBars: [string, number, THREE.Material, string, number][] = [
    ["v1", n.stepV1, kit.pink, t.pink, 0.1],
    ["v2", n.stepV2, kit.blue, t.blue, 0.27],
  ];
  t.card(root, 0.53, -0.04, 0.8, 0.034, () => [{ text: "attention per decode step, eager profile, contexts of at most 64 tokens", y: 0.72, size: 0.46, color: t.faint }], -0.5);
  for (const [name, us, material, color, z] of timeBars) {
    t.card(root, 0.53, z - 0.1, 0.8, 0.045, () => [{ text: `${name}: ${us} µs`, y: 0.74, size: 0.6, serif: true, color }], -0.5);
    t.trough(root, 0.14, z, 0.76, material, us / n.stepV1);
  }
  const looper = looped(9, 6.5, (time) => {
    const fill = (time - 0.3) / 2;
    cells.forEach((c, i) => t.level(c, 0.018 * smooth(0, 1, fill * slots - i) * (1 - smooth(8.4, 9, time)), 0.008));
    const p2 = (time - 2.6) / 1.6;
    const p3 = (time - 4.4) / 1.6;
    const active = fill < 1 && fill >= 0 ? 1 : p2 >= 0 && p2 < 1 ? 2 : p3 >= 0 && p3 < 1 ? 3 : 0;
    sweeper.visible = active > 0;
    const along = active === 1 ? fill : active === 2 ? p2 : p3;
    sweeper.position.set(sx(Math.min(slots - 1, Math.floor(along * slots))), 0.03, az);
    if (active !== pass) {
      pass = active;
      passCard.redraw();
    }
    const flow = (time * 0.06) % 0.05;
    stream.forEach((s, i) => {
      const x = 0.97 - i * 0.05 - flow;
      const inTray = x < 0.33;
      s.visible = x > 0.15 && (x < 0.33 || x > 0.62);
      s.position.set(x, inTray ? 0.018 : 0.01, bz);
      s.scale.y = inTray ? 1 : 0.6;
    });
    t.level(sum, jarH * 0.85 * (1 - Math.exp(-time / 3)), 0.001);
  });
  return { root, ...looper };
}

function splitExhibit(kit: Kit, t: Tools, n: PagedNumbers): Exhibit {
  const root = new THREE.Group();
  t.put(root, washPlane(t.blue, 1, 0.9, 0.12), 0.52, -0.2, 0.002);
  let mode: "v2" | "split" = "v2";
  const header = t.card(root, 0.53, -0.62, 0.8, 0.06, () => [
    { text: mode === "v2" ? "v2: one block per row and head, 32 blocks at batch 1" : "8 splits × 32 heads = 256 blocks, then a merge", y: 0.42, size: 0.34, color: mode === "v2" ? t.ink : t.blue },
    { text: "lit pins: thread blocks running on the GPU (illustrative)", y: 0.86, size: 0.28, color: t.faint },
  ]);
  const cz = -0.48;
  const rx = (k: number) => 0.18 + k * 0.088;
  const ranges = Array.from({ length: 8 }, (_, k) => {
    t.box(root, rx(k), cz, 0.082, 0.008, 0.05);
    const fill = new THREE.Mesh(new THREE.BoxGeometry(1, 0.016, 0.036), kit.blue);
    fill.castShadow = true;
    return t.put(root, fill, rx(k), cz, 0.016);
  });
  t.card(root, 0.53, -0.44, 0.8, 0.03, () => [{ text: "one row's context of 2,048 tokens, cut into 8 ranges", y: 0.74, size: 0.56, color: t.faint }], -0.5);
  const mergeX = 0.86;
  const mz = -0.35;
  t.box(root, mergeX, mz, 0.06, 0.03, 0.04);
  t.card(root, 0.55, mz - 0.005, 0.5, 0.03, () => [{ text: "merge: each range's partial result, in split order", y: 0.74, size: 0.5, color: t.faint, align: "right", x: 0.98 }], -0.5);
  const partials = ranges.map(() => t.bead(root, kit.pink, 0.009));
  const pins: THREE.Mesh[] = [];
  const pinGeometry = new THREE.CylinderGeometry(0.0055, 0.0055, 1, 10);
  for (let s = 0; s < 26; s++) {
    const x = 0.15 + (s % 13) * 0.062;
    const z = -0.22 + Math.floor(s / 13) * 0.1;
    t.box(root, x, z, 0.056, 0.01, 0.088);
    for (let k = 0; k < 8; k++) {
      const pin = new THREE.Mesh(pinGeometry, kit.ink);
      pin.castShadow = true;
      pin.position.set(x - 0.012 + (k % 2) * 0.024, 0.01, z - 0.03 + Math.floor(k / 2) * 0.02);
      root.add(pin);
      pins.push(pin);
    }
  }
  const lit = (count: number) => {
    const chosen = new Set(Array.from({ length: count }, (_, i) => Math.floor((i * pins.length) / count)));
    pins.forEach((p, i) => {
      const on = chosen.has(i);
      p.material = on ? kit.blue : kit.ink;
      p.scale.y = on ? 0.03 : 0.008;
      p.position.y = 0.01 + p.scale.y / 2;
    });
  };
  const timeBars: [string, number, THREE.Material, string, number][] = [
    ["v2", n.v2Loop, kit.pink, t.pink, 0.17],
    ["8 splits, merge included", n.split8, kit.blue, t.blue, 0.33],
  ];
  for (const [name, us, material, color, z] of timeBars) {
    t.card(root, 0.53, z - 0.1, 0.8, 0.045, () => [{ text: `${name}: ${us} µs`, y: 0.74, size: 0.58, serif: true, color }], -0.5);
    t.trough(root, 0.14, z, 0.76, material, us / n.v2Loop);
  }
  const run = 4;
  const splitRun = (run * n.split8) / n.v2Loop;
  const half = 6;
  const setMode = (next: "v2" | "split") => {
    if (next === mode) return;
    mode = next;
    lit(mode === "v2" ? 32 : Math.round(pins.length * 0.85));
    header.redraw();
  };
  lit(32);
  const setRange = (k: number, f: number) => {
    const w = 0.07 * f;
    ranges[k].visible = f > 0.002;
    ranges[k].scale.x = Math.max(0.0001, w);
    ranges[k].position.x = rx(k) - 0.035 + w / 2;
  };
  const looper = looped(2 * half, 2 * half - 0.5, (time) => {
    setMode(time < half ? "v2" : "split");
    if (time < half) {
      const along = smooth(0.3, 0.3 + run, time) * 8;
      ranges.forEach((_, k) => setRange(k, Math.min(1, Math.max(0, along - k)) * (1 - smooth(half - 0.5, half, time))));
      partials.forEach((b) => {
        b.visible = false;
      });
    } else {
      const local = time - half;
      const f = smooth(0.3, 0.3 + splitRun * 0.8, local) * (1 - smooth(half - 0.4, half, local));
      ranges.forEach((_, k) => setRange(k, f));
      partials.forEach((b, k) => t.fly(b, { x: rx(k), y: 0.03, z: cz }, { x: mergeX, y: 0.04, z: mz }, smooth(0.3 + splitRun * 0.8, 0.3 + splitRun + 0.9, local), 0.06));
    }
  });
  return { root, ...looper };
}

export function createPagedAttentionModel(kit: Kit, fonts: Fonts, numbers: PagedNumbers): BookModel {
  const t = tools(kit, fonts);
  const blocks = blocksExhibit(kit, t);
  const all: Exhibit[] = [
    prefillExhibit(kit, t, numbers),
    decodeExhibit(kit, t, numbers),
    cacheExhibit(kit, t, numbers),
    homeExhibit(kit, t),
    jobExhibit(kit, t),
    blocks,
    versionsExhibit(kit, t, numbers),
    splitExhibit(kit, t, numbers),
  ];
  return {
    exhibits: all.map((e) => e.root),
    handles: [blocks.knob],
    dragStart: () => blocks.setHeld(true),
    drag: (x) => blocks.drag(x),
    dragEnd: () => blocks.setHeld(false),
    enter: (stop) => all[stop]?.reset(),
    update: (dt, exhibit, _stop, reduced) => all[exhibit]?.update(dt, reduced),
  };
}
