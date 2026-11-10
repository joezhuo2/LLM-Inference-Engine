import * as THREE from "three";
import { lengths, slots } from "@/components/kernels/serving/batching";
import { batchGrids } from "@/components/kernels/serving/figures";
import { queueBlocks, queueConfig, queueRequests, queueRun } from "@/components/kernels/serving/queue";
import { servingRanges } from "@/components/kernels/serving/story";
import { blockBytes, blockTokens, fmt, fullBlocks, fullContext, fullMb, heldBy, kvValues, maxBatch, poolBlocks, poolFraction, poolGb, reservedFit, sweepRows, tokenBytes, usableBlocks } from "@/content/kernels/serving";
import { kvReplay, modelShape } from "@/lib/data";
import { createCard, type Kit, washPlane } from "../ink";
import type { BookModel } from "../book-scene";
import type { Fonts } from "../typeset";

export const SERVING_STOP = { budget: 0, fit: 1, batch: 2, admission: 3, preempt: 4, static: 5, roomy: 6, tight: 7 };

const layers = modelShape().layers;
const sample = kvReplay().requests[0];
const clamp01 = (t: number) => Math.min(1, Math.max(0, t));
const smooth = (a: number, b: number, x: number) => {
  const t = clamp01((x - a) / (b - a));
  return t * t * (3 - 2 * t);
};
const placed = <T extends THREE.Object3D>(object: T, x: number, y: number, z: number): T => {
  object.position.set(x, y, z);
  return object;
};
const settle = (dt: number, reduced: boolean) => (reduced ? 1 : 1 - Math.exp(-dt / 0.28));

type Line = { text: string; y: number; size: number; color: string; serif?: boolean; x?: number };

function textCard(kit: Kit, fonts: Fonts, w: number, h: number, lines: () => Line[], tilt?: number) {
  return createCard(
    kit,
    w,
    h,
    (ctx, W, H) => {
      for (const l of lines()) {
        ctx.fillStyle = l.color;
        ctx.font = `${H * l.size}px ${l.serif ? fonts.serif : fonts.sans}`;
        ctx.fillText(l.text, W * (l.x ?? 0.03), H * l.y);
      }
    },
    tilt,
  );
}

function floorLabel(w: number, d: number, draw: (ctx: CanvasRenderingContext2D, W: number, H: number) => void) {
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(w * 3200);
  canvas.height = Math.round(d * 3200);
  const ctx = canvas.getContext("2d")!;
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 16;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, d), new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false }));
  mesh.rotation.x = -Math.PI / 2;
  const redraw = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    draw(ctx, canvas.width, canvas.height);
    texture.needsUpdate = true;
  };
  redraw();
  return { mesh, redraw };
}

function rail(kit: Kit, z: number, x0 = 0.14, x1 = 0.9) {
  const group = new THREE.Group();
  const bar = kit.solid(new THREE.BoxGeometry(x1 - x0, 0.014, 0.018));
  bar.position.set((x0 + x1) / 2, 0.007, z);
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.026, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(x0, 0.042, z);
  group.add(bar, knob);
  return {
    group,
    knob,
    set: (f: number) => {
      knob.position.x = x0 + clamp01(f) * (x1 - x0);
    },
    frac: (x: number) => clamp01((x - x0) / (x1 - x0)),
  };
}

function colors(kit: Kit) {
  return {
    blue: new THREE.Color(kit.palette.blue),
    pink: new THREE.Color(kit.palette.pink),
    ink: new THREE.Color(kit.palette.ink),
    faint: new THREE.Color(kit.palette.faint),
    paper: new THREE.Color(kit.palette.paper),
  };
}

function boxes(count: number) {
  const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshLambertMaterial({ color: "#ffffff" }), count);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
  const dummy = new THREE.Object3D();
  return {
    mesh,
    set: (i: number, x: number, y0: number, z: number, w: number, h: number, d: number, color: THREE.Color) => {
      const shown = h > 0.0002 && w > 0.0002;
      dummy.position.set(x, y0 + (shown ? h / 2 : 0), z);
      dummy.scale.set(shown ? w : 0.0001, shown ? h : 0.0001, shown ? d : 0.0001);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      mesh.setColorAt(i, color);
    },
    commit: () => {
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    },
  };
}

function trays(kit: Kit, cols: number, rows: number, count: number, x0: number, z0: number, px: number, pz: number) {
  const group = new THREE.Group();
  const plate = kit.solid(new THREE.BoxGeometry(cols * px + 0.014, 0.008, rows * pz + 0.014));
  plate.position.set(x0 + (cols * px) / 2, 0.004, z0 + (rows * pz) / 2);
  const center = (i: number): [number, number] => [x0 + ((i % cols) + 0.5) * px, z0 + (Math.floor(i / cols) + 0.5) * pz];
  const lines: number[] = [];
  for (let i = 0; i < count; i++) {
    const [cx, cz] = center(i);
    const hx = px * 0.42;
    const hz = pz * 0.42;
    const y = 0.0085;
    lines.push(cx - hx, y, cz - hz, cx + hx, y, cz - hz, cx + hx, y, cz - hz, cx + hx, y, cz + hz, cx + hx, y, cz + hz, cx - hx, y, cz + hz, cx - hx, y, cz + hz, cx - hx, y, cz - hz);
  }
  const outline = new THREE.BufferGeometry();
  outline.setAttribute("position", new THREE.Float32BufferAttribute(lines, 3));
  const fills = boxes(count);
  group.add(plate, new THREE.LineSegments(outline, new THREE.LineBasicMaterial({ color: kit.palette.faint })), fills.mesh);
  return {
    group,
    center,
    set: (i: number, h: number, color: THREE.Color, part = 1) => {
      const [cx, cz] = center(i);
      const w = px * 0.78 * clamp01(part);
      fills.set(i, cx - (px * 0.78 - w) / 2, 0.008, cz, w, h, pz * 0.78, color);
    },
    commit: fills.commit,
  };
}

function wrap(text: string, width: number) {
  const out: string[] = [];
  let line = "";
  for (const word of text.split(" ")) {
    if (line && line.length + word.length + 1 > width) {
      out.push(line);
      line = word;
    } else line = line ? `${line} ${word}` : word;
  }
  if (line) out.push(line);
  return out;
}

function budgetExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  const c = colors(kit);
  root.add(placed(washPlane(kit.palette.blue, 0.95, 0.5, 0.12), 0.52, 0.002, -0.26));
  const title = textCard(kit, fonts, 0.84, 0.1, () => [
    { text: `one token: 2 × ${layers} layers × ${kvValues} numbers × 2 bytes = ${fmt(tokenBytes)} bytes`, y: 0.36, size: 0.25, color: kit.palette.ink, serif: true },
    { text: `one block: ${blockTokens} tokens, ${fmt(blockBytes)} bytes; each tray below is one block, its height the tokens written`, y: 0.76, size: 0.17, color: kit.palette.faint },
  ]);
  title.group.position.set(0.52, 0, -0.57);
  root.add(title.group);
  const cols = 16;
  const rows = fullBlocks / cols;
  const pitch = 0.046;
  const gx = 0.52 - (cols * pitch) / 2;
  const grid = trays(kit, cols, rows, fullBlocks, gx, -0.47, pitch, pitch);
  root.add(grid.group);
  const level = new Float32Array(fullBlocks);

  let tokens = sample.prompt;
  let shown = -1;
  let held = false;
  let time = 0;
  const blocks = () => Math.ceil(tokens / blockTokens);
  const readout = textCard(kit, fonts, 0.84, 0.11, () => [
    { text: `${fmt(tokens)} tokens: ${blocks()} ${blocks() === 1 ? "block" : "blocks"}, ${fmt((blocks() * blockBytes) / 1e6, 1)} MB of cache`, y: 0.36, size: 0.27, color: kit.palette.blue, serif: true },
    { text: `ceil(${fmt(tokens)} / ${blockTokens}) = ${blocks()}; one more block each time the length passes a multiple of ${blockTokens}`, y: 0.64, size: 0.16, color: kit.palette.ink },
    { text: `the ${fullBlocks} trays are a full ${fmt(fullContext)}-token context, ${fmt(fullMb)} MB`, y: 0.88, size: 0.16, color: kit.palette.faint },
  ]);
  readout.group.position.set(0.52, 0, -0.04);
  root.add(readout.group);

  const px0 = 0.14;
  const px1 = 0.9;
  const bar = kit.solid(new THREE.BoxGeometry(px1 - px0, 0.008, 0.05));
  bar.position.set((px0 + px1) / 2, 0.004, 0.09);
  const fullW = Math.max(0.006, (fullBlocks / poolBlocks) * (px1 - px0));
  const full = new THREE.Mesh(new THREE.BoxGeometry(fullW, 0.012, 0.042), new THREE.MeshLambertMaterial({ color: c.pink }));
  full.position.set(px0 + fullW / 2, 0.014, 0.09);
  const mine = new THREE.Mesh(new THREE.BoxGeometry(1, 0.03, 0.03), kit.blue);
  mine.castShadow = true;
  root.add(bar, full, mine);
  const poolCard = textCard(kit, fonts, 0.84, 0.075, () => [
    { text: `the pool: ${fmt(poolBlocks)} blocks, ${fmt(poolGb, 2)} GB, ${poolFraction}% of the memory free after the weights`, y: 0.4, size: 0.26, color: kit.palette.ink, serif: true },
    { text: `pink: one full context, ${fullBlocks} blocks; blue: this request, to scale`, y: 0.82, size: 0.21, color: kit.palette.pink },
  ]);
  poolCard.group.position.set(0.52, 0, 0.16);
  root.add(poolCard.group);
  const knob = rail(kit, 0.29, px0, px1);
  const railCard = textCard(kit, fonts, 0.84, 0.04, () => [{ text: `the request's length, 1 to ${fmt(fullContext)} tokens`, y: 0.7, size: 0.5, color: kit.palette.faint }]);
  railCard.group.position.set(0.52, 0, 0.345);
  root.add(knob.group, railCard.group);

  const apply = (dt: number, reduced: boolean) => {
    const k = settle(dt, reduced);
    const n = blocks();
    for (let i = 0; i < fullBlocks; i++) {
      const fill = i < n - 1 ? 1 : i === n - 1 ? (tokens - i * blockTokens) / blockTokens : 0;
      level[i] += (fill * 0.05 - level[i]) * k;
      grid.set(i, level[i], c.blue);
    }
    grid.commit();
    const w = Math.max(0.003, (n / poolBlocks) * (px1 - px0));
    mine.scale.x = w;
    mine.position.set(px0 + w / 2, 0.035, 0.09);
    knob.set((tokens - 1) / (fullContext - 1));
    if (tokens !== shown) {
      shown = tokens;
      readout.redraw();
    }
  };
  return {
    root,
    knob: knob.knob,
    reset: () => {
      time = 0;
    },
    setHeld: (v: boolean) => {
      held = v;
    },
    drag: (x: number) => {
      tokens = Math.max(1, Math.round(1 + knob.frac(x) * (fullContext - 1)));
    },
    update: (dt: number, reduced: boolean) => {
      if (reduced && !held) tokens = fullContext / 2;
      else if (!held) {
        time = (time + dt) % 20;
        tokens = Math.round(sample.prompt + (fullContext - sample.prompt) * smooth(1, 16, time));
      }
      apply(dt, reduced);
    },
  };
}

function fitExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  const c = colors(kit);
  root.add(placed(washPlane(kit.palette.pink, 0.95, 0.3, 0.1), 0.52, 0.002, -0.44));
  root.add(placed(washPlane(kit.palette.blue, 0.95, 0.3, 0.12), 0.52, 0.002, -0.15));
  const cols = 20;
  const rows = Math.ceil(reservedFit / cols);
  const px = 0.038;
  const pz = 0.04;
  const gx = 0.52 - (cols * px) / 2;
  const top = trays(kit, cols, rows, reservedFit, gx, -0.535, px, pz);
  const low = trays(kit, cols, rows, reservedFit, gx, -0.245, px, pz);
  root.add(top.group, low.group);
  const topLevel = new Float32Array(reservedFit);
  const lowLevel = new Float32Array(reservedFit);

  let mode: "fit" | "batch" = "fit";
  let n = reservedFit;
  let pick = 0;
  let held = false;
  let time = 0;
  let shown = "";
  const fitMax = 256;
  const row = () => sweepRows[pick];
  const reserved = () => (mode === "fit" ? n : row().batch) * fullBlocks;
  const heldBlocks = () => (mode === "fit" ? heldBy(n) : row().peak);

  const topCard = textCard(kit, fonts, 0.84, 0.05, () => {
    const count = mode === "fit" ? n : row().batch;
    const over = reserved() > usableBlocks;
    return [{ text: mode === "fit" ? (over ? `reserving ${fmt(fullContext)} tokens each: ${reservedFit} fit, ${count - reservedFit} wait outside` : `reserving ${fmt(fullContext)} tokens each: ${count} of ${reservedFit} trays`) : over ? `worst case, every request at ${fmt(fullContext)} tokens: ${fmt(reserved())} blocks, more than the pool` : `worst case, every request at ${fmt(fullContext)} tokens: ${fmt(reserved())} blocks`, y: 0.68, size: 0.46, color: over ? kit.palette.pink : kit.palette.ink, serif: true }];
  });
  topCard.group.position.set(0.52, 0, -0.585);
  const lowCard = textCard(kit, fonts, 0.84, 0.05, () => [{ text: mode === "fit" ? `holding only what is written: ${fmt(heldBlocks())} blocks, ${fmt((100 * heldBlocks()) / poolBlocks, 1)}% of the pool` : `most blocks a measured run held: ${fmt(heldBlocks())}, ${fmt((100 * heldBlocks()) / poolBlocks, 1)}% of the pool`, y: 0.68, size: 0.46, color: kit.palette.blue, serif: true }]);
  lowCard.group.position.set(0.52, 0, -0.295);
  const readout = textCard(kit, fonts, 0.84, 0.1, () =>
    mode === "fit"
      ? [
          { text: `${n} ${n === 1 ? "request" : "requests"} running at once`, y: 0.36, size: 0.28, color: kit.palette.ink, serif: true },
          { text: `a tray is ${fullBlocks} blocks, one full context; the ${reservedFit} trays are the pool of ${fmt(poolBlocks)} blocks`, y: 0.64, size: 0.16, color: kit.palette.faint },
          { text: "lengths: the recorded chat run's, repeated; memory only, max_batch not counted", y: 0.88, size: 0.16, color: kit.palette.faint },
        ]
      : [
          { text: `max_batch ${row().batch}: ${row().tpot} ms a step, ${fmt(row().tokens)} tokens per second`, y: 0.36, size: 0.28, color: kit.palette.ink, serif: true },
          { text: `measured: twice max_batch requests at once, 64 prompt and 64 output tokens`, y: 0.64, size: 0.16, color: kit.palette.faint },
          { text: `at max_batch ${maxBatch} the worst case is ${fmt(maxBatch * fullBlocks)} blocks: the pool cannot run out`, y: 0.88, size: 0.16, color: kit.palette.blue },
        ],
  );
  readout.group.position.set(0.52, 0, 0.04);
  const knob = rail(kit, 0.2);
  const railCard = textCard(kit, fonts, 0.84, 0.045, () =>
    mode === "fit" ? [{ text: `requests, 1 to ${fitMax}`, y: 0.7, size: 0.46, color: kit.palette.faint }] : sweepRows.map((r, i): Line => ({ text: `${r.batch}`, y: 0.7, size: 0.5, color: kit.palette.pink, x: (0.04 + (i / (sweepRows.length - 1)) * 0.76) / 0.84 - 0.012 })),
  );
  railCard.group.position.set(0.52, 0, 0.26);
  root.add(topCard.group, lowCard.group, readout.group, knob.group, railCard.group);

  const apply = (dt: number, reduced: boolean) => {
    const k = settle(dt, reduced);
    const full = reserved() / fullBlocks;
    const part = heldBlocks() / fullBlocks;
    for (let i = 0; i < reservedFit; i++) {
      topLevel[i] += ((i < full ? 0.022 : 0) - topLevel[i]) * k;
      top.set(i, topLevel[i], c.pink);
      const fill = clamp01(part - i);
      lowLevel[i] += ((fill > 0 ? 0.045 : 0) - lowLevel[i]) * k;
      low.set(i, lowLevel[i], c.blue, fill > 0 ? Math.max(0.12, fill) : 1);
    }
    top.commit();
    low.commit();
    knob.set(mode === "fit" ? (n - 1) / (fitMax - 1) : pick / (sweepRows.length - 1));
    const key = `${mode}-${n}-${pick}`;
    if (key !== shown) {
      shown = key;
      topCard.redraw();
      lowCard.redraw();
      readout.redraw();
    }
  };
  return {
    root,
    knob: knob.knob,
    setMode: (next: "fit" | "batch") => {
      mode = next;
      time = 0;
      shown = "";
      railCard.redraw();
    },
    setHeld: (v: boolean) => {
      held = v;
    },
    drag: (x: number) => {
      if (mode === "fit") n = Math.max(1, Math.round(1 + knob.frac(x) * (fitMax - 1)));
      else pick = Math.round(knob.frac(x) * (sweepRows.length - 1));
    },
    update: (dt: number, reduced: boolean) => {
      if (!held) {
        time = (time + dt) % 22;
        if (mode === "fit") n = reduced ? reservedFit : Math.max(1, Math.round(1 + (fitMax - 1) * smooth(1, 17, time)));
        else pick = reduced ? 0 : Math.min(sweepRows.length - 1, Math.floor(time / (22 / sweepRows.length)));
      }
      apply(dt, reduced);
    },
  };
}

export type QueueMode = "lead" | "admission" | "preempt";

export function queueExhibit(kit: Kit, fonts: Fonts, lead: boolean) {
  const root = new THREE.Group();
  const c = colors(kit);
  root.add(placed(washPlane(kit.palette.blue, 0.95, 0.5, 0.12), 0.52, 0.002, -0.32));
  const waitZ = -0.49;
  const runZ = -0.31;
  const poolZ = -0.13;
  const waitX = (k: number) => 0.17 + k * 0.066;
  const runX = (k: number) => 0.235 + k * 0.19;
  const doneX = (k: number) => 0.66 + k * 0.05;
  const size = 0.05;
  const height = (tokens: number) => 0.014 + tokens * 0.0022;

  const header = (text: string, x: number, z: number, w: number) => {
    const card = textCard(kit, fonts, w, 0.036, () => [{ text, y: 0.72, size: 0.56, color: kit.palette.faint, x: 0.02 }]);
    card.group.position.set(x, 0, z);
    root.add(card.group);
  };
  header("waiting queue, front first", 0.33, waitZ - 0.075, 0.42);
  header("finished", 0.79, waitZ - 0.075, 0.3);
  header(`running list, max_batch ${queueConfig.maxBatch}, newest last`, 0.52, runZ - 0.075, 0.8);
  for (let k = 0; k < queueConfig.maxBatch; k++) {
    const tray = kit.solid(new THREE.BoxGeometry(0.16, 0.006, 0.075));
    tray.position.set(runX(k), 0.003, runZ);
    root.add(tray);
  }
  const lane = kit.solid(new THREE.BoxGeometry(0.44, 0.004, 0.07));
  lane.position.set(0.345, 0.002, waitZ);
  const out = kit.solid(new THREE.BoxGeometry(0.32, 0.004, 0.07));
  out.position.set(0.78, 0.002, waitZ);
  root.add(lane, out);

  const requests = queueRequests.map((r) => {
    const group = new THREE.Group();
    const material = new THREE.MeshLambertMaterial({ color: c.blue });
    const body = new THREE.Mesh(new THREE.BoxGeometry(size, 1, size), material);
    body.castShadow = true;
    group.add(body);
    const tag = createCard(
      kit,
      0.034,
      0.034,
      (ctx, W, H) => {
        ctx.fillStyle = kit.palette.ink;
        ctx.font = `${H * 0.7}px ${fonts.serif}`;
        ctx.textAlign = "center";
        ctx.fillText(r.name, W / 2, H * 0.76);
      },
      -1.1,
    );
    tag.group.position.set(0, 0, size / 2 + 0.016);
    group.add(tag.group);
    root.add(group);
    return { group, body, material, h: height(r.prompt), x: 0, z: 0, s: 1, placed: false };
  });

  let range = lead ? { from: 0, to: queueRun.length - 1, still: servingRanges.preempt.still } : servingRanges.admission;
  let at = range.from;
  let clock = 0;
  let held = false;
  let shown = -1;
  const cell = 0.76 / queueBlocks;
  const pool = trays(kit, queueBlocks, 1, queueBlocks, 0.14, poolZ - 0.03, cell, 0.06);
  root.add(pool.group);
  const poolLevel = new Float32Array(queueBlocks);
  const poolColor = Array.from({ length: queueBlocks }, () => c.blue.clone());
  const owners = floorLabel(0.76, 0.03, (ctx, W, H) => {
    const snap = queueRun[Math.round(at)];
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${H * 0.7}px ${fonts.sans}`;
    ctx.textAlign = "center";
    for (let k = 0; k < queueBlocks; k++) {
      const o = snap.owner[k + 1];
      if (o >= 0) ctx.fillText(queueRequests[o].name, ((k + 0.5) / queueBlocks) * W, H * 0.78);
    }
  });
  owners.mesh.position.set(0.52, 0.003, poolZ + 0.052);
  root.add(owners.mesh);

  const snapNow = () => queueRun[Math.round(at)];
  const names = (ids: number[]) => ids.map((id) => queueRequests[id].name).join(", ") || "none";
  const note = textCard(kit, fonts, 0.86, 0.15, () => {
    const snap = snapNow();
    const poolLine = `the pool: ${queueBlocks} blocks of ${queueConfig.blockSize} tokens, ${snap.free} free`;
    if (lead) {
      const upto = queueRun.slice(0, Math.round(at) + 1);
      return [
        { text: `step ${snap.step}: ${snap.kind === "start" ? "nothing has run yet" : snap.kind === "prefill" ? `prefill of ${names(snap.seqs)}` : `one token each for ${names(snap.seqs)}`}`, y: 0.26, size: 0.19, color: kit.palette.ink, serif: true },
        { text: `running ${names(snap.running)} · waiting ${names(snap.waiting)} · ${poolLine}`, y: 0.5, size: 0.125, color: kit.palette.blue },
        { text: `so far: ${upto.reduce((n, s) => n + s.admitted.length, 0)} admissions, ${upto.reduce((n, s) => n + s.seqs.length, 0)} tokens written, ${upto.reduce((n, s) => n + s.evicted.length, 0)} preemptions`, y: 0.72, size: 0.125, color: kit.palette.ink },
        { text: "pink: the request that has to give its blocks back", y: 0.92, size: 0.115, color: kit.palette.pink },
      ];
    }
    const text = wrap(snap.note, 92);
    return [
      { text: `step ${snap.step} · ${poolLine}`, y: 0.22, size: 0.17, color: kit.palette.ink, serif: true },
      ...text.slice(0, 4).map((line, k): Line => ({ text: line, y: 0.42 + k * 0.17, size: 0.115, color: snap.evicted.length ? kit.palette.pink : kit.palette.blue })),
    ];
  });
  note.group.position.set(0.52, 0, 0.05);
  const knob = rail(kit, 0.24);
  const railCard = textCard(kit, fonts, 0.86, 0.04, () => [{ text: `illustrative: blocks of ${queueConfig.blockSize} tokens and max_batch ${queueConfig.maxBatch}, run through a port of Scheduler::schedule(); a request's height is its tokens`, y: 0.7, size: 0.4, color: kit.palette.faint }]);
  railCard.group.position.set(0.52, 0, 0.3);
  root.add(note.group, knob.group, railCard.group);

  const apply = (dt: number, reduced: boolean) => {
    const snap = snapNow();
    const k = settle(dt, reduced);
    requests.forEach((q, id) => {
      const r = queueRequests[id];
      const w = snap.waiting.indexOf(id);
      const run = snap.running.indexOf(id);
      const d = snap.done.indexOf(id);
      const x = w >= 0 ? waitX(w) : run >= 0 ? runX(run) : doneX(Math.max(0, d));
      const z = run >= 0 ? runZ : waitZ;
      const s = w >= 0 || run >= 0 ? 1 : 0.7;
      const h = height(run >= 0 ? snap.cached[id] : r.prompt + snap.output[id]);
      if (!q.placed) {
        q.x = x;
        q.z = z;
        q.s = s;
        q.h = h;
        q.placed = true;
      }
      q.x += (x - q.x) * k;
      q.z += (z - q.z) * k;
      q.s += (s - q.s) * k;
      q.h += (h - q.h) * k;
      q.group.position.set(q.x, 0.006, q.z);
      q.group.scale.setScalar(q.s);
      q.body.scale.y = q.h;
      q.body.position.y = q.h / 2;
      q.material.color.copy(snap.evicted.includes(id) ? c.pink : run >= 0 ? c.blue : c.faint);
    });
    for (let b = 0; b < queueBlocks; b++) {
      const o = snap.owner[b + 1];
      if (o >= 0) poolColor[b].copy(c.blue);
      poolLevel[b] += ((o >= 0 ? 0.034 : 0) - poolLevel[b]) * k;
      pool.set(b, poolLevel[b], poolColor[b]);
    }
    pool.commit();
    knob.set(range.to === range.from ? 0 : (at - range.from) / (range.to - range.from));
    const whole = Math.round(at);
    if (whole !== shown) {
      shown = whole;
      note.redraw();
      owners.redraw();
    }
  };
  return {
    root,
    knob: knob.knob,
    setMode: (mode: QueueMode) => {
      range = mode === "lead" ? { from: 0, to: queueRun.length - 1, still: servingRanges.preempt.still } : servingRanges[mode];
      at = range.from;
      clock = 0;
      shown = -1;
    },
    setHeld: (v: boolean) => {
      held = v;
    },
    drag: (x: number) => {
      at = Math.round(range.from + knob.frac(x) * (range.to - range.from));
    },
    update: (dt: number, reduced: boolean) => {
      if (reduced && !held) at = range.still;
      else if (!held) {
        clock += dt;
        if (clock > 2.6) {
          clock = 0;
          at = at >= range.to ? range.from : at + 1;
        }
      }
      apply(dt, reduced);
    },
  };
}

type Kind = "static" | "roomy" | "tight";

function batchingExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  const c = colors(kit);
  root.add(placed(washPlane(kit.palette.blue, 0.95, 0.5, 0.12), 0.52, 0.002, -0.2));
  const kinds: Kind[] = ["static", "roomy", "tight"];
  const columns = Math.max(...kinds.map((k) => batchGrids[k].steps));
  const gx0 = 0.15;
  const gx1 = 0.92;
  const cw = (gx1 - gx0) / columns;
  const pitch = 0.07;
  const gz0 = -0.38;
  const rowZ = (s: number) => gz0 + (s + 0.5) * pitch;
  const plate = kit.solid(new THREE.BoxGeometry(gx1 - gx0 + 0.016, 0.006, slots * pitch + 0.01));
  plate.position.set((gx0 + gx1) / 2, 0.003, gz0 + (slots * pitch) / 2);
  root.add(plate);
  const cells = boxes(slots * columns);
  root.add(cells.mesh);
  const marks = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 16, 12), kit.pink, slots * columns);
  marks.castShadow = true;
  root.add(marks);
  const dummy = new THREE.Object3D();

  const qx0 = 0.34;
  const qPitch = (0.92 - qx0) / lengths.length;
  const qz = -0.53;
  const chips = lengths.map((len, id) => {
    const chip = kit.solid(new THREE.BoxGeometry(qPitch * 0.78, len * 0.011, 0.05));
    chip.children[0].position.y = (len * 0.011) / 2;
    chip.children[chip.children.length - 1].position.y = (len * 0.011) / 2;
    chip.position.set(qx0 + (id + 0.5) * qPitch, 0, qz);
    root.add(chip);
    return chip;
  });
  const chipLevel = lengths.map(() => 1);
  const chipLabels = floorLabel(0.92 - qx0, 0.04, (ctx, W, H) => {
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${H * 0.62}px ${fonts.sans}`;
    ctx.textAlign = "center";
    lengths.forEach((len, id) => ctx.fillText(`${id}:${len}`, ((id + 0.5) / lengths.length) * W, H * 0.72));
  });
  chipLabels.mesh.position.set((qx0 + 0.92) / 2, 0.003, qz + 0.048);
  const queueCard = textCard(kit, fonts, 0.2, 0.075, () => [
    { text: "the queue", y: 0.42, size: 0.36, color: kit.palette.ink, serif: true, x: 0.05 },
    { text: "request:steps, illustrative", y: 0.8, size: 0.2, color: kit.palette.faint, x: 0.05 },
  ]);
  queueCard.group.position.set(0.22, 0, qz + 0.005);
  root.add(chipLabels.mesh, queueCard.group);

  let kind: Kind = "static";
  let focus = false;
  let col = 0;
  let held = false;
  let shownCol = -1;
  const g = () => batchGrids[kind];

  const ids = floorLabel(gx1 - gx0 + 0.06, slots * pitch, (ctx, W, H) => {
    const grid = g();
    const upto = Math.min(Math.floor(col + 0.5), grid.steps);
    const sx = (x: number) => ((x - (gx0 - 0.06)) / (gx1 - gx0 + 0.06)) * W;
    ctx.textAlign = "center";
    ctx.font = `${H * 0.09}px ${fonts.sans}`;
    for (let s = 0; s < slots; s++) {
      const y = ((s + 0.62) / slots) * H;
      ctx.fillStyle = kit.palette.faint;
      ctx.fillText(`${s + 1}`, sx(gx0 - 0.03), y);
      ctx.fillStyle = kit.palette.paper;
      for (let t = 0; t < upto; t++) {
        const id = grid.cells[s][t];
        if (id !== null && !grid.redo[s][t]) ctx.fillText(`${id}`, sx(gx0 + (t + 0.5) * cw), y);
      }
    }
  });
  ids.mesh.position.set((gx0 - 0.06 + gx1) / 2, 0.034, gz0 + (slots * pitch) / 2);
  root.add(ids.mesh);

  const idle = (grid: ReturnType<typeof g>, upto: number) => {
    let n = 0;
    for (let s = 0; s < grid.slots; s++) for (let t = 0; t < upto; t++) if (grid.cells[s][t] === null || grid.redo[s][t]) n++;
    return n;
  };
  const stats = textCard(kit, fonts, 0.8, 0.1, () => {
    const grid = g();
    const upto = Math.min(Math.floor(col + 0.5), grid.steps);
    const running = upto ? grid.running[upto - 1] : 0;
    const out =
      kind === "static"
        ? [
            ["steps so far", `${upto}`, kit.palette.ink],
            ["slots in use", `${running}`, kit.palette.ink],
            ["wasted slot-steps", `${idle(grid, upto)}`, kit.palette.pink],
            ["steps to finish", `${grid.steps}`, kit.palette.faint],
          ]
        : [
            ["requests running", `${running}`, focus ? kit.palette.blue : kit.palette.ink],
            ["preemptions", `${grid.evictions.slice(0, upto).reduce((n, e) => n + e, 0)}`, focus ? kit.palette.blue : kit.palette.pink],
            ["wasted slot-steps", `${idle(grid, upto)}`, kit.palette.pink],
            ["static batching wasted", `${batchGrids.static.wasted}`, kit.palette.faint],
          ];
    return out.flatMap(([label, value, color], i): Line[] => [
      { text: label, y: 0.32, size: 0.17, color: kit.palette.faint, x: 0.03 + i * 0.245 },
      { text: value, y: 0.8, size: 0.42, color, serif: true, x: 0.03 + i * 0.245 },
    ]);
  });
  stats.group.position.set(0.53, 0, 0.0);
  root.add(stats.group);

  const knob = rail(kit, 0.17, gx0, gx1);
  const readout = textCard(kit, fonts, 0.36, 0.065, () => [
    { text: `step ${Math.min(Math.floor(col + 0.5), g().steps)} of ${g().steps}`, y: 0.48, size: 0.4, color: kit.palette.ink, serif: true, x: 0.04 },
    { text: "one column per step, 4 slots", y: 0.86, size: 0.2, color: kit.palette.faint, x: 0.04 },
  ]);
  readout.group.position.set(0.53, 0, 0.23);
  root.add(knob.group, readout.group);

  const apply = (dt: number, reduced: boolean) => {
    const grid = g();
    const upto = Math.min(col, grid.steps);
    for (let s = 0; s < slots; s++) {
      for (let t = 0; t < columns; t++) {
        const i = s * columns + t;
        const id = t < grid.steps ? grid.cells[s][t] : null;
        const redo = t < grid.steps && grid.redo[s][t];
        const grow = t < grid.steps ? smooth(t, t + 1, upto) : 0;
        const x = gx0 + (t + 0.5) * cw;
        if (id !== null && !redo) cells.set(i, x, 0.006, rowZ(s), cw * 0.86, 0.026 * grow, pitch * 0.78, c.blue);
        else if (redo) cells.set(i, x, 0.006, rowZ(s), cw * 0.86, 0.026 * grow, pitch * 0.78, c.pink);
        else cells.set(i, x, 0.006, rowZ(s), cw * 0.86, t < grid.steps ? 0.005 * grow : 0, pitch * 0.78, c.pink);
        const gone = t < grid.steps && grid.preempted[s].includes(t);
        dummy.position.set(x, 0.05, rowZ(s));
        dummy.scale.setScalar(gone ? 0.011 * grow + 0.0001 : 0.0001);
        dummy.updateMatrix();
        marks.setMatrixAt(i, dummy.matrix);
      }
    }
    cells.commit();
    marks.instanceMatrix.needsUpdate = true;
    const whole = Math.min(Math.floor(col + 0.5), grid.steps);
    const waiting = new Set(whole === 0 ? lengths.map((_, id) => id) : grid.waiting[whole - 1]);
    const k = settle(dt, reduced);
    chips.forEach((chip, id) => {
      chipLevel[id] += ((waiting.has(id) ? 1 : 0.02) - chipLevel[id]) * k;
      chip.scale.y = Math.max(0.001, chipLevel[id]);
    });
    knob.set(upto / grid.steps);
    if (whole !== shownCol) {
      shownCol = whole;
      ids.redraw();
      stats.redraw();
      readout.redraw();
    }
  };

  return {
    root,
    knob: knob.knob,
    setKind: (next: Kind, nextFocus: boolean) => {
      kind = next;
      focus = nextFocus;
      col = 0;
      shownCol = -1;
    },
    setHeld: (v: boolean) => {
      held = v;
    },
    drag: (x: number) => {
      col = knob.frac(x) * g().steps;
    },
    update: (dt: number, reduced: boolean) => {
      const steps = g().steps;
      if (reduced && !held) col = steps;
      else if (!held) col = col + dt / 0.7 > steps + 4 ? 0 : col + dt / 0.7;
      apply(dt, reduced);
    },
  };
}

export function createServingModel(kit: Kit, fonts: Fonts): BookModel {
  const budget = budgetExhibit(kit, fonts);
  const fit = fitExhibit(kit, fonts);
  const queue = queueExhibit(kit, fonts, false);
  const batching = batchingExhibit(kit, fonts);
  const draggable = [budget, fit, queue, batching];
  let active: (typeof draggable)[number] | null = null;
  let entered = -1;
  return {
    exhibits: [budget.root, fit.root, queue.root, batching.root],
    handles: draggable.map((d) => d.knob),
    dragStart: (handle) => {
      active = draggable.find((d) => d.knob === handle) ?? null;
      active?.setHeld(true);
    },
    drag: (x) => active?.drag(x),
    dragEnd: () => {
      active?.setHeld(false);
      active = null;
    },
    enter: (stop) => {
      if (stop === entered) return;
      entered = stop;
      if (stop === SERVING_STOP.budget) budget.reset();
      if (stop === SERVING_STOP.fit) fit.setMode("fit");
      if (stop === SERVING_STOP.batch) fit.setMode("batch");
      if (stop === SERVING_STOP.admission) queue.setMode("admission");
      if (stop === SERVING_STOP.preempt) queue.setMode("preempt");
      if (stop === SERVING_STOP.static) batching.setKind("static", false);
      if (stop === SERVING_STOP.roomy) batching.setKind("roomy", false);
      if (stop === SERVING_STOP.tight) batching.setKind("tight", false);
    },
    update: (dt, exhibit, _stop, reduced) => {
      if (exhibit === 0) budget.update(dt, reduced);
      if (exhibit === 1) fit.update(dt, reduced);
      if (exhibit === 2) queue.update(dt, reduced);
      if (exhibit === 3) batching.update(dt, reduced);
    },
  };
}
