import * as THREE from "three";
import { first, fullContext, heldAt, lender, lookup, reqs, writeShare, writeUs } from "@/components/kernels/paged-kv/figures";
import { stepAt } from "@/components/kernels/paged-kv/story";
import { kvReplay } from "@/lib/data";
import { createCard, type Kit, ribbonTube, washPlane } from "../ink";
import type { BookModel } from "../book-scene";
import type { Fonts } from "../typeset";

const replay = kvReplay();
const B = replay.meta.block_size;
const COLS = 12;
const ROWS = Math.ceil((replay.max_block + 1) / COLS);
const POOL = COLS * ROWS;

export const KV_STOP = { why: 0, pool: 1, write: 2, append: 3, reuse: 4, replay: 5 };

const fmt = (v: number, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
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
      const shown = h > 0.0002;
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

function poolGrid(kit: Kit, x0: number, z0: number, pitch: number) {
  const group = new THREE.Group();
  const plate = kit.solid(new THREE.BoxGeometry(COLS * pitch + 0.016, 0.008, ROWS * pitch + 0.016));
  plate.position.set(x0 + (COLS * pitch) / 2, 0.004, z0 + (ROWS * pitch) / 2);
  const center = (id: number): [number, number] => [x0 + ((id % COLS) + 0.5) * pitch, z0 + (Math.floor(id / COLS) + 0.5) * pitch];
  const half = pitch * 0.42;
  const lines: number[] = [];
  for (let id = 0; id < POOL; id++) {
    const [cx, cz] = center(id);
    const y = 0.0085;
    lines.push(cx - half, y, cz - half, cx + half, y, cz - half, cx + half, y, cz - half, cx + half, y, cz + half, cx + half, y, cz + half, cx - half, y, cz + half, cx - half, y, cz + half, cx - half, y, cz - half);
  }
  const outline = new THREE.BufferGeometry();
  outline.setAttribute("position", new THREE.Float32BufferAttribute(lines, 3));
  const fills = boxes(POOL);
  group.add(plate, new THREE.LineSegments(outline, new THREE.LineBasicMaterial({ color: kit.palette.faint })), fills.mesh);
  return {
    group,
    center,
    set: (id: number, h: number, color: THREE.Color) => {
      const [cx, cz] = center(id);
      fills.set(id, cx, 0.008, cz, pitch * 0.78, h, pitch * 0.78, color);
    },
    commit: fills.commit,
  };
}

function whyExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  const rows = reqs.slice(0, 4);
  const fullBlocks = fullContext / B;
  const reserved = rows.length * fullBlocks;
  const held = rows.reduce((n, q) => n + q.blocks.length, 0);
  const x0 = 0.32;
  const L = 0.6;
  const bp = L / Math.max(...rows.map((q) => q.blocks.length));
  const zoom = bp / ((B / fullContext) * L);
  root.add(placed(washPlane(kit.palette.pink, 0.9, 0.36, 0.1), 0.55, 0.002, -0.42));
  root.add(placed(washPlane(kit.palette.blue, 0.9, 0.34, 0.12), 0.55, 0.002, -0.0));
  const rowZ1 = (i: number) => -0.55 + i * 0.062;
  const rowZ2 = (i: number) => -0.12 + i * 0.062;
  const bars = rows.map((q, i) => {
    const trough = kit.solid(new THREE.BoxGeometry(L, 0.026, 0.04), { material: kit.glass });
    trough.position.set(x0 + L / 2, 0.013, rowZ1(i));
    const fill = new THREE.Mesh(new THREE.BoxGeometry(1, 0.02, 0.032), kit.blue);
    fill.castShadow = true;
    fill.position.set(x0, 0.011, rowZ1(i));
    root.add(trough, fill);
    return { fill, length: (q.final_length / fullContext) * L };
  });
  const blocks = rows.flatMap((q, i) =>
    q.blocks.map((_, j) => {
      const fillLevel = j < q.blocks.length - 1 ? 1 : (q.final_length - j * B) / B;
      const tray = kit.solid(new THREE.BoxGeometry(bp * 0.86, 0.006, 0.04));
      tray.position.set(x0 + (j + 0.5) * bp, 0.003, rowZ2(i));
      const fill = new THREE.Mesh(new THREE.BoxGeometry(bp * 0.8 * fillLevel, 0.022, 0.034), kit.blue);
      fill.castShadow = true;
      fill.position.set(x0 + j * bp + bp * 0.03 + (bp * 0.8 * fillLevel) / 2, 0.017, rowZ2(i));
      const group = new THREE.Group();
      group.add(tray, fill);
      root.add(group);
      return { group, at: 3.2 + (j / q.blocks.length) * 2.6 + i * 0.12 };
    }),
  );
  const rowLabels = (z: (i: number) => number, text: (q: (typeof rows)[number]) => string) => {
    const label = floorLabel(0.2, 0.25, (ctx, W, H) => {
      ctx.font = `${H * 0.07}px ${fonts.sans}`;
      rows.forEach((q, i) => {
        ctx.fillStyle = kit.palette.ink;
        ctx.fillText(text(q), 0, ((z(i) - z(0) + 0.03) / 0.25) * H + H * 0.03);
      });
    });
    label.mesh.position.set(0.21, 0.003, z(0) - 0.03 + 0.125);
    root.add(label.mesh);
  };
  rowLabels(rowZ1, (q) => `request ${q.id}: ${q.final_length} tokens`);
  rowLabels(rowZ2, (q) => `${q.blocks.length} blocks, ${q.blocks.length * B - q.final_length} spare`);
  const top = textCard(kit, fonts, 0.8, 0.08, () => [
    { text: `one contiguous row per request, sized for the full ${fmt(fullContext)} tokens`, y: 0.38, size: 0.26, color: kit.palette.ink },
    { text: `${fmt(reserved)} blocks of room reserved, the rest of each row never written`, y: 0.8, size: 0.24, color: kit.palette.pink },
  ]);
  top.group.position.set(0.52, 0, -0.27);
  const bottom = textCard(kit, fonts, 0.8, 0.08, () => [
    { text: `the same requests in blocks of ${B} tokens, taken as they grow (drawn ${Math.round(zoom)}× wider)`, y: 0.38, size: 0.24, color: kit.palette.ink },
    { text: `${held} blocks held: at most 15 slots unused per request`, y: 0.8, size: 0.24, color: kit.palette.blue },
  ]);
  bottom.group.position.set(0.52, 0, 0.16);
  root.add(top.group, bottom.group);

  let time = 0;
  return {
    root,
    reset: () => {
      time = 0;
    },
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 9 : (time + dt) % 11;
      const grow = smooth(0.3, 2.6, time);
      for (const b of bars) {
        const w = Math.max(0.0001, b.length * grow);
        b.fill.scale.x = w;
        b.fill.position.x = x0 + w / 2;
      }
      for (const b of blocks) b.group.scale.y = Math.max(0.001, smooth(b.at, b.at + 0.8, time));
    },
  };
}

function poolExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  const c = colors(kit);
  const table = first.blocks.map((b) => b.id);
  const tx0 = 0.14;
  const tp = 0.78 / table.length;
  const tz = -0.5;
  root.add(placed(washPlane(kit.palette.blue, 0.95, 0.75, 0.12), 0.53, 0.002, -0.2));
  table.forEach((_, j) => {
    const tile = kit.solid(new THREE.BoxGeometry(tp * 0.84, 0.02, 0.05));
    tile.position.set(tx0 + (j + 0.5) * tp, 0.01, tz);
    root.add(tile);
  });
  const tableText = floorLabel(0.78, 0.13, (ctx, W, H) => {
    ctx.textAlign = "center";
    table.forEach((id, j) => {
      const x = ((j + 0.5) / table.length) * W;
      ctx.fillStyle = kit.palette.faint;
      ctx.font = `${H * 0.15}px ${fonts.sans}`;
      ctx.fillText(`${j * B}–${j * B + B - 1}`, x, H * 0.2);
      ctx.fillStyle = kit.palette.ink;
      ctx.font = `${H * 0.24}px ${fonts.serif}`;
      ctx.fillText(`${id}`, x, H * 0.62);
    });
  });
  tableText.mesh.position.set(tx0 + 0.39, 0.0205, tz - 0.0156);
  root.add(tableText.mesh);

  const pool = poolGrid(kit, 0.14, -0.38, 0.065);
  root.add(pool.group);
  const others = new Set(
    heldAt(lookup.step)
      .filter((b) => b.req !== first.id)
      .map((b) => b.id),
  );
  for (let id = 0; id < POOL; id++) {
    if (id === 0) pool.set(id, 0.012, c.ink);
    else if (table.includes(id)) pool.set(id, 0.05, c.blue);
    else if (others.has(id)) pool.set(id, 0.024, c.faint);
    else pool.set(id, 0, c.paper);
  }
  pool.commit();

  const ribbon = (j: number, material: THREE.Material, radius: number) => {
    const [cx, cz] = pool.center(table[j]);
    const sx = tx0 + (j + 0.5) * tp;
    return ribbonTube([new THREE.Vector3(sx, 0.02, tz + 0.025), new THREE.Vector3((sx * 2 + cx) / 3, 0.12, tz + 0.06), new THREE.Vector3((sx + cx * 2) / 3, 0.1, (tz + cz) / 2 + 0.04), new THREE.Vector3(cx, 0.058, cz)], radius, material);
  };
  const inkRibbons = table.map((_, j) => ribbon(j, kit.ink, 0.0016));
  const pinkRibbons = table.map((_, j) => ribbon(j, kit.pink, 0.0028));
  root.add(...inkRibbons, ...pinkRibbons);
  const slab = new THREE.Mesh(new THREE.BoxGeometry(0.065 * 0.9, 0.004, 0.065 * 0.9), kit.pink);
  root.add(slab);

  let pos = lookup.pos;
  let held = false;
  let time = 0;
  let shown = -1;
  const at = () => {
    const p = Math.floor(pos);
    const k = Math.floor(p / B);
    return { p, k, phys: table[k], row: p % B, slot: table[k] * B + (p % B) };
  };
  const card = textCard(kit, fonts, 0.84, 0.18, () => {
    const { p, k, phys, row, slot } = at();
    return [
      { text: `request ${first.id}'s block table, then the pool at step ${lookup.step}`, y: 0.18, size: 0.12, color: kit.palette.faint },
      { text: `position ${p}: table[${p} / ${B}] = table[${k}] = ${phys}`, y: 0.42, size: 0.18, color: kit.palette.ink, serif: true },
      { text: `slot = ${phys} · ${B} + ${p} mod ${B} = ${slot}: row ${row} of block ${phys}`, y: 0.67, size: 0.18, color: kit.palette.pink, serif: true },
      { text: `blue: request ${first.id} · grey: other requests · ink: block 0, never handed out`, y: 0.9, size: 0.12, color: kit.palette.faint },
    ];
  });
  card.group.position.set(0.53, 0, 0.1);
  const knob = rail(kit, 0.32);
  root.add(card.group, knob.group);

  const apply = () => {
    const { p, k, phys, row } = at();
    const drawn = Math.floor((time - 0.3) / 0.18);
    inkRibbons.forEach((r, j) => {
      r.visible = j < drawn && j !== k;
    });
    pinkRibbons.forEach((r, j) => {
      r.visible = j === k && drawn >= table.length;
    });
    const [cx, cz] = pool.center(phys);
    slab.position.set(cx, 0.008 + ((row + 0.5) / B) * 0.05, cz);
    slab.visible = drawn >= table.length;
    knob.set(p / (first.final_length - 1));
    if (p !== shown) {
      shown = p;
      card.redraw();
    }
  };
  return {
    root,
    knob: knob.knob,
    reset: () => {
      time = 0;
      pos = lookup.pos;
    },
    setHeld: (v: boolean) => {
      held = v;
    },
    drag: (x: number) => {
      pos = knob.frac(x) * (first.final_length - 1);
      time = Math.max(time, 3);
      apply();
    },
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 9 : time + dt;
      if (!held && !reduced && time > 5) pos = (pos + dt * 3) % first.final_length;
      apply();
    },
  };
}

function writeExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  const { row, phys, slot } = lookup;
  root.add(placed(washPlane(kit.palette.blue, 0.9, 0.7, 0.12), 0.52, 0.002, -0.25));
  const hx = (h: number) => 0.1 + (h + 0.5) * 0.0205;
  const rz = -0.53;
  for (let h = 0; h < 40; h++) {
    const material = h < 32 ? kit.ink : h < 36 ? kit.blue : kit.pink;
    const height = h < 32 ? 0.022 : 0.032;
    const head = kit.solid(new THREE.BoxGeometry(0.017, height, 0.05), { material, threshold: 40 });
    head.position.set(hx(h), height / 2, rz);
    root.add(head);
  }
  const rowCard = textCard(
    kit,
    fonts,
    0.84,
    0.06,
    () => [
      { text: "one token's fused QKV row, 2,560 bf16 values:", y: 0.62, size: 0.34, color: kit.palette.ink },
      { text: "Q: 32 heads × 64", y: 0.62, size: 0.34, color: kit.palette.faint, x: 0.55 },
      { text: "K", y: 0.62, size: 0.34, color: kit.palette.blue, x: 0.84 },
      { text: "V", y: 0.62, size: 0.34, color: kit.palette.pink, x: 0.92 },
    ],
    -0.5,
  );
  rowCard.group.position.set(0.52, 0, rz + 0.06);
  root.add(rowCard.group);

  const tileW = 0.085;
  const tz0 = -0.34;
  const rp = 0.0125;
  const tileX = (cache: number, h: number) => (cache === 0 ? 0.12 : 0.54) + h * 0.095 + tileW / 2;
  const rowZ = (r: number) => tz0 + (r + 0.5) * rp;
  const flyers: { mesh: THREE.Mesh; from: THREE.Vector3; to: THREE.Vector3; at: number }[] = [];
  for (let cache = 0; cache < 2; cache++) {
    const material = cache === 0 ? kit.blue : kit.pink;
    for (let h = 0; h < 4; h++) {
      const x = tileX(cache, h);
      const base = kit.solid(new THREE.BoxGeometry(tileW + 0.008, 0.006, B * rp + 0.008));
      base.position.set(x, 0.003, tz0 + (B * rp) / 2);
      const lines: number[] = [];
      for (let r = 1; r < B; r++) lines.push(x - tileW / 2, 0.0065, tz0 + r * rp, x + tileW / 2, 0.0065, tz0 + r * rp);
      const grid = new THREE.BufferGeometry();
      grid.setAttribute("position", new THREE.Float32BufferAttribute(lines, 3));
      const earlier = new THREE.Mesh(new THREE.BoxGeometry(tileW * 0.94, 0.008, row * rp * 0.94), material);
      earlier.position.set(x, 0.01, tz0 + (row * rp) / 2);
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
      mesh.castShadow = true;
      root.add(base, new THREE.LineSegments(grid, new THREE.LineBasicMaterial({ color: kit.palette.faint })), earlier, mesh);
      flyers.push({ mesh, from: new THREE.Vector3(hx(32 + cache * 4 + h), 0.032, rz), to: new THREE.Vector3(x, 0.014, rowZ(row)), at: 0.8 + cache * 1.9 + h * 0.35 });
    }
  }
  const cacheCard = (cache: number) => {
    const card = textCard(kit, fonts, 0.38, 0.075, () => [
      { text: `${cache === 0 ? "K" : "V"} cache, block ${phys} of layer l`, y: 0.42, size: 0.32, color: cache === 0 ? kit.palette.blue : kit.palette.pink, serif: true },
      { text: `[KV head, ${B} rows, 64 dims]; row ${row} is slot ${slot}`, y: 0.82, size: 0.2, color: kit.palette.faint },
    ]);
    card.group.position.set(cache === 0 ? 0.31 : 0.73, 0, -0.08);
    root.add(card.group);
  };
  cacheCard(0);
  cacheCard(1);
  const note = textCard(kit, fonts, 0.84, 0.18, () => [
    { text: "256 threads a token, one per (token, KV head, dim): 4 · 64,", y: 0.2, size: 0.13, color: kit.palette.ink },
    { text: "each copying one K and one V value bit for bit", y: 0.4, size: 0.13, color: kit.palette.ink },
    { text: `index = ((block · 4 + head) · ${B} + row) · 64 + dim; slot -1 writes nothing`, y: 0.64, size: 0.14, color: kit.palette.ink, serif: true },
    { text: `22 launches per batch-1 decode step: ${fmt(writeUs, 1)} µs, ${writeShare}% of the kernel time`, y: 0.88, size: 0.14, color: kit.palette.blue, serif: true },
  ]);
  note.group.position.set(0.52, 0, 0.1);
  root.add(note.group);

  const small = new THREE.Vector3(0.017, 0.008, 0.05);
  const big = new THREE.Vector3(tileW * 0.94, 0.012, rp * 0.9);
  let time = 0;
  return {
    root,
    reset: () => {
      time = 0;
    },
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 8 : (time + dt) % 9;
      for (const f of flyers) {
        const t = smooth(f.at, f.at + 1.1, time);
        f.mesh.position.lerpVectors(f.from, f.to, t);
        f.mesh.position.y += Math.sin(Math.PI * t) * 0.14;
        f.mesh.scale.lerpVectors(small, big, smooth(0.6, 1, t));
      }
    },
  };
}

function appendExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  const c = colors(kit);
  const blocks = first.blocks.slice(0, 4);
  const start = first.prompt;
  const last = blocks.length * B - 1;
  const sx = (s: number) => 0.28 + (s + 0.5) * 0.039;
  const rowZ = (j: number) => -0.52 + j * 0.095;
  root.add(placed(washPlane(kit.palette.blue, 0.9, 0.45, 0.12), 0.55, 0.002, -0.38));
  const rows = blocks.map((b, j) => {
    const group = new THREE.Group();
    const tray = kit.solid(new THREE.BoxGeometry(B * 0.039 + 0.01, 0.006, 0.07));
    tray.position.set(0.28 + (B * 0.039) / 2, 0.003, rowZ(j));
    const label = floorLabel(0.13, 0.05, (ctx, W, H) => {
      ctx.fillStyle = kit.palette.ink;
      ctx.font = `${H * 0.5}px ${fonts.serif}`;
      ctx.fillText(`block ${b.id}`, 0, H * 0.66);
    });
    label.mesh.position.set(0.195, 0.003, rowZ(j));
    group.add(tray, label.mesh);
    root.add(group);
    return group;
  });
  const cells = boxes(blocks.length * B);
  root.add(cells.mesh);
  const allocations = first.blocks.length - first.blocks.filter((b) => b.step === first.admitted).length;
  let p = start;
  let held = false;
  let time = 0;
  let shown = -1;
  const phys = (q: number) => blocks[Math.floor(q / B)].id;
  const card = textCard(kit, fonts, 0.84, 0.22, () => {
    const q = Math.floor(p);
    const fresh = q >= 3 * B;
    return [
      { text: `request ${first.id}, ${start}-token prompt; position ${q}: slot = ${phys(q)} · ${B} + ${q % B} = ${phys(q) * B + (q % B)}`, y: 0.16, size: 0.12, color: kit.palette.ink, serif: true },
      { text: `position ${3 * B}: length % ${B} == 0, so append_slot calls allocate_block(),`, y: 0.36, size: 0.11, color: fresh ? kit.palette.pink : kit.palette.faint },
      { text: `which takes the lowest free block: ${blocks[3].id}, at step ${blocks[3].step}`, y: 0.52, size: 0.11, color: fresh ? kit.palette.pink : kit.palette.faint },
      { text: `${allocations} allocations in ${first.final_length - start} decode steps; every other step fills a slot it holds`, y: 0.72, size: 0.1, color: kit.palette.faint },
      { text: "low blue: prompt (prefill) · tall blue: decoded · pink: the newest token", y: 0.9, size: 0.1, color: kit.palette.faint },
    ];
  });
  card.group.position.set(0.53, 0, -0.07);
  const knob = rail(kit, 0.28);
  root.add(card.group, knob.group);

  const apply = (dt: number, reduced: boolean) => {
    const q = Math.floor(p);
    blocks.forEach((_, j) => {
      for (let s = 0; s < B; s++) {
        const n = j * B + s;
        const i = n;
        if (n < start) cells.set(i, sx(s), 0.006, rowZ(j), 0.032, 0.016, 0.056, c.blue);
        else if (n <= q) cells.set(i, sx(s), 0.006, rowZ(j), 0.032, 0.03, 0.056, n === q ? c.pink : c.blue);
        else cells.set(i, sx(s), 0.006, rowZ(j), 0.032, 0, 0.056, c.blue);
      }
    });
    cells.commit();
    const k = settle(dt, reduced);
    const target = q >= 3 * B ? 1 : 0.001;
    rows[3].scale.y = Math.max(0.001, rows[3].scale.y + (target - rows[3].scale.y) * k);
    rows[3].visible = rows[3].scale.y > 0.01;
    knob.set((p - start) / (last - start));
    if (q !== shown) {
      shown = q;
      card.redraw();
    }
  };
  return {
    root,
    knob: knob.knob,
    reset: () => {
      time = 0;
      p = start;
    },
    setHeld: (v: boolean) => {
      held = v;
    },
    drag: (x: number) => {
      p = start + knob.frac(x) * (last - start);
    },
    update: (dt: number, reduced: boolean) => {
      if (reduced && !held) p = last;
      else if (!held) {
        time += dt;
        const run = (last - start) / 2.2;
        const loop = time % (run + 3);
        p = start + Math.min(last - start, loop * 2.2);
      }
      apply(dt, reduced);
    },
  };
}

function reuseExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  const c = colors(kit);
  const lent = lender ?? reqs[1];
  const lentIds = lent.blocks.map((b) => b.id);
  const takers = reqs
    .filter((q) => q.id !== lent.id)
    .flatMap((q) => q.blocks.filter((b) => lentIds.includes(b.id) && b.step > lent.finished).map((b) => ({ q: q.id, ...b })))
    .sort((a, b) => a.step - b.step);
  const firstTake = lentIds.map((id) => takers.find((t) => t.id === id)).filter((t) => t !== undefined);
  const before = heldAt(Math.max(0, lent.finished - 1)).filter((b) => b.req !== lent.id);
  root.add(placed(washPlane(kit.palette.blue, 0.95, 0.5, 0.12), 0.53, 0.002, -0.36));
  const pool = poolGrid(kit, 0.14, -0.58, 0.065);
  root.add(pool.group);
  const level = new Float32Array(POOL);
  const card = textCard(kit, fonts, 0.84, 0.14, () => [
    { text: `request ${lent.id} finishes at step ${lent.finished}: blocks ${lentIds.join(" and ")} freed`, y: 0.27, size: 0.19, color: kit.palette.blue, serif: true },
    { text: firstTake.map((t) => `block ${t.id} to request ${t.q} at step ${t.step}`).join(", "), y: 0.57, size: 0.19, color: kit.palette.pink, serif: true },
    { text: `lowest free id first: the whole run stays in blocks 1 to ${replay.max_block}; grey: other requests`, y: 0.86, size: 0.14, color: kit.palette.faint },
  ]);
  card.group.position.set(0.53, 0, -0.13);
  root.add(card.group);

  const fullZ = 0.1;
  const fp = 0.78 / 14;
  const full = Array.from({ length: 14 }, (_, i) => {
    const tray = kit.solid(new THREE.BoxGeometry(fp * 0.86, 0.006, 0.05));
    tray.position.set(0.14 + (i + 0.5) * fp, 0.003, fullZ);
    const fill = new THREE.Mesh(new THREE.BoxGeometry(fp * 0.78, 1, 0.042), new THREE.MeshLambertMaterial({ color: i >= 11 ? c.pink : c.faint }));
    fill.castShadow = true;
    fill.position.set(0.14 + (i + 0.5) * fp, 0.006, fullZ);
    root.add(tray, fill);
    return fill;
  });
  const fullCard = textCard(kit, fonts, 0.84, 0.14, () => [
    { text: "a full pool: the newest request (pink) is preempted, its blocks", y: 0.27, size: 0.17, color: kit.palette.ink },
    { text: "freed, its prompt and output recomputed when there is room", y: 0.54, size: 0.17, color: kit.palette.ink },
    { text: "56 blocks, 6 preemptions: all 16 outputs equal the roomy run's", y: 0.85, size: 0.18, color: kit.palette.blue, serif: true },
  ]);
  fullCard.group.position.set(0.53, 0, 0.2);
  root.add(fullCard.group);

  let time = 0;
  return {
    root,
    reset: () => {
      time = 0;
    },
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 9.5 : (time + dt) % 11;
      const k = settle(dt, reduced);
      const freed = time > 2;
      for (let id = 0; id < POOL; id++) {
        let target = 0;
        let color = c.faint;
        if (id === 0) {
          target = 0.012;
          color = c.ink;
        } else if (lentIds.includes(id)) {
          const take = firstTake.findIndex((t) => t.id === id);
          const back = take >= 0 && time > 4 + take * 1.1;
          target = back ? 0.045 : freed ? 0 : 0.045;
          color = back ? c.pink : c.blue;
        } else if (before.some((b) => b.id === id)) target = 0.024;
        level[id] += (target - level[id]) * k;
        pool.set(id, level[id], color);
      }
      pool.commit();
      const gone = smooth(7, 7.9, time);
      full.forEach((fill, i) => {
        const h = Math.max(0.0001, 0.03 * (i >= 11 ? 1 - gone : 1));
        fill.scale.y = h;
        fill.position.y = 0.006 + h / 2;
      });
    },
  };
}

function replayExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  const c = colors(kit);
  root.add(placed(washPlane(kit.palette.blue, 0.95, 0.45, 0.12), 0.53, 0.002, -0.38));
  const pool = poolGrid(kit, 0.14, -0.58, 0.065);
  root.add(pool.group);
  const level = new Float32Array(POOL);
  const tEnd = replay.duration_s;
  const cx0 = 0.14;
  const cx1 = 0.9;
  const tx = (t: number) => cx0 + (t / tEnd) * (cx1 - cx0);
  const hy = (h: number) => (h / replay.peak.blocks) * 0.07;
  const shape = new THREE.Shape();
  shape.moveTo(cx0, 0);
  replay.steps.forEach((s, i) => {
    if (i % 3 === 0 || i === replay.steps.length - 1) shape.lineTo(tx(s[0]), hy(s[2]));
  });
  shape.lineTo(cx1, 0);
  shape.lineTo(cx0, 0);
  const area = new THREE.ExtrudeGeometry(shape, { depth: 0.03, bevelEnabled: false });
  const chart = kit.solid(area, { threshold: 50 });
  chart.position.z = 0.02;
  const cursor = new THREE.Mesh(new THREE.BoxGeometry(0.004, 0.1, 0.05), kit.pink);
  cursor.position.set(cx0, 0.05, 0.035);
  root.add(chart, cursor);

  let t = replay.peak.t;
  let held = false;
  let shown = -1;
  const stats = { step: 0, running: 0, blocks: 0, filled: 0 };
  const statCard = textCard(kit, fonts, 0.84, 0.08, () => [
    { text: `step ${stats.step}, ${replay.steps[stats.step][0].toFixed(2)} s into the run`, y: 0.38, size: 0.28, color: kit.palette.ink, serif: true },
    { text: `${stats.running} requests running · ${stats.blocks} blocks held of ${fmt(replay.meta.kv_blocks)} · ${Math.round(stats.filled)}% of their slots filled`, y: 0.82, size: 0.22, color: kit.palette.blue },
  ]);
  statCard.group.position.set(0.53, 0, -0.14);
  const chartCard = textCard(kit, fonts, 0.84, 0.06, () => [
    { text: `blocks held over the ${tEnd.toFixed(1)} s run, peak ${replay.peak.blocks}; blocks ${POOL} to ${fmt(replay.meta.kv_blocks - 1)} stay free all run`, y: 0.62, size: 0.3, color: kit.palette.faint },
  ]);
  chartCard.group.position.set(0.53, 0, 0.12);
  const knob = rail(kit, 0.27, cx0, cx1);
  root.add(statCard.group, chartCard.group, knob.group);

  const apply = (dt: number, reduced: boolean) => {
    const step = stepAt(Math.min(t, tEnd));
    const fills = new Map(heldAt(step).map((b) => [b.id, b.fill]));
    const k = settle(dt, reduced);
    for (let id = 0; id < POOL; id++) {
      const target = id === 0 ? 0.012 : (fills.get(id) ?? 0) * 0.05;
      level[id] += (target - level[id]) * k;
      pool.set(id, level[id], id === 0 ? c.ink : c.blue);
    }
    pool.commit();
    const x = tx(Math.min(t, tEnd));
    cursor.position.x = x;
    knob.set((x - cx0) / (cx1 - cx0));
    if (step !== shown) {
      shown = step;
      const list = heldAt(step);
      stats.step = step;
      stats.running = new Set(list.map((b) => b.req)).size;
      stats.blocks = replay.steps[step][2];
      stats.filled = list.length ? (100 * list.reduce((n, b) => n + b.fill, 0)) / list.length : 0;
      statCard.redraw();
    }
  };
  return {
    root,
    knob: knob.knob,
    reset: () => {
      t = 0;
    },
    setHeld: (v: boolean) => {
      held = v;
    },
    drag: (x: number) => {
      t = knob.frac(x) * tEnd;
    },
    update: (dt: number, reduced: boolean) => {
      if (reduced && !held) t = replay.peak.t;
      else if (!held) t = t + dt * 0.5 > tEnd + 1.5 ? 0 : t + dt * 0.5;
      apply(dt, reduced);
    },
  };
}

export function createPagedKvModel(kit: Kit, fonts: Fonts): BookModel {
  const why = whyExhibit(kit, fonts);
  const pool = poolExhibit(kit, fonts);
  const write = writeExhibit(kit, fonts);
  const append = appendExhibit(kit, fonts);
  const reuse = reuseExhibit(kit, fonts);
  const replayed = replayExhibit(kit, fonts);
  const draggable = [pool, append, replayed];
  let active: (typeof draggable)[number] | null = null;
  let entered = -1;
  return {
    exhibits: [why.root, pool.root, write.root, append.root, reuse.root, replayed.root],
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
      if (stop === KV_STOP.why) why.reset();
      if (stop === KV_STOP.pool) pool.reset();
      if (stop === KV_STOP.write) write.reset();
      if (stop === KV_STOP.append) append.reset();
      if (stop === KV_STOP.reuse) reuse.reset();
      if (stop === KV_STOP.replay) replayed.reset();
    },
    update: (dt, exhibit, _stop, reduced) => {
      if (exhibit === 0) why.update(dt, reduced);
      if (exhibit === 1) pool.update(dt, reduced);
      if (exhibit === 2) write.update(dt, reduced);
      if (exhibit === 3) append.update(dt, reduced);
      if (exhibit === 4) reuse.update(dt, reduced);
      if (exhibit === 5) replayed.update(dt, reduced);
    },
  };
}
