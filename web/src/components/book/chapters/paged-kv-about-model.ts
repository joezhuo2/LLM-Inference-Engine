import * as THREE from "three";
import { kvAbout, lowestFree } from "@/components/kernels/paged-kv/about";
import type { BookModel } from "../book-scene";
import { createCard, inkSegments, type Kit, ribbonTube, washPlane } from "../ink";
import type { Fonts } from "../typeset";

const B = kvAbout.block;
const OURS = lowestFree.slice(0, 4);
const MOST = OURS.length * B;
const PITCH = 0.0105;
const CELL = 0.0085;
const DEPTH = 0.02;
const H = 0.026;
const TRAY = B * PITCH + 0.006;
const COL_X = (c: number) => 0.13 + c * (TRAY + 0.03);
const ROW_Z = [-0.45, -0.3];
const STRIP_Z = -0.14;
const LIFT = 0.012;
const STEP_S = 0.95;
const HOLD_S = 3.5;

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const approach = (value: number, goal: number, rate: number, reduced: boolean) => (reduced ? goal : value + (goal - value) * Math.min(1, rate));
const fmt = (v: number) => v.toLocaleString("en-US");

type Pen = (text: string, x: number, y: number, size: number, color: string, serif?: boolean) => void;
const pen =
  (ctx: CanvasRenderingContext2D, fonts: Fonts, w: number, h: number): Pen =>
  (text, x, y, size, color, serif = false) => {
    ctx.fillStyle = color;
    ctx.font = `${h * size}px ${serif ? fonts.serif : fonts.sans}`;
    ctx.fillText(text, w * x, h * y);
  };

const where = (id: number) => {
  const i = kvAbout.pool.indexOf(id);
  return { x: COL_X(i % 4), z: ROW_Z[Math.floor(i / 4)] };
};

export function createPagedKvAboutModel(kit: Kit, fonts: Fonts): BookModel {
  const root = new THREE.Group();
  const wash = washPlane(kit.palette.blue, 1, 0.42, 0.12);
  wash.position.set(0.5, 0.002, -0.33);
  root.add(wash);

  const title = createCard(kit, 0.86, 0.11, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("KV cache: the Keys and Values of every earlier token", 0.03, 0.34, 0.22, kit.palette.ink, true);
    write(`paged: kept in blocks of ${B} tokens, found through a block table`, 0.03, 0.64, 0.22, kit.palette.ink, true);
    write(`one layer of ${kvAbout.layers} · one column per token, its key in front and its value behind`, 0.03, 0.9, 0.13, kit.palette.faint);
  });
  title.group.position.set(0.5, 0, -0.55);
  root.add(title.group);

  const trayLabels = new Map<number, ReturnType<typeof createCard>>();
  let held = 0;
  kvAbout.pool.forEach((id) => {
    const { x, z } = where(id);
    const tray = kit.solid(new THREE.BoxGeometry(TRAY, 0.004, DEPTH * 2 + 0.012));
    tray.position.set(x + TRAY / 2, 0.002, z);
    root.add(tray);
    const other = kvAbout.others.get(id);
    if (other !== undefined) {
      for (const side of [1, -1]) {
        const lane = kit.solid(new THREE.BoxGeometry(other * PITCH - (PITCH - CELL), H, DEPTH - 0.004));
        lane.position.set(x + 0.003 + (other * PITCH - (PITCH - CELL)) / 2, 0.004 + H / 2, z + (side * (DEPTH + 0.002)) / 2);
        root.add(lane);
      }
    }
    const ours = OURS.indexOf(id);
    const card = createCard(kit, TRAY, 0.042, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      write(`block ${id}`, 0.05, 0.46, 0.42, kit.palette.ink, true);
      const mine = ours >= 0 && held > ours;
      write(other !== undefined ? "other request" : mine ? `logical ${ours}` : "free", 0.05, 0.88, 0.34, other !== undefined ? kit.palette.faint : mine ? kit.palette.blue : kit.palette.faint);
    });
    card.group.position.set(x + TRAY / 2, 0, z + DEPTH + 0.035);
    root.add(card.group);
    trayLabels.set(id, card);
  });

  const cellGeo = new THREE.BoxGeometry(CELL, H, DEPTH - 0.004).translate(0, H / 2, 0);
  const tileGeo = new THREE.BoxGeometry(CELL, 0.006, 0.03).translate(0, 0.003, 0);
  const cellMesh = new THREE.InstancedMesh(cellGeo, kit.blue, MOST * 2);
  const tileMesh = new THREE.InstancedMesh(tileGeo, kit.blue, MOST);
  cellMesh.castShadow = true;
  const newestCells = [0, 1].map(() => new THREE.Mesh(cellGeo, kit.pink));
  const newestTile = new THREE.Mesh(tileGeo, kit.pink);
  root.add(cellMesh, tileMesh, ...newestCells, newestTile);
  const cells = Array.from({ length: MOST }, (_, t) => {
    const j = Math.floor(t / B);
    const { x, z } = where(OURS[j]);
    const cx = x + 0.003 + CELL / 2 + (t % B) * PITCH;
    return { cx, zs: [1, -1].map((side) => z + (side * (DEPTH + 0.002)) / 2), tx: COL_X(j) + 0.003 + CELL / 2 + (t % B) * PITCH, shown: 0 };
  });
  const matrix = new THREE.Matrix4();
  const place = (cx: number, y: number, z: number, s: number) => matrix.makeScale(1, s, 1).setPosition(cx, y, z);

  const strips = OURS.map((_, j) => {
    const plate = kit.solid(new THREE.BoxGeometry(TRAY, 0.004, 0.04));
    plate.position.set(COL_X(j) + TRAY / 2, 0.002, STRIP_Z);
    root.add(plate);
    const { x, z } = where(OURS[j]);
    const a = new THREE.Vector3(COL_X(j) + TRAY / 2, 0.012, STRIP_Z - 0.02);
    const b = new THREE.Vector3(x + TRAY / 2, 0.03, z + DEPTH);
    const mid = a.clone().lerp(b, 0.5);
    mid.y = 0.11;
    const arc = ribbonTube([a, a.clone().lerp(mid, 0.5).setY(0.08), mid, mid.clone().lerp(b, 0.5).setY(0.08), b], 0.0026, kit.blue);
    arc.visible = false;
    root.add(arc);
    return { arc, shown: 0 };
  });

  let tokens = 0;
  const tables = OURS.map((id, j) => {
    const card = createCard(kit, TRAY, 0.05, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      write(`logical ${j}: tokens ${j * B} to ${j * B + B - 1}`, 0.05, 0.4, 0.3, kit.palette.ink);
      const taken = held > j;
      write(taken ? `table → block ${id}` : "not taken yet", 0.05, 0.84, 0.34, taken ? kit.palette.blue : kit.palette.faint, true);
    });
    card.group.position.set(COL_X(j) + TRAY / 2, 0, STRIP_Z + 0.04);
    root.add(card.group);
    return card;
  });

  const readout = createCard(kit, 0.86, 0.11, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    const n = tokens;
    const blocks = Math.ceil(n / B);
    write(`${n} token${n === 1 ? "" : "s"} so far: ${n} key${n === 1 ? "" : "s"} and ${n} value${n === 1 ? "" : "s"} in this layer`, 0.03, 0.36, 0.28, kit.palette.blue, true);
    write(`${blocks} block${blocks === 1 ? "" : "s"} of ${B} held; the next one is taken at token ${blocks * B + 1}`, 0.03, 0.64, 0.21, kit.palette.ink);
    write(`all ${kvAbout.layers} layers: ${n} × ${fmt(kvAbout.tokenBytes)} bytes = ${fmt(n * kvAbout.tokenBytes)} bytes`, 0.03, 0.9, 0.21, kit.palette.ink);
  });
  readout.group.position.set(0.5, 0, 0.04);
  root.add(readout.group);

  const railZ = 0.2;
  const railX = [0.16, 0.84];
  const rail = kit.solid(new THREE.BoxGeometry(railX[1] - railX[0] + 0.04, 0.016, 0.02));
  rail.position.set(0.5, 0.008, railZ);
  const at = (n: number) => railX[0] + ((n - 1) / (MOST - 1)) * (railX[1] - railX[0]);
  const ticks: number[] = [];
  for (let n = B; n < MOST; n += B) ticks.push(at(n + 1), 0.017, railZ - 0.02, at(n + 1), 0.017, railZ + 0.02);
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.028, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(railX[0], 0.045, railZ);
  const railCard = createCard(kit, 0.68, 0.045, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write(`tokens so far, 1 to ${MOST}`, 0.02, 0.7, 0.48, kit.palette.faint);
    write(`a tick every ${B}: a new block`, 0.52, 0.7, 0.48, kit.palette.blue);
  });
  railCard.group.position.set(0.5, 0, railZ + 0.06);
  const note = createCard(kit, 0.86, 0.04, (ctx, w, h) => {
    pen(ctx, fonts, w, h)(`illustrative: a pool of ${kvAbout.pool.length} blocks; new blocks go to the lowest free id, as in the engine`, 0.03, 0.7, 0.5, kit.palette.faint);
  });
  note.group.position.set(0.5, 0, 0.36);
  root.add(rail, inkSegments(ticks, kit.thin), knob, railCard.group, note.group);

  const setTokens = (n: number) => {
    if (n === tokens) return;
    const blocks = Math.ceil(n / B);
    tokens = n;
    readout.redraw();
    if (blocks !== held) {
      held = blocks;
      tables.forEach((c) => c.redraw());
      OURS.forEach((id) => trayLabels.get(id)?.redraw());
    }
    read = 0;
  };

  let dragging = false;
  let knobX = railX[0];
  let clock = 0;
  let read = 10;
  let wasReduced = false;
  return {
    exhibits: [root],
    handles: [knob],
    dragStart: () => {
      dragging = true;
    },
    drag: (x) => {
      knobX = Math.min(railX[1], Math.max(railX[0], x));
      setTokens(1 + Math.round(((knobX - railX[0]) / (railX[1] - railX[0])) * (MOST - 1)));
    },
    dragEnd: () => {
      dragging = false;
      clock = (tokens - 1) * STEP_S;
    },
    enter: () => {
      clock = 0;
      setTokens(1);
    },
    update: (dt, _exhibit, _stop, reduced) => {
      if (reduced && !wasReduced && !dragging) setTokens(kvAbout.tokens);
      wasReduced = reduced;
      if (!reduced && !dragging) {
        clock = (clock + dt) % (MOST * STEP_S + HOLD_S);
        setTokens(Math.min(MOST, 1 + Math.floor(clock / STEP_S)));
      }
      read += dt;
      knob.position.x = dragging ? knobX : approach(knob.position.x, at(tokens), dt * 2.5, reduced);
      const wave = reduced ? 0 : smooth(0, 0.45, read) * (1 - smooth(0.45, 1.1, read));
      cells.forEach((c, t) => {
        c.shown = approach(c.shown, t < tokens ? 1 : 0, dt * 3.2, reduced);
        const newest = t === tokens - 1;
        const s = c.shown > 0.01 && !newest ? c.shown : 0.0001;
        const y = 0.004 + (!newest && t < tokens ? LIFT * wave : 0);
        c.zs.forEach((z, k) => cellMesh.setMatrixAt(t * 2 + k, place(c.cx, y, z, s)));
        tileMesh.setMatrixAt(t, place(c.tx, 0.004, STRIP_Z, s));
        if (newest) {
          const n = Math.max(0.0001, c.shown);
          newestCells.forEach((m, k) => {
            m.position.set(c.cx, 0.004, c.zs[k]);
            m.scale.set(1, n, 1);
          });
          newestTile.position.set(c.tx, 0.004, STRIP_Z);
          newestTile.scale.set(1, n, 1);
        }
      });
      cellMesh.instanceMatrix.needsUpdate = true;
      tileMesh.instanceMatrix.needsUpdate = true;
      for (const m of [...newestCells, newestTile]) m.visible = tokens > 0;
      strips.forEach((s, j) => {
        s.shown = approach(s.shown, j < held ? 1 : 0, dt * 2.4, reduced);
        s.arc.visible = s.shown > 0.02;
        s.arc.scale.set(1, Math.max(0.0001, s.shown), 1);
      });
    },
  };
}
