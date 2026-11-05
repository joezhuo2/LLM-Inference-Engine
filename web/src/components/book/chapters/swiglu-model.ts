import * as THREE from "three";
import { bf16, silu, sms } from "@/components/kernels/swiglu/figures";
import { createCard, type Kit, liquidMaterial, ribbonTube, washPlane } from "../ink";
import type { BookModel } from "../book-scene";
import type { Fonts } from "../typeset";

export type SwigluNumbers = {
  naive1: number;
  draft1: number;
  smallNaive: number;
  smallVector: number;
  smallFast: number;
  peakGbs: number;
  bandwidth: { label: string; gbs: number; accent: boolean }[];
};

const FF = 5632;
const VEC = 8;
const THREADS = 256;
const DRAWN = 16;

const placed = <T extends THREE.Object3D>(object: T, x: number, y: number, z: number): T => {
  object.position.set(x, y, z);
  return object;
};
const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const lines = (kit: Kit, points: number[], color = kit.palette.ink) => {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
  return new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color }));
};
const ulp = (v: number) => 2 ** (Math.floor(Math.log2(Math.abs(v))) - 7);
const fmt = (v: number, digits: number) => v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });

function rail(kit: Kit, root: THREE.Group, z: number, from: number, to: number, ticks: number) {
  const bar = kit.solid(new THREE.BoxGeometry(to - from, 0.018, 0.022));
  bar.position.set((from + to) / 2, 0.009, z);
  const marks: number[] = [];
  for (let i = 0; i <= ticks; i++) {
    const x = from + (i / ticks) * (to - from);
    marks.push(x, 0.019, z - 0.018, x, 0.019, z + 0.018);
  }
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.03, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(from, 0.05, z);
  root.add(bar, lines(kit, marks), knob);
  return knob;
}

function jobExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 1, 1.0, 0.12), 0.5, 0.002, -0.2));
  const pitch = 0.026;
  const rowZ = -0.52;
  const outZ = -0.33;
  const gateX = (j: number) => 0.075 + j * pitch + pitch / 2;
  const upX = (j: number) => 0.505 + j * pitch + pitch / 2;
  const outX = (j: number) => 0.5 + (j - DRAWN / 2 + 0.5) * pitch;
  const cellGeometry = new THREE.BoxGeometry(pitch - 0.004, 0.024, 0.04);
  const outCells: THREE.MeshLambertMaterial[] = [];
  for (let j = 0; j < DRAWN; j++) {
    root.add(placed(kit.solid(cellGeometry, { material: kit.blue, threshold: 40 }), gateX(j), 0.012, rowZ));
    root.add(placed(kit.solid(cellGeometry, { material: kit.pink, threshold: 40 }), upX(j), 0.012, rowZ));
    const material = kit.ink.clone();
    outCells.push(material);
    root.add(placed(kit.solid(cellGeometry, { material, threshold: 40 }), outX(j), 0.012, outZ));
  }
  const zips = Array.from({ length: DRAWN }, (_, j) => {
    const arc = (x: number, material: THREE.Material) => {
      const o = outX(j);
      return ribbonTube([new THREE.Vector3(x, 0.026, rowZ + 0.02), new THREE.Vector3((x * 2 + o) / 3, 0.07, rowZ + 0.07), new THREE.Vector3((x + o * 2) / 3, 0.06, outZ - 0.07), new THREE.Vector3(o, 0.026, outZ - 0.02)], 0.0016, material);
    };
    const pair = new THREE.Group();
    pair.add(arc(gateX(j), kit.blue), arc(upX(j), kit.pink));
    pair.visible = false;
    root.add(pair);
    return pair;
  });
  const label = (x: number, text: string, color: string) => {
    const card = createCard(kit, 0.4, 0.04, (ctx, w, h) => {
      ctx.fillStyle = color;
      ctx.font = `${h * 0.5}px ${fonts.sans}`;
      ctx.fillText(text, w * 0.02, h * 0.66);
    }, -0.5);
    card.group.position.set(x, 0, -0.588);
    root.add(card.group);
  };
  label(0.285, "gate: the first 5,632 of the row", kit.palette.blue);
  label(0.715, "up: the second 5,632", kit.palette.pink);
  const outCard = createCard(kit, 0.5, 0.04, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.5}px ${fonts.sans}`;
    ctx.fillText(`out: 5,632 values, ${DRAWN} drawn, into the down projection`, w * 0.02, h * 0.66);
  }, -0.5);
  outCard.group.position.set(0.5, 0, outZ + 0.04);
  root.add(outCard.group);

  const curveZ = -0.02;
  const gx = (g: number) => 0.1 + ((g + 5) / 10) * 0.8;
  const sy = (s: number) => 0.02 + s * 0.05;
  const points: THREE.Vector3[] = [];
  for (let k = 0; k <= 40; k++) {
    const g = -5 + k / 4;
    points.push(new THREE.Vector3(gx(g), sy(silu(g)), curveZ));
  }
  root.add(ribbonTube(points, 0.0028, kit.blue));
  root.add(lines(kit, [gx(-5), sy(0), curveZ, gx(5), sy(0), curveZ, gx(0), 0.004, curveZ, gx(0), sy(3), curveZ]));
  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.016, 22, 16), kit.blue);
  ball.castShadow = true;
  root.add(ball);
  const formula = createCard(kit, 0.3, 0.045, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.56}px ${fonts.serif}`;
    ctx.fillText("silu(g) = g / (1 + e⁻ᵍ)", w * 0.03, h * 0.7);
  }, -0.5);
  formula.group.position.set(0.25, 0, curveZ - 0.06);
  root.add(formula.group);

  const railZ = 0.12;
  const knob = rail(kit, root, railZ, 0.1, 0.9, 10);
  let g = 1.5;
  let shown = Number.NaN;
  const u = 2.25;
  const readout = createCard(kit, 0.8, 0.12, (ctx, w, h) => {
    const q = Math.round(g * 10) / 10;
    const s = bf16(silu(q));
    const out = bf16(s * u);
    ctx.font = `${h * 0.2}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.pink;
    ctx.fillText(`gate g = ${fmt(q, 1)}`, w * 0.02, h * 0.28);
    ctx.fillStyle = kit.palette.blue;
    ctx.fillText(`silu(g) → BF16 = ${fmt(s, 4)}`, w * 0.27, h * 0.28);
    ctx.fillStyle = kit.palette.ink;
    ctx.fillText(`× up ${u} → out = ${fmt(out, 4)}`, w * 0.62, h * 0.28);
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.17}px ${fonts.sans}`;
    ctx.fillText(q < -2.5 ? "a gate well below 0: silu is about 0, so this value is switched off" : q > 2.5 ? "a large gate: silu is about g, so up passes through, scaled" : "in between: the gate lets part of up through", w * 0.02, h * 0.64);
    ctx.fillText("per value: read 4 bytes, write 2, a few dozen FLOPs", w * 0.02, h * 0.9);
  });
  readout.group.position.set(0.5, 0, railZ + 0.065);
  root.add(readout.group);

  let held = false;
  let sweep = 0;
  let time = 0;
  const blue = new THREE.Color(kit.palette.blue);
  const paper = new THREE.Color(kit.palette.paper);
  const apply = () => {
    knob.position.x = gx(g);
    ball.position.set(gx(g), sy(silu(g)) + 0.016, curveZ);
    const q = Math.round(g * 10) / 10;
    if (q !== shown) {
      shown = q;
      readout.redraw();
    }
  };
  apply();
  return {
    root,
    knob,
    setHeld: (value: boolean) => {
      held = value;
      if (!value) sweep = Math.asin(Math.max(-1, Math.min(1, g / 4.5)));
    },
    drag: (x: number) => {
      g = Math.min(5, Math.max(-5, ((x - 0.1) / 0.8) * 10 - 5));
      apply();
    },
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 8 : (time + dt) % 9;
      const shownPairs = Math.min(DRAWN, Math.floor(time / 0.4));
      zips.forEach((pair, j) => {
        pair.visible = j < shownPairs;
        outCells[j].color.copy(paper).lerp(blue, smooth(j * 0.4 + 0.2, j * 0.4 + 0.9, time) * 0.75);
      });
      if (!held && !reduced) {
        sweep += dt * 0.35;
        g = 4.5 * Math.sin(sweep);
      }
      apply();
    },
  };
}

function bitsExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.pink, 1, 0.9, 0.1), 0.5, 0.002, -0.2));
  const g = 2;
  const u = 2.25;
  const s = silu(g);
  const kept = bf16(s);
  const keptProduct = Math.fround(kept * u);
  const right = bf16(keptProduct);
  const skipProduct = Math.fround(s * u);
  const wrong = bf16(skipProduct);

  const makeLine = (z: number, anchor: number, title: string) => {
    const step = ulp(anchor);
    const centers = [-1, 0, 1, 2].map((k) => anchor + k * step);
    const x = (v: number) => 0.1 + ((v - (anchor - 1.5 * step)) / (4 * step)) * 0.8;
    centers.forEach((c, k) => {
      const bin = kit.solid(new THREE.BoxGeometry(0.192, 0.02, 0.07));
      bin.position.set(x(c), 0.01, z);
      root.add(bin);
      const card = createCard(kit, 0.18, 0.05, (ctx, w, h) => {
        ctx.fillStyle = kit.palette.ink;
        ctx.font = `${h * 0.6}px ${fonts.sans}`;
        ctx.textAlign = "center";
        ctx.fillText(String(c), w / 2, h * 0.72);
      }, -0.6);
      card.group.position.set(x(c), 0, z + 0.05);
      root.add(card.group);
      if (k > 0) root.add(lines(kit, [x(c - step / 2), 0.021, z - 0.06, x(c - step / 2), 0.021, z + 0.05], kit.palette.faint));
    });
    const head = createCard(kit, 0.8, 0.05, (ctx, w, h) => {
      ctx.fillStyle = kit.palette.ink;
      ctx.font = `${h * 0.5}px ${fonts.sans}`;
      ctx.fillText(title, w * 0.01, h * 0.66);
    }, -0.5);
    head.group.position.set(0.5, 0, z - 0.13);
    root.add(head.group);
    return x;
  };
  const z1 = -0.4;
  const z2 = -0.1;
  const x1 = makeLine(z1, kept, `rounding 1: s = silu(${g}) = ${s.toPrecision(8)} in FP32, then BF16`);
  const x2 = makeLine(z2, right, `rounding 2: × up ${u}, then BF16`);

  const ball = (material: THREE.Material) => {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.018, 22, 16), material);
    mesh.castShadow = true;
    root.add(mesh);
    return mesh;
  };
  const keep = ball(kit.blue);
  const skip = ball(kit.pink);
  const result = createCard(kit, 0.82, 0.12, (ctx, w, h) => {
    ctx.font = `${h * 0.25}px ${fonts.serif}`;
    ctx.fillStyle = kit.palette.blue;
    ctx.fillText(`both roundings: ${right}, Hugging Face's bits`, w * 0.02, h * 0.36);
    ctx.fillStyle = kit.palette.pink;
    ctx.fillText(`skip rounding 1, multiply in FP32: ${wrong}, one step off`, w * 0.02, h * 0.76);
  });
  result.group.position.set(0.5, 0, 0.15);
  root.add(result.group);

  const hover = 0.08;
  const rest = 0.038;
  let time = 0;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 9 : (time + dt) % 11;
      const drop1 = smooth(1.2, 2.2, time);
      const slide1 = smooth(2.2, 3.2, time);
      const move = smooth(3.8, 5.2, time);
      const drop2 = smooth(5.6, 6.6, time);
      const slide2 = smooth(6.6, 7.6, time);
      const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
      const keepX1 = lerp(x1(s), x1(kept), slide1);
      const keepY1 = lerp(hover, rest, drop1);
      keep.position.set(lerp(keepX1, x2(keptProduct), move), lerp(lerp(keepY1, hover, move), rest, drop2), lerp(z1, z2, move));
      if (drop2 > 0) keep.position.x = lerp(x2(keptProduct), x2(right), slide2);
      skip.position.set(lerp(x1(s), x2(skipProduct), move), lerp(hover + 0.05, rest, drop2), lerp(z1, z2, move));
      if (drop2 > 0) skip.position.x = lerp(x2(skipProduct), x2(wrong), slide2);
    },
  };
}

function gridExhibit(kit: Kit, fonts: Fonts, numbers: SwigluNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 0.95, 0.8, 0.12), 0.45, 0.002, -0.2));
  const blocks = FF / THREADS;
  const rows = 4;
  const pitch = 0.03;
  const x0 = 0.07;
  const rowZ = (r: number) => -0.27 - r * 0.05;
  const flatPitch = (blocks * pitch) / (blocks * rows);
  const flatZ = -0.06;
  const tiles: { mesh: THREE.Mesh; from: THREE.Vector3; to: THREE.Vector3; row: number }[] = [];
  const pinkRow = liquidMaterial(kit.palette.pink);
  for (let r = 0; r < rows; r++) {
    for (let b = 0; b < blocks; b++) {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(pitch - 0.005, 1, 0.034), r % 2 ? pinkRow : kit.blue);
      mesh.castShadow = true;
      const i = r * blocks + b;
      tiles.push({ mesh, row: r, from: new THREE.Vector3(x0 + (b + 0.5) * pitch, 0, rowZ(r)), to: new THREE.Vector3(x0 + (i + 0.5) * flatPitch, 0, flatZ) });
      root.add(mesh);
    }
  }
  const plate = kit.solid(new THREE.BoxGeometry(blocks * pitch + 0.02, 0.008, rows * 0.05 + 0.01));
  plate.position.set(x0 + (blocks * pitch) / 2, 0.004, (rowZ(0) + rowZ(rows - 1)) / 2);
  root.add(plate);
  const dots: number[] = [];
  for (let k = 0; k < 3; k++) dots.push(x0 + 0.33, 0.01, rowZ(rows - 1) - 0.04 - k * 0.015, x0 + 0.33, 0.012, rowZ(rows - 1) - 0.045 - k * 0.015);
  root.add(lines(kit, dots));
  const ceiling = new THREE.Mesh(new THREE.BoxGeometry(blocks * pitch, 0.012, 0.01), kit.pink);
  ceiling.position.set(x0 + (blocks * pitch) / 2, 0.006, -0.5);
  root.add(ceiling);
  const ceilingCard = createCard(kit, 0.66, 0.042, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.pink;
    ctx.font = `${h * 0.48}px ${fonts.sans}`;
    ctx.fillText("gridDim.y ≤ 65,535: a batch of 70,000 rows would not launch", w * 0.01, h * 0.66);
  }, -0.5);
  ceilingCard.group.position.set(x0 + (blocks * pitch) / 2, 0, -0.585);
  root.add(ceilingCard.group);
  const gridCard = createCard(kit, 0.66, 0.042, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.48}px ${fonts.sans}`;
    ctx.fillText(`${blocks} blocks of ${THREADS} threads per row, one row per blockIdx.y`, w * 0.01, h * 0.66);
  }, -0.5);
  gridCard.group.position.set(x0 + (blocks * pitch) / 2, 0, -0.22);
  root.add(gridCard.group);
  const flatCard = createCard(kit, 0.66, 0.07, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.3}px ${fonts.sans}`;
    ctx.fillText(`the fix: one flat grid over rows × ${FF.toLocaleString("en-US")} values`, w * 0.01, h * 0.38);
    ctx.fillStyle = kit.palette.faint;
    ctx.fillText(`row = index / ${FF.toLocaleString("en-US")}; gridDim.x goes to 2³¹ − 1`, w * 0.01, h * 0.82);
  }, -0.5);
  flatCard.group.position.set(x0 + (blocks * pitch) / 2, 0, flatZ + 0.04);
  root.add(flatCard.group);

  const barMax = 0.2;
  const top = Math.max(numbers.naive1, numbers.draft1);
  [
    { x: 0.81, us: numbers.naive1, material: kit.ink },
    { x: 0.9, us: numbers.draft1, material: kit.blue },
  ].forEach(({ x, us, material }) => {
    const height = (us / top) * barMax;
    root.add(placed(kit.solid(new THREE.BoxGeometry(0.06, height, 0.06), { material }), x, height / 2, -0.36));
  });
  const barCard = createCard(kit, 0.17, 0.09, (ctx, w, h) => {
    ctx.font = `${h * 0.2}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.faint;
    ctx.fillText("one token, µs", w * 0.05, h * 0.25);
    ctx.fillStyle = kit.palette.ink;
    ctx.fillText(`naive ${fmt(numbers.naive1, 1)}`, w * 0.05, h * 0.55);
    ctx.fillStyle = kit.palette.blue;
    ctx.fillText(`draft ${fmt(numbers.draft1, 1)}`, w * 0.05, h * 0.85);
  });
  barCard.group.position.set(0.855, 0, -0.26);
  root.add(barCard.group);

  let time = 0;
  const point = new THREE.Vector3();
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 8 : (time + dt) % 12;
      const flatten = smooth(5, 6.6, time) * (1 - smooth(10.2, 11.8, time));
      for (const { mesh, from, to, row } of tiles) {
        const lit = smooth(0.3 + row * 0.6, 1.1 + row * 0.6, reduced ? 9 : time);
        const height = 0.006 + 0.018 * Math.max(lit, flatten);
        point.copy(from).lerp(to, flatten);
        mesh.position.set(point.x, height / 2 + 0.008 * (1 - flatten), point.z);
        mesh.scale.set(1 - flatten * (1 - flatPitch / pitch), height, 1);
      }
    },
  };
}

function vectorExhibit(kit: Kit, fonts: Fonts, numbers: SwigluNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 1, 0.9, 0.1), 0.5, 0.002, -0.2));
  const cellX = (k: number) => 0.1 + k * 0.034;
  const slotX = (k: number) => 0.62 + k * 0.034;
  const lanes = [-0.52, -0.33];
  const cube = new THREE.BoxGeometry(0.026, 0.026, 0.026);
  const movers = lanes.map((z) => {
    root.add(placed(kit.solid(new THREE.BoxGeometry(0.29, 0.012, 0.05)), 0.219, 0.006, z));
    root.add(placed(kit.solid(new THREE.BoxGeometry(0.29, 0.012, 0.05)), 0.739, 0.006, z));
    return Array.from({ length: VEC }, () => {
      const mesh = new THREE.Mesh(cube, kit.blue);
      mesh.castShadow = true;
      root.add(mesh);
      return mesh;
    });
  });
  const tray = new THREE.Group();
  const trayBox = new THREE.BoxGeometry(VEC * 0.034, 0.034, 0.038);
  tray.add(new THREE.LineSegments(new THREE.EdgesGeometry(trayBox), new THREE.LineBasicMaterial({ color: kit.palette.ink })));
  root.add(tray);
  const laneCard = (z: number, title: string, note: string, color: string) => {
    const card = createCard(kit, 0.8, 0.06, (ctx, w, h) => {
      ctx.fillStyle = color;
      ctx.font = `${h * 0.36}px ${fonts.sans}`;
      ctx.fillText(title, w * 0.01, h * 0.42);
      ctx.fillStyle = kit.palette.faint;
      ctx.font = `${h * 0.3}px ${fonts.sans}`;
      ctx.fillText(note, w * 0.01, h * 0.86);
    }, -0.5);
    card.group.position.set(0.5, 0, z + 0.05);
    root.add(card.group);
  };
  laneCard(lanes[0], "scalar: 8 gate loads of 2 bytes, one after another", "24 memory instructions per 8 outputs: 8 gate + 8 up loads, 8 stores", kit.palette.ink);
  laneCard(lanes[1], "vector: one 16-byte uint4 load brings all 8", "3 memory instructions per 8 outputs: 2 loads, 1 store", kit.palette.blue);

  const barZ = -0.08;
  const barMax = 0.17;
  const top = Math.max(numbers.peakGbs, ...numbers.bandwidth.map((b) => b.gbs)) * 1.05;
  const bars = numbers.bandwidth.map((b, k) => {
    const x = 0.28 + k * 0.22;
    const bar = kit.solid(new THREE.BoxGeometry(0.08, 1, 0.06), { material: b.accent ? kit.blue : kit.ink });
    bar.position.set(x, 0, barZ);
    root.add(bar);
    const card = createCard(kit, 0.18, 0.038, (ctx, w, h) => {
      ctx.fillStyle = b.accent ? kit.palette.blue : kit.palette.ink;
      ctx.font = `${h * 0.52}px ${fonts.sans}`;
      ctx.textAlign = "center";
      ctx.fillText(`${b.label} ${Math.round(b.gbs)} GB/s`, w / 2, h * 0.7);
    }, -0.5);
    card.group.position.set(x, 0, barZ + 0.06);
    root.add(card.group);
    return { bar, height: (b.gbs / top) * barMax };
  });
  const peakY = (numbers.peakGbs / top) * barMax;
  const peakGeometry = new THREE.BoxGeometry(0.68, 0.001, 0.1);
  const peak = new THREE.Mesh(peakGeometry, kit.glass);
  peak.position.set(0.5, peakY, barZ);
  const peakEdges = new THREE.LineSegments(new THREE.EdgesGeometry(peakGeometry), new THREE.LineDashedMaterial({ color: kit.palette.ink, dashSize: 0.01, gapSize: 0.006 }));
  peakEdges.position.copy(peak.position);
  peakEdges.computeLineDistances();
  root.add(peak, peakEdges);
  const peakCard = createCard(kit, 0.8, 0.07, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.3}px ${fonts.sans}`;
    ctx.fillText(`4,096 tokens: the same DRAM bytes ÷ each kernel's time`, w * 0.01, h * 0.38);
    ctx.fillStyle = kit.palette.faint;
    ctx.fillText(`the plane: the measured peak, ${numbers.peakGbs} GB/s`, w * 0.01, h * 0.82);
  }, -0.5);
  peakCard.group.position.set(0.5, 0, 0.12);
  root.add(peakCard.group);

  let time = 0;
  const arc = (mesh: THREE.Object3D, fromX: number, toX: number, z: number, t: number) => {
    mesh.position.set(fromX + (toX - fromX) * t, 0.026 + Math.sin(Math.PI * t) * 0.05, z);
  };
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 9.5 : (time + dt) % 10;
      movers[0].forEach((mesh, k) => arc(mesh, cellX(k), slotX(k), lanes[0], smooth(0.4 + k * 1.0, 1.3 + k * 1.0, time)));
      const together = smooth(0.4, 1.6, time);
      movers[1].forEach((mesh, k) => arc(mesh, cellX(k), slotX(k), lanes[1], together));
      arc(tray, (cellX(0) + cellX(VEC - 1)) / 2, (slotX(0) + slotX(VEC - 1)) / 2, lanes[1], together);
      const grow = smooth(0.4, 2.4, time);
      for (const { bar, height } of bars) {
        const h = Math.max(0.002, height * grow);
        bar.scale.y = h;
        bar.position.y = h / 2;
      }
    },
  };
}

function smExhibit(kit: Kit, fonts: Fonts, numbers: SwigluNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 1, 0.9, 0.1), 0.5, 0.002, -0.2));
  const columns = 13;
  const pitch = 0.06;
  const x0 = 0.5 - ((columns - 1) * pitch) / 2;
  const unitH = 0.014;
  const maxUnits = 8;
  const threshold = (sms * THREADS) / 2;
  const boards = [
    { z: -0.5, chains: VEC, perToken: FF / VEC, name: "vector" },
    { z: -0.22, chains: 1, perToken: FF, name: "scalar" },
  ].map((board) => {
    const smAt = (s: number) => new THREE.Vector3(x0 + (s % columns) * pitch, 0, board.z + Math.floor(s / columns) * 0.065);
    const tileGeometry = new THREE.BoxGeometry(0.05, 0.01, 0.05);
    for (let s = 0; s < sms; s++) {
      const p = smAt(s);
      root.add(placed(kit.solid(tileGeometry), p.x, 0.005, p.z));
    }
    const units = new THREE.InstancedMesh(new THREE.BoxGeometry(0.04, 1, 0.04), new THREE.MeshLambertMaterial({ color: "#ffffff" }), sms * maxUnits);
    units.castShadow = true;
    units.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(sms * maxUnits * 3), 3);
    root.add(units);
    let tokens = 1;
    const card = createCard(kit, 0.8, 0.07, (ctx, w, h) => {
      const work = tokens * board.perToken;
      const blocks = Math.ceil(work / THREADS);
      const picked = (board.chains > 1) === tokens * (FF / VEC) >= threshold;
      ctx.fillStyle = picked ? kit.palette.blue : kit.palette.faint;
      ctx.font = `${h * 0.36}px ${fonts.sans}`;
      ctx.fillText(`${board.name} kernel: ${work.toLocaleString("en-US")} ${board.chains > 1 ? "vectors" : "values"} = ${blocks} blocks, ${board.chains} chain${board.chains > 1 ? "s" : ""} per thread`, w * 0.01, h * 0.42);
      ctx.fillStyle = picked ? kit.palette.pink : kit.palette.faint;
      ctx.font = `${h * 0.3}px ${fonts.sans}`;
      ctx.fillText(picked ? "the dispatch takes this path" : `${Math.max(0, sms - blocks)} SMs idle`, w * 0.01, h * 0.86);
    }, -0.5);
    card.group.position.set(0.5, 0, board.z + 0.115);
    root.add(card.group);
    return { ...board, smAt, units, card, levels: new Array<number>(sms).fill(0), setTokens: (t: number) => { tokens = t; card.redraw(); } };
  });

  const railZ = 0.1;
  const knob = rail(kit, root, railZ, 0.15, 0.85, 7);
  let tokens = 1;
  const readout = createCard(kit, 0.8, 0.15, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.pink;
    ctx.font = `${h * 0.2}px ${fonts.serif}`;
    ctx.fillText(`${tokens} token${tokens > 1 ? "s" : ""}: ${(tokens * (FF / VEC)).toLocaleString("en-US")} vectors`, w * 0.02, h * 0.24);
    ctx.font = `${h * 0.14}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.faint;
    ctx.fillText(`the vector path needs ${sms} SMs × ${THREADS} / 2 = ${threshold.toLocaleString("en-US")} vectors`, w * 0.02, h * 0.48);
    ctx.fillStyle = kit.palette.ink;
    ctx.fillText(`one token, µs per launch: naive ${fmt(numbers.smallNaive, 2)}, vector ${fmt(numbers.smallVector, 2)}, dispatch ${fmt(numbers.smallFast, 2)}`, w * 0.02, h * 0.7);
    ctx.fillStyle = kit.palette.faint;
    ctx.fillText("each slab is one chain per thread; the tallest stack sets the time", w * 0.02, h * 0.92);
  });
  readout.group.position.set(0.5, 0, railZ + 0.075);
  root.add(readout.group);

  const setTokens = (t: number) => {
    tokens = t;
    knob.position.x = 0.15 + ((t - 1) / 7) * 0.7;
    readout.redraw();
    for (const board of boards) board.setTokens(t);
  };
  setTokens(1);

  const dummy = new THREE.Object3D();
  const blue = new THREE.Color(kit.palette.blue);
  const paper = new THREE.Color(kit.palette.paper);
  return {
    root,
    knob,
    drag: (x: number) => {
      const t = Math.min(8, Math.max(1, Math.round(1 + ((x - 0.15) / 0.7) * 7)));
      if (t !== tokens) setTokens(t);
    },
    update: (dt: number, reduced: boolean) => {
      const vectorPath = tokens * (FF / VEC) >= threshold;
      for (const board of boards) {
        const blocks = Math.ceil((tokens * board.perToken) / THREADS);
        const picked = (board.chains > 1) === vectorPath;
        for (let s = 0; s < sms; s++) {
          const target = Math.min(maxUnits, (Math.floor(blocks / sms) + (s < blocks % sms ? 1 : 0)) * board.chains);
          const level = board.levels[s];
          board.levels[s] = reduced ? target : level + (target - level) * Math.min(1, dt * 2.5);
          const p = board.smAt(s);
          for (let k = 0; k < maxUnits; k++) {
            const fill = Math.min(1, Math.max(0, board.levels[s] - k));
            const h = Math.max(0.0001, fill * (unitH - 0.003));
            dummy.position.set(p.x, 0.01 + k * unitH + h / 2, p.z);
            dummy.scale.set(fill > 0.01 ? 1 : 0, h, fill > 0.01 ? 1 : 0);
            dummy.updateMatrix();
            board.units.setMatrixAt(s * maxUnits + k, dummy.matrix);
            board.units.setColorAt(s * maxUnits + k, picked ? blue : paper);
          }
        }
        board.units.instanceMatrix.needsUpdate = true;
        if (board.units.instanceColor) board.units.instanceColor.needsUpdate = true;
      }
    },
  };
}

export function createSwigluModel(kit: Kit, fonts: Fonts, numbers: SwigluNumbers): BookModel {
  const job = jobExhibit(kit, fonts);
  const bits = bitsExhibit(kit, fonts);
  const grid = gridExhibit(kit, fonts, numbers);
  const vector = vectorExhibit(kit, fonts, numbers);
  const sm = smExhibit(kit, fonts, numbers);
  let active: THREE.Object3D | null = null;
  return {
    exhibits: [job.root, bits.root, grid.root, vector.root, sm.root],
    handles: [job.knob, sm.knob],
    dragStart: (handle) => {
      active = handle;
      if (handle === job.knob) job.setHeld(true);
    },
    drag: (x) => {
      if (active === job.knob) job.drag(x);
      if (active === sm.knob) sm.drag(x);
    },
    dragEnd: () => {
      if (active === job.knob) job.setHeld(false);
      active = null;
    },
    update: (dt, exhibit, _stop, reduced) => {
      if (exhibit === 0) job.update(dt, reduced);
      if (exhibit === 1) bits.update(dt, reduced);
      if (exhibit === 2) grid.update(dt, reduced);
      if (exhibit === 3) vector.update(dt, reduced);
      if (exhibit === 4) sm.update(dt, reduced);
    },
  };
}
