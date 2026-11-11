import * as THREE from "three";
import { mergePartials } from "@/components/kernels/split-attention/figures";
import type { SplitNumbers } from "@/components/kernels/split-attention/story";
import type { BookModel } from "../book-scene";
import { createCard, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";

export const SPLIT_STOP = { few: 0, cut: 1, scratch: 2, merge: 3, graphs: 4, waiting: 5, wave: 6 };

const placed = <T extends THREE.Object3D>(object: T, x: number, y: number, z: number): T => {
  object.position.set(x, y, z);
  return object;
};
const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const fmt = (n: number) => n.toLocaleString("en-US");

function occupancyExhibit(kit: Kit, fonts: Fonts, numbers: SplitNumbers) {
  const root = new THREE.Group();
  const { perSm, sms } = numbers;
  const capacity = perSm * sms;
  const smW = 0.064;
  const smD = 0.15;
  const gridX = 0.06;
  const rowZ = [-0.45, -0.28];
  const smCenter = (sm: number) => ({ x: gridX + (sm % 13) * 0.068 + smW / 2, z: rowZ[Math.floor(sm / 13)] });
  const slotAt = (j: number) => {
    const sm = j % sms;
    const k = Math.floor(j / sms) % perSm;
    const c = smCenter(sm);
    return { x: c.x + ((k % 2) - 0.5) * 0.03, z: c.z + (Math.floor(k / 2) - 1) * 0.045 };
  };
  root.add(placed(washPlane(kit.palette.blue, 1, 0.5, 0.14), 0.5, 0.002, -0.36));
  for (let sm = 0; sm < sms; sm++) {
    const c = smCenter(sm);
    const plate = kit.solid(new THREE.BoxGeometry(smW, 0.008, smD));
    plate.position.set(c.x, 0.004, c.z);
    root.add(plate);
  }
  const sockets: number[] = [];
  for (let j = 0; j < capacity; j++) {
    const p = slotAt(j);
    const hw = 0.012;
    const hd = 0.019;
    const y = 0.0085;
    sockets.push(p.x - hw, y, p.z - hd, p.x + hw, y, p.z - hd, p.x + hw, y, p.z - hd, p.x + hw, y, p.z + hd, p.x + hw, y, p.z + hd, p.x - hw, y, p.z + hd, p.x - hw, y, p.z + hd, p.x - hw, y, p.z - hd);
  }
  const socketGeometry = new THREE.BufferGeometry();
  socketGeometry.setAttribute("position", new THREE.Float32BufferAttribute(sockets, 3));
  root.add(new THREE.LineSegments(socketGeometry, new THREE.LineDashedMaterial({ color: kit.palette.faint, dashSize: 0.004, gapSize: 0.003 })).computeLineDistances());

  const titleCard = createCard(kit, 0.62, 0.04, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.5}px ${fonts.sans}`;
    ctx.fillText(`${sms} SMs × ${perSm} slots = ${capacity} blocks resident at once`, w * 0.01, h * 0.66);
  }, -0.5);
  titleCard.group.position.set(0.37, 0, -0.16);
  root.add(titleCard.group);

  const blocks = new THREE.InstancedMesh(new THREE.BoxGeometry(0.022, 1, 0.034), kit.blue, 256);
  blocks.castShadow = true;
  root.add(blocks);

  const railZ = 0.03;
  const stopX = [0.2, 0.4, 0.6, 0.8];
  const choices = numbers.waves;
  const rail = new THREE.Group();
  const bar = kit.solid(new THREE.BoxGeometry(0.7, 0.016, 0.02));
  bar.position.set(0.5, 0.008, railZ);
  const ticks: number[] = [];
  for (const x of stopX) ticks.push(x, 0.017, railZ - 0.02, x, 0.017, railZ + 0.02);
  const tickGeometry = new THREE.BufferGeometry();
  tickGeometry.setAttribute("position", new THREE.Float32BufferAttribute(ticks, 3));
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.028, 28, 22), kit.pink);
  knob.castShadow = true;
  const stopCard = createCard(kit, 0.78, 0.036, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.52}px ${fonts.sans}`;
    ctx.textAlign = "center";
    choices.forEach((c, i) => ctx.fillText(`${c.splits} split${c.splits > 1 ? "s" : ""}`, ((stopX[i] - 0.11) / 0.78) * w, h * 0.68));
    ctx.textAlign = "left";
  }, -0.6);
  stopCard.group.position.set(0.5, 0, railZ + 0.035);
  rail.add(bar, new THREE.LineSegments(tickGeometry, new THREE.LineBasicMaterial({ color: kit.palette.ink })), knob, stopCard.group);
  root.add(rail);

  let mode: "v2" | "split" = "v2";
  let choice = 3;
  let knobX = stopX[choice];
  let held = false;
  let elapsed = 0;
  const current = () => (mode === "v2" ? numbers.v2 : choices[choice]);
  const readout = createCard(kit, 0.66, 0.13, (ctx, w, h) => {
    const o = current();
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.15}px ${fonts.sans}`;
    ctx.fillText(mode === "v2" ? "v2: one block per row and query head, 1 row × 32 heads" : `split kernel: ${o.splits} split${o.splits > 1 ? "s" : ""} × 32 heads, each block ${fmt(2048 / o.splits)} tokens`, w * 0.03, h * 0.2);
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.3}px ${fonts.serif}`;
    ctx.fillText(`${o.blocks} blocks · ${o.waves} waves`, w * 0.03, h * 0.54);
    ctx.font = `${h * 0.15}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.blue;
    ctx.fillText(`warps active ${o.warps}%`, w * 0.03, h * 0.85);
    ctx.fillStyle = kit.palette.ink;
    ctx.fillText(`${elapsed.toFixed(1)} of ${o.us.toFixed(1)} µs`, w * 0.42, h * 0.85);
  });
  readout.group.position.set(0.5, 0, 0.22);
  root.add(readout.group);

  const dummy = new THREE.Object3D();
  const secondsPerMicro = 0.02;
  const launch = 0.9;
  const hop = 0.9;
  let time = 0;
  let shownClock = -1;

  const timeline = () => {
    const o = current();
    const run = o.us * secondsPerMicro;
    const two = o.blocks > capacity;
    return { o, run, two, total: launch + run + (two ? hop : 0) + 2.2 };
  };

  const layout = (reduced: boolean) => {
    const { o, run, two } = timeline();
    const t = reduced ? launch : time;
    const height = 0.1 / (o.splits === 1 ? 1 : o.splits);
    const grow = smooth(0, launch, t);
    const firstRun = two ? run / 2 : run;
    const left1 = 1 - smooth(launch, launch + firstRun, t);
    const hopT = smooth(launch + firstRun, launch + firstRun + hop, t);
    const left2 = 1 - smooth(launch + firstRun + hop, launch + run + hop, t);
    for (let i = 0; i < 256; i++) {
      let scale = 0.0001;
      let x = 0;
      let z = 0;
      let y = 0.009;
      if (i < o.blocks) {
        if (i < capacity) {
          const p = slotAt(i);
          x = p.x;
          z = p.z;
          scale = Math.max(0.0001, height * grow * left1);
        } else {
          const q = i - capacity;
          const from = { x: 0.07 + q * 0.0086, z: -0.085 };
          const to = slotAt(q);
          x = from.x + (to.x - from.x) * hopT;
          z = from.z + (to.z - from.z) * hopT;
          y = 0.009 + Math.sin(hopT * Math.PI) * 0.06;
          scale = Math.max(0.0001, height * grow * (hopT < 1 ? 1 : left2));
        }
      }
      const shrink = i >= capacity ? 1 - 0.65 * (1 - hopT) : 1;
      dummy.position.set(x, y + scale / 2, z);
      if (i < o.blocks) dummy.scale.set(shrink, scale, shrink);
      else dummy.scale.set(0.0001, 0.0001, 0.0001);
      dummy.updateMatrix();
      blocks.setMatrixAt(i, dummy.matrix);
    }
    blocks.instanceMatrix.needsUpdate = true;
    const clock = reduced ? o.us : Math.min(o.us, Math.max(0, (t - launch - (two && t > launch + firstRun ? Math.min(hop, t - launch - firstRun) : 0)) / secondsPerMicro));
    const rounded = Math.round(clock * 10) / 10;
    if (rounded !== shownClock) {
      shownClock = rounded;
      elapsed = rounded;
      readout.redraw();
    }
  };

  const setChoice = (next: number) => {
    if (next === choice && mode === "split") return;
    choice = next;
    time = 0;
    shownClock = -1;
  };

  return {
    root,
    knob,
    setMode: (next: "v2" | "split") => {
      mode = next;
      rail.visible = next === "split";
      time = 0;
      shownClock = -1;
    },
    setHeld: (value: boolean) => {
      held = value;
      if (!value) knobX = stopX[choice];
    },
    drag: (x: number) => {
      knobX = Math.min(stopX[3], Math.max(stopX[0], x));
      const nearest = stopX.reduce((best, s, i) => (Math.abs(s - knobX) < Math.abs(stopX[best] - knobX) ? i : best), 0);
      setChoice(nearest);
    },
    update: (dt: number, reduced: boolean) => {
      const target = held ? knobX : stopX[choice];
      knob.position.set(knob.position.x + (target - knob.position.x) * Math.min(1, dt * 8), 0.044, railZ);
      if (!reduced) time = (time + dt) % timeline().total;
      layout(reduced);
    },
    init: () => {
      knob.position.set(stopX[choice], 0.044, railZ);
      rail.visible = false;
      layout(false);
    },
  };
}

function waitingExhibit(kit: Kit, fonts: Fonts, numbers: SplitNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.pink, 0.8, 0.45, 0.1), 0.38, 0.002, -0.37));
  const laneX = 0.06;
  const laneLength = 0.6;
  const loadW = 0.02;
  const waitW = 0.12;
  const mathW = 0.016;
  const period = loadW + waitW + mathW;
  const lanes = [-0.48, -0.41, -0.34, -0.27];
  const phases = [0, 0.047, 0.098, 0.021];
  const maths: { mesh: THREE.Mesh; from: number; to: number }[] = [];
  const mathMaterial = (hot: boolean) => (hot ? kit.pink : kit.blue);
  lanes.forEach((z, lane) => {
    const track = kit.solid(new THREE.BoxGeometry(laneLength, 0.004, 0.034), { shadow: false });
    track.position.set(laneX + laneLength / 2, 0.002, z);
    root.add(track);
    for (let start = -period + phases[lane]; start < laneLength; start += period) {
      const segments: [number, number, "load" | "wait" | "math"][] = [
        [start, start + loadW, "load"],
        [start + loadW, start + loadW + waitW, "wait"],
        [start + loadW + waitW, start + period, "math"],
      ];
      for (const [a0, b0, kind] of segments) {
        const a = Math.max(0, a0);
        const b = Math.min(laneLength, b0);
        if (b - a < 0.002) continue;
        const cx = laneX + (a + b) / 2;
        if (kind === "load") {
          const box = kit.solid(new THREE.BoxGeometry(b - a - 0.002, 0.026, 0.026));
          box.position.set(cx, 0.017, z);
          root.add(box);
        } else if (kind === "wait") {
          const box = new THREE.Mesh(new THREE.BoxGeometry(b - a - 0.002, 0.012, 0.022), kit.glass);
          box.position.set(cx, 0.01, z);
          const dashes: number[] = [];
          for (let x = a + 0.004; x < b - 0.006; x += 0.012) dashes.push(laneX + x, 0.017, z, laneX + x + 0.006, 0.017, z);
          const dashGeometry = new THREE.BufferGeometry();
          dashGeometry.setAttribute("position", new THREE.Float32BufferAttribute(dashes, 3));
          root.add(box, new THREE.LineSegments(dashGeometry, new THREE.LineBasicMaterial({ color: kit.palette.faint })));
        } else {
          const mesh = new THREE.Mesh(new THREE.BoxGeometry(b - a - 0.002, 0.03, 0.026), mathMaterial(false));
          mesh.position.set(cx, 0.019, z);
          mesh.castShadow = true;
          root.add(mesh);
          maths.push({ mesh, from: laneX + a, to: laneX + b });
        }
      }
    }
  });
  const titleCard = createCard(kit, 0.6, 0.036, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.52}px ${fonts.sans}`;
    ctx.fillText("4 warps, each looping over its keys (a sketch)", w * 0.01, h * 0.68);
  }, -0.5);
  titleCard.group.position.set(0.36, 0, -0.575);
  root.add(titleCard.group);

  const cursor = new THREE.Mesh(new THREE.BoxGeometry(0.003, 0.05, 0.3), kit.pink);
  cursor.position.set(laneX, 0.025, -0.375);
  root.add(cursor);

  const lampX = 0.8;
  const lampZ = -0.4;
  const post = kit.solid(new THREE.CylinderGeometry(0.008, 0.014, 0.08, 16), { hull: true });
  post.position.set(lampX, 0.04, lampZ);
  const lamp = new THREE.Mesh(new THREE.SphereGeometry(0.026, 24, 18), kit.ink);
  lamp.position.set(lampX, 0.1, lampZ);
  lamp.castShadow = true;
  const lampCard = createCard(kit, 0.24, 0.07, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.3}px ${fonts.serif}`;
    ctx.fillText("the SM", w * 0.05, h * 0.36);
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.2}px ${fonts.sans}`;
    ctx.fillText("lights only when a warp", w * 0.05, h * 0.64);
    ctx.fillText("has math to do", w * 0.05, h * 0.88);
  });
  lampCard.group.position.set(lampX, 0, lampZ + 0.07);
  root.add(post, lamp, lampCard.group);

  const legend = createCard(kit, 0.88, 0.05, (ctx, w, h) => {
    ctx.font = `${h * 0.36}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.ink;
    ctx.fillText("■ load issued", w * 0.01, h * 0.64);
    ctx.fillStyle = kit.palette.faint;
    ctx.fillText(`┄ waiting: ${numbers.dramCycles} cycles from DRAM, ${numbers.l2Cycles} from L2`, w * 0.18, h * 0.64);
    ctx.fillStyle = kit.palette.blue;
    ctx.fillText("■ math", w * 0.86, h * 0.64);
  }, -0.5);
  legend.group.position.set(0.5, 0, -0.18);
  root.add(legend.group);

  const gauges = [
    { label: "warps active", value: numbers.v2.warps, material: kit.blue },
    { label: "SM throughput", value: numbers.smThroughput, material: kit.blue },
    { label: "DRAM throughput", value: numbers.dram, material: kit.pink },
  ];
  const jarR = 0.045;
  const jarH = 0.14;
  const jarZ = 0.1;
  const fills = gauges.map((g, i) => {
    const x = 0.2 + i * 0.3;
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
    const fill = new THREE.Mesh(new THREE.CylinderGeometry(jarR * 0.94, jarR * 0.94, 1, 32), g.material);
    fill.castShadow = true;
    glass.add(fill, wall, new THREE.LineSegments(rimGeometry, new THREE.LineBasicMaterial({ color: kit.palette.ink })));
    glass.position.set(x, 0, jarZ);
    root.add(glass);
    const card = createCard(kit, 0.2, 0.07, (ctx, w, h) => {
      ctx.fillStyle = kit.palette.faint;
      ctx.font = `${h * 0.24}px ${fonts.sans}`;
      ctx.fillText(g.label, w * 0.05, h * 0.3);
      ctx.fillStyle = kit.palette.ink;
      ctx.font = `${h * 0.46}px ${fonts.serif}`;
      ctx.fillText(`${g.value}%`, w * 0.05, h * 0.8);
    });
    card.group.position.set(x, 0, jarZ + jarR + 0.035);
    root.add(card.group);
    return { fill, value: g.value };
  });
  const sourceCard = createCard(kit, 0.6, 0.032, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.52}px ${fonts.sans}`;
    ctx.fillText("Nsight Compute, v2, 1 row at 2,048 tokens; full jar = 100%", w * 0.01, h * 0.68);
  }, -0.6);
  sourceCard.group.position.set(0.5, 0, 0.32);
  root.add(sourceCard.group);

  let time = 0;
  let lit = false;
  const sweep = 6;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 1.4 : (time + dt) % (sweep + 1.5);
      const grow = reduced ? 1 : smooth(0, 1.2, time);
      for (const { fill, value } of fills) {
        const height = Math.max(0.0005, (value / 100) * jarH * 0.96 * grow);
        fill.scale.y = height;
        fill.position.y = height / 2 + 0.002;
      }
      const x = laneX + Math.min(1, time / sweep) * laneLength;
      cursor.position.x = x;
      let any = false;
      for (const m of maths) {
        const hot = x >= m.from && x <= m.to;
        if (hot) any = true;
        m.mesh.material = mathMaterial(hot);
      }
      if (any !== lit) {
        lit = any;
        lamp.material = any ? kit.pink : kit.ink;
      }
    },
  };
}

function cutExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  const total = 128;
  const splits = 8;
  const per = total / splits;
  const x0 = 0.06;
  const cw = 0.88 / total;
  const rowZ = -0.55;
  const laneZ = (s: number) => -0.44 + s * 0.072;
  root.add(placed(washPlane(kit.palette.blue, 0.9, 0.5, 0.08), 0.5, 0.002, -0.18));
  const tiles = new THREE.InstancedMesh(new THREE.BoxGeometry(cw - 0.0012, 1, 0.03), new THREE.MeshLambertMaterial({ color: "#ffffff" }), total);
  tiles.castShadow = true;
  tiles.receiveShadow = true;
  tiles.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(total * 3), 3);
  root.add(tiles);
  const trough = kit.solid(new THREE.BoxGeometry(0.89, 0.004, 0.04), { shadow: false });
  trough.position.set(0.5, 0.002, rowZ);
  root.add(trough);

  const cuts = Array.from({ length: splits - 1 }, (_, s) => {
    const plane = new THREE.Mesh(new THREE.BoxGeometry(0.0025, 1, 0.06), kit.pink);
    plane.position.set(x0 + (s + 1) * per * cw, 0, rowZ);
    root.add(plane);
    return plane;
  });
  const pucks = Array.from({ length: splits }, () => {
    const puck = new THREE.Mesh(new THREE.SphereGeometry(0.011, 18, 14), kit.pink);
    puck.castShadow = true;
    root.add(puck);
    return puck;
  });
  const laneCards = Array.from({ length: splits }, (_, s) => {
    const card = createCard(kit, 0.2, 0.032, (ctx, w, h) => {
      ctx.fillStyle = kit.palette.ink;
      ctx.font = `${h * 0.5}px ${fonts.sans}`;
      ctx.fillText(`split ${s}: tokens ${fmt(s * per * 16)} to ${fmt((s + 1) * per * 16 - 1)}`, w * 0.04, h * 0.66);
    }, -0.6);
    const right = s < 4;
    card.group.position.set(right ? x0 + (s + 1) * per * cw + 0.115 : x0 + s * per * cw - 0.115, 0, laneZ(s) + 0.006);
    root.add(card.group);
    return card;
  });
  const topCard = createCard(kit, 0.88, 0.1, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.24}px ${fonts.serif}`;
    ctx.fillText(`one head's 2,048 tokens: ${total} KV blocks of 16; thread block = (split, head, row)`, w * 0.02, h * 0.34);
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.19}px ${fonts.sans}`;
    ctx.fillText(`blocks per split = ⌈${total} / ${splits}⌉ = ${per}, so a cut never lands inside a block;`, w * 0.02, h * 0.62);
    ctx.fillText("each range runs v2's online-softmax loop on its own, all 8 at once", w * 0.02, h * 0.86);
  });
  topCard.group.position.set(0.5, 0, 0.2);
  root.add(topCard.group);

  const dummy = new THREE.Object3D();
  const color = new THREE.Color();
  const paper = new THREE.Color(kit.palette.paper);
  const blue = new THREE.Color(kit.palette.blue);
  let time = 0;
  const loop = 11;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 9 : (time + dt) % loop;
      const rise = smooth(0.6, 1.8, time) * (1 - smooth(9.6, 10.6, time));
      const move = smooth(2.2, 3.8, time) * (1 - smooth(9.8, 10.8, time));
      const walk = smooth(4.4, 8.4, time) * (1 - smooth(9.6, 10.2, time));
      cuts.forEach((c) => {
        const h = Math.max(0.0001, rise * 0.05);
        c.scale.y = h;
        c.position.y = h / 2;
      });
      for (let b = 0; b < total; b++) {
        const s = Math.floor(b / per);
        const z = rowZ + (laneZ(s) - rowZ) * move;
        const lift = Math.sin(move * Math.PI) * 0.05;
        const done = (b % per) / per < walk ? 1 : 0;
        const h = 0.012;
        dummy.position.set(x0 + (b + 0.5) * cw, 0.004 + lift + h / 2, z);
        dummy.scale.set(1, h, 1);
        dummy.updateMatrix();
        tiles.setMatrixAt(b, dummy.matrix);
        color.copy(paper).lerp(blue, done);
        tiles.setColorAt(b, color);
      }
      tiles.instanceMatrix.needsUpdate = true;
      if (tiles.instanceColor) tiles.instanceColor.needsUpdate = true;
      pucks.forEach((p, s) => {
        p.visible = move > 0.98;
        p.position.set(x0 + (s * per + walk * per) * cw, 0.027, laneZ(s));
      });
      laneCards.forEach((c) => {
        c.group.visible = move > 0.98;
      });
    },
  };
}

function scratchExhibit(kit: Kit, fonts: Fonts, numbers: SplitNumbers) {
  const root = new THREE.Group();
  const splits = 8;
  const trayZ = -0.38;
  const recordX = (s: number) => 0.115 + s * 0.11;
  root.add(placed(washPlane(kit.palette.pink, 1, 0.45, 0.1), 0.5, 0.002, -0.37));
  let seed = 7;
  const random = () => {
    seed = (seed * 16807) % 2147483647;
    return seed / 2147483647;
  };
  const records = Array.from({ length: splits }, (_, s) => {
    const x = recordX(s);
    const tray = kit.solid(new THREE.BoxGeometry(0.095, 0.008, 0.17));
    tray.position.set(x, 0.004, trayZ);
    const group = new THREE.Group();
    const m = new THREE.Mesh(new THREE.BoxGeometry(0.084, 1, 0.022), kit.pink);
    m.position.set(x, 0, trayZ - 0.06);
    const l = new THREE.Mesh(new THREE.BoxGeometry(0.084, 1, 0.022), kit.pink);
    l.position.set(x, 0, trayZ - 0.032);
    m.castShadow = l.castShadow = true;
    const values = new THREE.InstancedMesh(new THREE.BoxGeometry(0.0085, 1, 0.0085), kit.blue, numbers.headDim);
    values.castShadow = true;
    const heights = Array.from({ length: numbers.headDim }, () => 0.006 + random() * 0.03);
    group.add(m, l, values);
    root.add(tray, group);
    const card = createCard(kit, 0.09, 0.03, (ctx, w, h) => {
      ctx.fillStyle = kit.palette.ink;
      ctx.font = `${h * 0.56}px ${fonts.sans}`;
      ctx.fillText(`split ${s}`, w * 0.08, h * 0.7);
    }, -0.6);
    card.group.position.set(x, 0, trayZ + 0.105);
    root.add(card.group);
    return { m, l, values, heights, x, rank: (s * 5) % splits };
  });
  const header = createCard(kit, 0.7, 0.036, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.52}px ${fonts.sans}`;
    ctx.fillText(`scratch: float [rows, heads, splits, ${numbers.headDim} + 2], one record per split`, w * 0.01, h * 0.68);
  }, -0.5);
  header.group.position.set(0.41, 0, -0.585);
  root.add(header.group);
  const legend = createCard(kit, 0.88, 0.12, (ctx, w, h) => {
    ctx.font = `${h * 0.17}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.pink;
    ctx.fillText("m: the largest score in the range (back bar)", w * 0.02, h * 0.24);
    ctx.fillText("l: the sum of exp(score − m) over the range (front bar)", w * 0.02, h * 0.52);
    ctx.fillStyle = kit.palette.blue;
    ctx.fillText(`acc: ${numbers.headDim} values weighted by exp(score − m), not yet divided by l`, w * 0.02, h * 0.8);
  });
  legend.group.position.set(0.5, 0, -0.13);
  const bytes = createCard(kit, 0.88, 0.09, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.27}px ${fonts.serif}`;
    ctx.fillText(`1 row × 32 heads × 8 splits × ${numbers.headDim + 2} floats × 4 bytes = ${fmt(numbers.scratchBytes)} bytes`, w * 0.02, h * 0.4);
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.21}px ${fonts.sans}`;
    ctx.fillText("allocated once by the runner; records fill in whatever order the splits finish", w * 0.02, h * 0.8);
  });
  bytes.group.position.set(0.5, 0, 0.12);
  root.add(legend.group, bytes.group);

  const dummy = new THREE.Object3D();
  let time = 0;
  const loop = 10;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 8 : (time + dt) % loop;
      const fade = 1 - smooth(9, 9.8, time);
      for (const r of records) {
        const start = 0.6 + r.rank * 0.8;
        const g = smooth(start, start + 0.8, time) * fade;
        const mh = Math.max(0.0001, 0.026 * g);
        const lh = Math.max(0.0001, 0.018 * g);
        r.m.scale.y = mh;
        r.m.position.y = 0.008 + mh / 2;
        r.l.scale.y = lh;
        r.l.position.y = 0.008 + lh / 2;
        r.heights.forEach((height, k) => {
          const h = Math.max(0.0001, height * g);
          dummy.position.set(r.x - 0.0368 + (k % 8) * 0.0105, 0.008 + h / 2, trayZ - 0.008 + Math.floor(k / 8) * 0.0105);
          dummy.scale.set(1, h, 1);
          dummy.updateMatrix();
          r.values.setMatrixAt(k, dummy.matrix);
        });
        r.values.instanceMatrix.needsUpdate = true;
      }
    },
  };
}

function mergeExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  const M = Math.max(...mergePartials.map((p) => p.m));
  const scales = mergePartials.map((p) => Math.exp(p.m - M));
  const scaled = mergePartials.map((p, i) => p.l * scales[i]);
  const total = scaled.reduce((a, b) => a + b, 0);
  const lUnit = 0.0028;
  const mUnit = 0.042;
  const colZ = -0.36;
  const colX = (i: number) => 0.1 + i * 0.165;
  const stackX = 0.84;
  root.add(placed(washPlane(kit.palette.blue, 1, 0.6, 0.12), 0.5, 0.002, -0.3));
  const columns = mergePartials.map((p, i) => {
    const x = colX(i);
    const base = kit.solid(new THREE.BoxGeometry(0.11, 0.008, 0.09));
    base.position.set(x, 0.004, colZ);
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, p.m * mUnit, 12), kit.pink);
    post.position.set(x - 0.04, 0.008 + (p.m * mUnit) / 2, colZ - 0.025);
    post.castShadow = true;
    const pillar = new THREE.Mesh(new THREE.BoxGeometry(0.05, 1, 0.05), kit.blue);
    pillar.castShadow = true;
    root.add(base, post, pillar);
    let factor = 1;
    const card = createCard(kit, 0.15, 0.085, (ctx, w, h) => {
      ctx.fillStyle = kit.palette.ink;
      ctx.font = `${h * 0.24}px ${fonts.serif}`;
      ctx.fillText(`split ${i}`, w * 0.05, h * 0.26);
      ctx.font = `${h * 0.18}px ${fonts.sans}`;
      ctx.fillStyle = kit.palette.pink;
      ctx.fillText(`m = ${p.m.toFixed(1)}`, w * 0.05, h * 0.5);
      ctx.fillStyle = kit.palette.blue;
      ctx.fillText(`l = ${p.l.toFixed(1)}`, w * 0.5, h * 0.5);
      ctx.fillStyle = kit.palette.ink;
      ctx.fillText(factor === 1 ? "× exp(m − M) = ?" : `× ${scales[i].toFixed(2)} → ${scaled[i].toFixed(1)}`, w * 0.05, h * 0.82);
    });
    card.group.position.set(x, 0, colZ + 0.09);
    root.add(card.group);
    return {
      pillar,
      card,
      x,
      setFactor: (shown: boolean) => {
        const next = shown ? 0 : 1;
        if (next !== factor) {
          factor = next;
          card.redraw();
        }
      },
    };
  });
  const ruler = new THREE.Mesh(new THREE.BoxGeometry(colX(3) - colX(0) + 0.04, 0.003, 0.003), kit.pink);
  ruler.position.set((colX(0) + colX(3)) / 2 - 0.04, 0.008 + M * mUnit, colZ - 0.025);
  root.add(ruler);
  const tray = kit.solid(new THREE.BoxGeometry(0.1, 0.008, 0.09));
  tray.position.set(stackX, 0.004, colZ);
  root.add(tray);
  const header = createCard(kit, 0.8, 0.036, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.52}px ${fonts.sans}`;
    ctx.fillText("4 partial results (example values): pink posts are m, blue pillars are l", w * 0.01, h * 0.68);
  }, -0.5);
  header.group.position.set(0.46, 0, -0.1);
  let phase = 0;
  const stackCard = createCard(kit, 0.17, 0.085, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.18}px ${fonts.sans}`;
    ctx.fillText("added in split order", w * 0.05, h * 0.26);
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.3}px ${fonts.serif}`;
    ctx.fillText(phase >= 4 ? `Σ = ${total.toFixed(1)}` : `Σ = ${scaled.slice(0, phase).reduce((a, b) => a + b, 0).toFixed(1)}`, w * 0.05, h * 0.62);
    ctx.font = `${h * 0.17}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.faint;
    ctx.fillText(`${phase} of 4 splits`, w * 0.05, h * 0.88);
  });
  stackCard.group.position.set(stackX, 0, colZ + 0.09);
  const formula = createCard(kit, 0.88, 0.13, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.pink;
    ctx.font = `${h * 0.15}px ${fonts.sans}`;
    ctx.fillText(`M = the largest m = ${M.toFixed(1)}; each split is scaled by exp(m − M)`, w * 0.02, h * 0.2);
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.22}px ${fonts.serif}`;
    ctx.fillText("out = Σ exp(mᵢ − M) accᵢ / Σ exp(mᵢ − M) lᵢ", w * 0.02, h * 0.52);
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.15}px ${fonts.sans}`;
    ctx.fillText("the same answer as one pass over the whole context, and the same bits every run", w * 0.02, h * 0.84);
  });
  formula.group.position.set(0.5, 0, 0.1);
  root.add(header.group, stackCard.group, formula.group);

  let time = 0;
  let shownPhase = -1;
  const loop = 12.5;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 10.5 : (time + dt) % loop;
      const reset = smooth(11.5, 12.4, time);
      const rulerOn = smooth(1.2, 2.2, time) * (1 - reset);
      ruler.scale.x = Math.max(0.001, rulerOn);
      const shrink = smooth(2.8, 4.4, time);
      let stackTop = 0.008;
      let landed = 0;
      columns.forEach((c, i) => {
        const factor = 1 + (scales[i] - 1) * shrink;
        const height = Math.max(0.0001, mergePartials[i].l * lUnit * factor);
        const start = 5 + i * 1.3;
        const go = smooth(start, start + 1, time) * (1 - reset);
        if (go >= 1) landed++;
        const fromY = 0.008 + height / 2;
        const toY = stackTop + height / 2;
        c.pillar.scale.y = height;
        c.pillar.position.set(c.x + 0.012 + (stackX - c.x - 0.012) * go, fromY + (toY - fromY) * go + Math.sin(go * Math.PI) * 0.08, colZ + 0.005);
        c.pillar.visible = reset < 0.98 || go > 0;
        stackTop += mergePartials[i].l * lUnit * scales[i];
        c.setFactor(shrink > 0.5 && reset < 0.5);
      });
      const nextPhase = reset > 0.5 ? 0 : landed;
      if (nextPhase !== shownPhase) {
        shownPhase = nextPhase;
        phase = nextPhase;
        stackCard.redraw();
      }
    },
  };
}

function graphsExhibit(kit: Kit, fonts: Fonts, numbers: SplitNumbers) {
  const root = new THREE.Group();
  const keys = numbers.graphKeys;
  const allSplits = [1, 2, 4, 8];
  const rowZ = (i: number) => -0.53 + i * 0.075;
  const colX = (j: number) => 0.33 + j * 0.155;
  root.add(placed(washPlane(kit.palette.blue, 1, 0.45, 0.1), 0.5, 0.002, -0.38));
  const tile = (text: string) =>
    createCard(kit, 0.12, 0.046, (ctx, w, h) => {
      ctx.fillStyle = kit.palette.ink;
      ctx.font = `${h * 0.5}px ${fonts.sans}`;
      ctx.textAlign = "center";
      ctx.fillText(text, w / 2, h * 0.68);
      ctx.textAlign = "left";
    }, -1.1);
  const tiles: { group: THREE.Group; bucket: number; splits: number; extra: boolean }[] = [];
  keys.forEach((k, i) => {
    const label = createCard(kit, 0.17, 0.04, (ctx, w, h) => {
      ctx.fillStyle = kit.palette.faint;
      ctx.font = `${h * 0.55}px ${fonts.sans}`;
      ctx.fillText(`batch ${k.bucket}`, w * 0.06, h * 0.7);
    }, -0.8);
    label.group.position.set(0.15, 0, rowZ(i));
    root.add(label.group);
    k.splits.forEach((s) => {
      const t = tile(`${s} split${s > 1 ? "s" : ""}`);
      t.group.position.set(colX(allSplits.indexOf(s)), 0.006, rowZ(i));
      root.add(t.group);
      tiles.push({ group: t.group, bucket: k.bucket, splits: s, extra: s !== 1 });
    });
  });
  const plinth = new THREE.Mesh(new THREE.BoxGeometry(0.13, 0.006, 0.06), kit.pink);
  plinth.position.y = 0.006;
  root.add(plinth);
  const oneX = colX(0);
  const replayX = colX(allSplits.indexOf(numbers.replaySplits));
  const total = keys.reduce((a, k) => a + k.splits.length, 0);
  let after = false;
  const status = createCard(kit, 0.88, 0.075, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.3}px ${fonts.serif}`;
    ctx.fillText(after ? `after: one graph per batch size and split count, ${total} in all` : `before: ${keys.length} graphs, each captured on a dummy batch with no context`, w * 0.02, h * 0.42);
    ctx.fillStyle = kit.palette.pink;
    ctx.font = `${h * 0.24}px ${fonts.sans}`;
    ctx.fillText(after ? `batch 1 at 1,984 tokens replays its ${numbers.replaySplits}-split graph` : "so batch 1 at 1,984 tokens replays 1 split: v2's grid", w * 0.02, h * 0.82);
  });
  status.group.position.set(0.5, 0, -0.13);
  root.add(status.group);

  const bars = numbers.graphBars;
  const max = Math.max(...bars.map((b) => b.ms));
  const barZ = (i: number) => 0.02 + i * 0.075;
  const barX = 0.42;
  const barLength = 0.5;
  const barMeshes = bars.map((b, i) => {
    const length = (b.ms / max) * barLength;
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 0.02, 0.036), kit.blue);
    mesh.position.set(barX, 0.01, barZ(i));
    mesh.castShadow = true;
    root.add(mesh);
    const card = createCard(kit, 0.33, 0.04, (ctx, w, h) => {
      ctx.fillStyle = kit.palette.ink;
      ctx.font = `${h * 0.5}px ${fonts.sans}`;
      ctx.fillText(b.label, w * 0.03, h * 0.68);
      ctx.textAlign = "right";
      ctx.fillStyle = kit.palette.faint;
      ctx.fillText(`${b.ms.toFixed(2)} ms`, w * 0.97, h * 0.68);
      ctx.textAlign = "left";
    }, -0.7);
    card.group.position.set(0.235, 0, barZ(i));
    root.add(card.group);
    return { mesh, length };
  });
  const axisCard = createCard(kit, 0.6, 0.03, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.55}px ${fonts.sans}`;
    ctx.fillText("ms per token, 1,984-token prompt, batch 1 (bar length)", w * 0.01, h * 0.7);
  }, -0.6);
  axisCard.group.position.set(0.5, 0, barZ(3) + 0.06);
  root.add(axisCard.group);

  let time = 0;
  const loop = 11;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 9 : (time + dt) % loop;
      const grow = smooth(3, 4.6, time) * (1 - smooth(10, 10.9, time));
      const move = smooth(5, 6.4, time) * (1 - smooth(10, 10.9, time));
      for (const t of tiles) {
        const s = t.extra ? Math.max(0.001, grow) : 1;
        t.group.scale.set(s, s, s);
        t.group.visible = s > 0.002;
      }
      plinth.position.set(oneX + (replayX - oneX) * move, 0.006, rowZ(0) + 0.008);
      const showAfter = move > 0.5;
      if (showAfter !== after) {
        after = showAfter;
        status.redraw();
      }
      const reveal = reduced ? 1 : smooth(0, 1.4, time);
      barMeshes.forEach((b, i) => {
        const length = Math.max(0.001, b.length * reveal);
        b.mesh.scale.x = length;
        b.mesh.position.x = barX + length / 2;
        b.mesh.material = (i === 3 && after) || (i === 2 && !after) ? kit.pink : kit.blue;
      });
    },
  };
}

export function createSplitModel(kit: Kit, fonts: Fonts, numbers: SplitNumbers): BookModel {
  const occupancy = occupancyExhibit(kit, fonts, numbers);
  const waiting = waitingExhibit(kit, fonts, numbers);
  const cut = cutExhibit(kit, fonts);
  const scratch = scratchExhibit(kit, fonts, numbers);
  const merge = mergeExhibit(kit, fonts);
  const graphs = graphsExhibit(kit, fonts, numbers);
  occupancy.init();
  return {
    exhibits: [occupancy.root, waiting.root, cut.root, scratch.root, merge.root, graphs.root],
    handles: [occupancy.knob],
    dragStart: () => occupancy.setHeld(true),
    drag: (x) => occupancy.drag(x),
    dragEnd: () => occupancy.setHeld(false),
    enter: (stop) => {
      if (stop === SPLIT_STOP.few) occupancy.setMode("v2");
      if (stop === SPLIT_STOP.wave) occupancy.setMode("split");
    },
    update: (dt, exhibit, _stop, reduced) => {
      if (exhibit === 0) occupancy.update(dt, reduced);
      if (exhibit === 1) waiting.update(dt, reduced);
      if (exhibit === 2) cut.update(dt, reduced);
      if (exhibit === 3) scratch.update(dt, reduced);
      if (exhibit === 4) merge.update(dt, reduced);
      if (exhibit === 5) graphs.update(dt, reduced);
    },
  };
}
