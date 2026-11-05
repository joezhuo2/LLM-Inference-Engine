import * as THREE from "three";
import type { foundationsNumbers } from "@/components/kernels/foundations/story";
import { createCard, type Kit, washPlane } from "../ink";
import type { BookModel } from "../book-scene";
import type { Fonts } from "../typeset";

export type FoundationsNumbers = typeof foundationsNumbers;

export const FOUNDATIONS_STOP = { hierarchy: 0, memory: 1, bytes: 2, roofline: 3, fusion: 5 };

const placed = <T extends THREE.Object3D>(object: T, x: number, y: number, z: number): T => {
  object.position.set(x, y, z);
  return object;
};
const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const whole = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 0 });
const lines = (kit: Kit, points: number[], dashed = false) => {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
  if (!dashed) return new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: kit.palette.ink }));
  return new THREE.LineSegments(geometry, new THREE.LineDashedMaterial({ color: kit.palette.faint, dashSize: 0.01, gapSize: 0.008 })).computeLineDistances();
};
const strip = (kit: Kit, fonts: Fonts, width: number, text: string, color?: string) =>
  createCard(
    kit,
    width,
    0.045,
    (ctx, w, h) => {
      ctx.fillStyle = color ?? kit.palette.ink;
      ctx.font = `${h * 0.5}px ${fonts.sans}`;
      ctx.fillText(text, w * 0.02, h * 0.66);
    },
    -0.5,
  );

function hierarchyExhibit(kit: Kit, fonts: Fonts, n: FoundationsNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 1, 1.0, 0.14), 0.5, 0.002, -0.14));
  const perRow = Math.ceil(n.sms / 2);
  const pitch = 0.86 / perRow;
  const tile = pitch * 0.84;
  const smAt = (i: number) => ({ x: 0.07 + (i % perRow) * pitch + pitch / 2, z: -0.56 + Math.floor(i / perRow) * pitch });
  for (let i = 0; i < n.sms; i++) {
    const { x, z } = smAt(i);
    root.add(placed(kit.solid(new THREE.BoxGeometry(tile, 0.01, tile)), x, 0.005, z));
  }
  const picked = 5;
  const cube = tile * 0.62;
  const cubeGeometry = new THREE.BoxGeometry(cube, cube, cube);
  const blocks = Array.from({ length: n.blocks }, (_, i) => {
    const block = kit.solid(cubeGeometry, { material: i === picked ? kit.blue : kit.ink });
    root.add(block);
    return block;
  });
  const smCard = strip(kit, fonts, 0.86, `${n.sms} SMs, ${n.blocks} blocks of ${n.threadsPerBlock} threads: each block lands on one SM and stays there`);
  smCard.group.position.set(0.5, 0, -0.43);

  const warps = n.threadsPerBlock / 32;
  const warpPick = 2;
  const warpW = 0.8 / warps;
  const warpZ = -0.22;
  for (let i = 0; i < warps; i++) {
    const warp = kit.solid(new THREE.BoxGeometry(warpW * 0.86, 0.035, 0.06), { material: i === warpPick ? kit.pink : kit.ink });
    root.add(placed(warp, 0.1 + i * warpW + warpW / 2, 0.0175, warpZ));
  }
  const warpCard = strip(kit, fonts, 0.8, `one block: ${n.threadsPerBlock} threads = ${warps} warps of 32`);
  warpCard.group.position.set(0.5, 0, warpZ + 0.045);

  const blockAt = smAt(picked);
  const threadZ = -0.03;
  const threadPitch = 0.8 / 16;
  root.add(
    lines(kit, [blockAt.x - cube / 2, 0.003, -0.39, 0.1, 0.003, warpZ - 0.035, blockAt.x + cube / 2, 0.003, -0.39, 0.9, 0.003, warpZ - 0.035], true),
    lines(kit, [0.1 + warpPick * warpW + warpW * 0.07, 0.003, -0.145, 0.1, 0.003, threadZ - 0.022, 0.1 + (warpPick + 1) * warpW - warpW * 0.07, 0.003, -0.145, 0.9, 0.003, threadZ - 0.022], true),
  );
  const pegs: THREE.Mesh[] = [];
  const pegGeometry = new THREE.CylinderGeometry(0.011, 0.011, 1, 16);
  for (let i = 0; i < 32; i++) {
    const x = 0.1 + (i % 16) * threadPitch + threadPitch / 2;
    const z = threadZ + Math.floor(i / 16) * 0.045;
    root.add(placed(kit.solid(new THREE.CylinderGeometry(0.016, 0.016, 0.006, 18)), x, 0.003, z));
    const peg = new THREE.Mesh(pegGeometry, kit.blue);
    peg.castShadow = true;
    peg.position.set(x, 0, z);
    pegs.push(peg);
    root.add(peg);
  }
  let instruction = 0;
  const threadCard = createCard(kit, 0.8, 0.09, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.26}px ${fonts.sans}`;
    ctx.fillText(`one warp: 32 threads, instruction ${instruction} of 4 for all 32 at once`, w * 0.02, h * 0.36);
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.24}px ${fonts.serif}`;
    ctx.fillText("each thread finds its value: blockIdx × blockDim + threadIdx", w * 0.02, h * 0.78);
  }, -0.5);
  threadCard.group.position.set(0.5, 0, threadZ + 0.1);
  root.add(smCard.group, warpCard.group, threadCard.group);

  let time = 0;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 13 : (time + dt) % 14;
      blocks.forEach((block, i) => {
        const { x, z } = smAt(i);
        const start = 0.4 + i * 0.12;
        block.position.set(x, 0.01 + cube / 2 + 0.22 * (1 - smooth(start, start + 0.9, time)), z);
      });
      const run = Math.min(4, Math.max(0, (time - 4.6) / 1.6));
      const step = Math.floor(run);
      const height = 0.012 + 0.012 * (step + smooth(0.1, 0.7, run - step));
      for (const peg of pegs) {
        peg.scale.y = height;
        peg.position.y = height / 2 + 0.006;
      }
      const shown = Math.min(4, run < 0.4 ? 0 : step + (run - step > 0.4 ? 1 : 0));
      if (shown !== instruction) {
        instruction = shown;
        threadCard.redraw();
      }
    },
  };
}

function memoryExhibit(kit: Kit, fonts: Fonts, n: FoundationsNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 0.7, 1.0, 0.14), 0.3, 0.002, -0.2));
  const zs = [-0.53, -0.34, -0.15, 0.04];
  const widths = [0.46, 0.36, 0.26, 0.16];
  const slabX = 0.25;
  const slabH = 0.03;
  n.tiers.forEach((tier, i) => {
    root.add(placed(kit.solid(new THREE.BoxGeometry(widths[i], slabH, 0.11)), slabX, slabH / 2, zs[i]));
    const card = createCard(kit, 0.4, 0.135, (ctx, w, h) => {
      ctx.fillStyle = kit.palette.ink;
      ctx.font = `${h * 0.2}px ${fonts.serif}`;
      ctx.fillText(tier.name, w * 0.03, h * 0.2);
      ctx.fillStyle = kit.palette.blue;
      ctx.font = `${h * 0.14}px ${fonts.sans}`;
      ctx.fillText(tier.size, w * 0.03, h * 0.39);
      ctx.fillStyle = kit.palette.faint;
      tier.lines.forEach((line, k) => ctx.fillText(line, w * 0.03, h * (0.6 + k * 0.165)));
    });
    card.group.position.set(0.71, 0, zs[i] - 0.04);
    root.add(card.group);
  });
  const { dramNs, l2Ns, smemNs, dramCycles, l2Cycles, smemCycles, fmaCycles } = n.chip;
  const footer = createCard(kit, 0.9, 0.08, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.3}px ${fonts.sans}`;
    ctx.fillText("closer to the thread: smaller and faster", w * 0.02, h * 0.38);
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.26}px ${fonts.sans}`;
    ctx.fillText(`latency in cycles: DRAM ${whole(dramCycles)}, L2 ${whole(l2Cycles)}, shared ${whole(smemCycles)}, FMA ${Math.round(fmaCycles)}`, w * 0.02, h * 0.8);
  }, -0.5);
  footer.group.position.set(0.5, 0, 0.21);
  root.add(footer.group);

  const packet = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.022, 0.03), kit.blue);
  const result = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.022, 0.03), kit.pink);
  const fma = new THREE.Mesh(new THREE.SphereGeometry(0.02, 20, 14), kit.pink);
  for (const mesh of [packet, result, fma]) {
    mesh.castShadow = true;
    root.add(mesh);
  }
  const rest = slabH + 0.011;
  const hopTimes = [dramNs, l2Ns, smemNs].map((ns) => 0.8 + (2.6 * ns) / dramNs);
  const hops: [number, number][] = [];
  let clock = 0.5;
  for (const d of hopTimes) {
    hops.push([clock, clock + d]);
    clock += d + 0.3;
  }
  const fmaAt: [number, number] = [clock, clock + 0.9];
  const back: [number, number] = [clock + 1.1, clock + 3.1];
  const loop = back[1] + 1.4;
  const arc = (mesh: THREE.Mesh, from: number, to: number, t: number, lift: number) => {
    const z = zs[from] + (zs[to] - zs[from]) * t;
    mesh.position.set(slabX, rest + Math.sin(Math.PI * t) * lift, z);
  };
  let time = 0;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? fmaAt[1] : (time + dt) % loop;
      const goingBack = time >= back[0];
      packet.visible = time < back[0];
      result.visible = goingBack;
      let tier = 0;
      let along = 0;
      hops.forEach(([a, b], k) => {
        if (time >= a) {
          tier = k;
          along = smooth(a, b, time);
        }
      });
      arc(packet, tier, tier + 1, along, 0.06);
      const pulse = smooth(fmaAt[0], fmaAt[0] + 0.45, time) * (1 - smooth(back[0], back[0] + 0.3, time));
      fma.scale.setScalar(Math.max(0.001, pulse));
      fma.position.set(slabX + 0.05, rest + 0.012, zs[3]);
      if (goingBack) arc(result, 3, 0, smooth(back[0], back[1], time), 0.12);
    },
  };
}

function bytesExhibit(kit: Kit, fonts: Fonts, n: FoundationsNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.pink, 1, 0.9, 0.12), 0.5, 0.002, -0.2));
  const barX = 0.31;
  const barL = 0.62;
  const maxUs = Math.max(...n.lanes.flatMap((l) => [l.memUs, l.computeUs]));
  const duration = 7;
  const fills: { mesh: THREE.Mesh; us: number }[] = [];
  const laneZ = [-0.54, -0.2];
  n.lanes.forEach((lane, li) => {
    const z0 = laneZ[li];
    [
      { name: "memory: bytes ÷ GB/s", us: lane.memUs, material: kit.blue },
      { name: "compute: FLOPs ÷ TFLOP/s", us: lane.computeUs, material: kit.pink },
    ].forEach((bar, bi) => {
      const z = z0 + bi * 0.08;
      root.add(placed(kit.solid(new THREE.BoxGeometry(barL, 0.034, 0.046), { material: kit.glass, shadow: false }), barX + barL / 2, 0.017, z));
      const fill = new THREE.Mesh(new THREE.BoxGeometry(1, 0.026, 0.036), bar.material);
      fill.castShadow = true;
      fill.position.set(barX, 0.015, z);
      fills.push({ mesh: fill, us: bar.us });
      root.add(fill);
      const label = createCard(kit, 0.26, 0.05, (ctx, w, h) => {
        ctx.fillStyle = bar.material === kit.blue ? kit.palette.blue : kit.palette.pink;
        ctx.font = `${h * 0.42}px ${fonts.sans}`;
        ctx.fillText(bar.name, w * 0.02, h * 0.42);
        ctx.fillStyle = kit.palette.ink;
        ctx.fillText(`${bar.us.toFixed(bar.us < 10 ? 1 : 0)} µs`, w * 0.02, h * 0.9);
      }, -1.25);
      label.group.position.set(0.15, 0, z - 0.03);
      root.add(label.group);
    });
    const card = createCard(kit, 0.86, 0.075, (ctx, w, h) => {
      ctx.fillStyle = kit.palette.ink;
      ctx.font = `${h * 0.3}px ${fonts.serif}`;
      ctx.fillText(`${lane.title}: ${lane.rows} row${lane.rows > 1 ? "s" : ""}, ${lane.flopPerByte.toFixed(lane.rows > 1 ? 0 : 1)} FLOP per byte`, w * 0.02, h * 0.36);
      ctx.fillStyle = kit.palette.faint;
      ctx.font = `${h * 0.26}px ${fonts.sans}`;
      ctx.fillText(lane.note, w * 0.02, h * 0.8);
    }, -0.5);
    card.group.position.set(0.5, 0, z0 + 0.16);
    root.add(card.group);
  });
  let shown = -1;
  let elapsed = 0;
  const clock = createCard(kit, 0.86, 0.075, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.34}px ${fonts.serif}`;
    ctx.fillText(`${whole(elapsed)} µs`, w * 0.02, h * 0.38);
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.24}px ${fonts.sans}`;
    ctx.fillText(`the longer bar sets the time; gate/up at 1 row measured: ${n.measuredDecodeUs.toFixed(1)} µs per layer, at the roof`, w * 0.02, h * 0.8);
  }, -0.5);
  clock.group.position.set(0.5, 0, 0.17);
  root.add(clock.group);
  let time = 0;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? duration : (time + dt) % (duration + 3);
      const now = Math.min(1, time / duration) * maxUs;
      for (const { mesh, us } of fills) {
        const length = Math.max(0.003, (Math.min(now, us) / maxUs) * barL);
        mesh.scale.x = length;
        mesh.position.x = barX + length / 2;
      }
      const rounded = Math.round(now);
      if (rounded !== shown) {
        shown = rounded;
        elapsed = rounded;
        clock.redraw();
      }
    },
  };
}

function rooflineExhibit(kit: Kit, fonts: Fonts, n: FoundationsNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 1, 0.9, 0.14), 0.5, 0.002, -0.15));
  const xr = [0.5, 1000];
  const yr = [0.05, 100];
  const lx = (v: number) => 0.08 + ((Math.log10(v) - Math.log10(xr[0])) / (Math.log10(xr[1]) - Math.log10(xr[0]))) * 0.84;
  const fromX = (x: number) => 10 ** (Math.log10(xr[0]) + ((x - 0.08) / 0.84) * (Math.log10(xr[1]) - Math.log10(xr[0])));
  const hy = (v: number) => 0.02 + ((Math.log10(v) - Math.log10(yr[0])) / (Math.log10(yr[1]) - Math.log10(yr[0]))) * 0.3;
  const roof = (f: number) => Math.min((f * n.gbs) / 1000, n.tflops);
  const wallZ = -0.3;
  const wallD = 0.05;
  const shape = new THREE.Shape();
  shape.moveTo(lx(xr[0]), 0);
  shape.lineTo(lx(xr[0]), hy(roof(xr[0])));
  shape.lineTo(lx(n.ridge), hy(n.tflops));
  shape.lineTo(lx(xr[1]), hy(n.tflops));
  shape.lineTo(lx(xr[1]), 0);
  shape.closePath();
  const wall = kit.solid(new THREE.ExtrudeGeometry(shape, { depth: wallD, bevelEnabled: false }), { threshold: 20 });
  wall.position.z = wallZ - wallD / 2;
  root.add(wall);
  const front = wallZ + wallD / 2 + 0.002;
  const ticks: number[] = [];
  for (const v of [1, 10, 100, 1000]) ticks.push(lx(v), 0.002, front, lx(v), 0.002, front + 0.02);
  root.add(lines(kit, ticks), lines(kit, [lx(n.ridge), 0.002, front, lx(n.ridge), hy(n.tflops) + 0.05, front], true));
  const axis = createCard(kit, 0.9, 0.05, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.38}px ${fonts.sans}`;
    ctx.textAlign = "center";
    for (const v of [1, 10, 100, 1000]) ctx.fillText(String(v), ((lx(v) - 0.05) / 0.9) * w, h * 0.4);
    ctx.fillText("FLOP per byte ≈ rows per weight read", w * 0.5, h * 0.88);
    ctx.textAlign = "left";
  }, -0.6);
  axis.group.position.set(0.5, 0, front + 0.025);
  const slope = strip(kit, fonts, 0.42, `slope: ${n.gbs} GB/s, memory bound`, kit.palette.ink);
  slope.group.position.set(0.27, 0, front + 0.1);
  const top = strip(kit, fonts, 0.42, `flat top: ${n.tflops} TFLOP/s, compute bound`, kit.palette.ink);
  top.group.position.set(0.7, 0, front + 0.17);
  const ridgeCard = strip(kit, fonts, 0.26, `ridge: ${Math.round(n.ridge)} FLOP/byte`, kit.palette.faint);
  ridgeCard.group.position.set(0.66, 0, front + 0.1);
  root.add(axis.group, slope.group, top.group, ridgeCard.group);

  for (const m of n.measured) {
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.011, 16, 12), kit.pink);
    dot.position.set(lx(m.x), hy(m.y), front + 0.006);
    root.add(dot);
  }

  const point = new THREE.Mesh(new THREE.SphereGeometry(0.02, 22, 16), kit.blue);
  point.castShadow = true;
  const railZ = 0.12;
  const minX = lx(n.flopPerByte1);
  const maxX = lx(n.flopPerByte1 * n.prefillRows);
  const rail = kit.solid(new THREE.BoxGeometry(maxX - minX + 0.04, 0.018, 0.022));
  rail.position.set((minX + maxX) / 2, 0.009, railZ);
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.03, 28, 22), kit.pink);
  knob.castShadow = true;
  const stem = lines(kit, [0, 0, 0, 0, 0, 1]);
  root.add(point, rail, knob, stem);

  let rows = 1;
  let shownRows = -1;
  const readout = createCard(kit, 0.6, 0.07, (ctx, w, h) => {
    const f = rows * n.flopPerByte1;
    const bound = f < n.ridge ? "memory bound" : "compute bound";
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.28}px ${fonts.sans}`;
    ctx.fillText(`gate/up, computed from the ceilings · ${roof(f).toFixed(1)} TFLOP/s`, w * 0.03, h * 0.32);
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.48}px ${fonts.serif}`;
    ctx.fillText(`${rows} row${rows > 1 ? "s" : ""}: `, w * 0.03, h * 0.84);
    const left = ctx.measureText(`${rows} row${rows > 1 ? "s" : ""}: `).width;
    ctx.fillStyle = f < n.ridge ? kit.palette.blue : kit.palette.pink;
    ctx.fillText(bound, w * 0.03 + left, h * 0.84);
  });
  readout.group.position.set(0.5, 0, railZ + 0.05);
  root.add(readout.group);

  let held = false;
  let manual = false;
  let time = 0;
  const apply = (rowsF: number) => {
    const f = rowsF * n.flopPerByte1;
    const x = lx(f);
    point.position.set(x, hy(roof(f)) + 0.02, wallZ);
    knob.position.set(x, 0.05, railZ);
    stem.position.set(x, 0.002, front);
    stem.scale.set(1, 1, railZ - front - 0.03);
    const r = Math.round(rowsF);
    if (r !== shownRows) {
      shownRows = r;
      rows = r;
      readout.redraw();
    }
  };
  const maxLog = Math.log10(n.prefillRows);
  let current = 1;
  apply(current);
  return {
    root,
    knob,
    setHeld: (value: boolean) => {
      held = value;
      if (value) manual = true;
    },
    reset: () => {
      manual = false;
      time = 0;
    },
    drag: (x: number) => {
      current = Math.min(n.prefillRows, Math.max(1, fromX(Math.min(maxX, Math.max(minX, x))) / n.flopPerByte1));
      apply(current);
    },
    update: (dt: number, reduced: boolean) => {
      if (held || manual) return;
      if (reduced) current = n.ridge / n.flopPerByte1;
      else {
        time = (time + dt) % 16;
        current = 10 ** (maxLog * (smooth(1, 9, time) - smooth(11.5, 14.5, time)));
      }
      apply(current);
    },
  };
}

function fusionExhibit(kit: Kit, fonts: Fonts, n: FoundationsNumbers) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 1, 1.0, 0.12), 0.5, 0.002, -0.12));
  type Trip = { x: number; up: boolean };
  const lane = (dramZ: number, kernels: { x: number; w: number; name: string }[], trips: Trip[], title: string, kb: number, held: boolean) => {
    const kernelZ = dramZ + 0.19;
    root.add(placed(kit.solid(new THREE.BoxGeometry(0.84, 0.02, 0.07)), 0.5, 0.01, dramZ));
    const dramCard = strip(kit, fonts, 0.2, "global memory", kit.palette.faint);
    dramCard.group.position.set(0.18, 0, dramZ + 0.04);
    root.add(dramCard.group);
    for (const k of kernels) {
      root.add(placed(kit.solid(new THREE.BoxGeometry(k.w, 0.06, 0.07)), k.x, 0.03, kernelZ));
      const card = strip(kit, fonts, k.w, k.name);
      card.group.position.set(k.x, 0.06, kernelZ);
      root.add(card.group);
    }
    const sum = new THREE.Mesh(new THREE.SphereGeometry(0.016, 18, 12), kit.pink);
    sum.position.set(kernels[0].x, 0.075, kernelZ);
    sum.visible = held;
    root.add(sum);
    const packets = trips.map((trip) => {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.026, 0.018, 0.026), trip.up ? kit.blue : kit.pink);
      mesh.castShadow = true;
      root.add(mesh);
      return mesh;
    });
    const tallies = trips.map((trip, k) => {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.022, 0.014, 0.022), trip.up ? kit.blue : kit.pink);
      mesh.position.set(0.09 + k * 0.03, 0.007, kernelZ + 0.11);
      root.add(mesh);
      return mesh;
    });
    const tally = createCard(kit, 0.62, 0.05, (ctx, w, h) => {
      ctx.fillStyle = kit.palette.ink;
      ctx.font = `${h * 0.44}px ${fonts.sans}`;
      ctx.fillText(`${title}: ${trips.length} transfers, ${whole(kb)} KB`, w * 0.02, h * 0.42);
      ctx.fillStyle = kit.palette.faint;
      ctx.font = `${h * 0.34}px ${fonts.sans}`;
      ctx.fillText(held ? "the pink sum stays in registers; blue: read from DRAM, pink: written back" : "blue: read from DRAM, pink: written back", w * 0.02, h * 0.88);
    }, -0.5);
    tally.group.position.set(0.6, 0, kernelZ + 0.09);
    root.add(tally.group);
    return { trips, packets, tallies, sum, dramZ, kernelZ };
  };
  const unfused = lane(
    -0.55,
    [
      { x: 0.32, w: 0.16, name: "add" },
      { x: 0.66, w: 0.2, name: "RMSNorm" },
    ],
    [
      { x: 0.28, up: true },
      { x: 0.36, up: true },
      { x: 0.32, up: false },
      { x: 0.61, up: true },
      { x: 0.66, up: true },
      { x: 0.71, up: false },
    ],
    "unfused",
    n.fusion.unfusedKb,
    false,
  );
  const fused = lane(
    -0.18,
    [{ x: 0.5, w: 0.3, name: "add + RMSNorm" }],
    [
      { x: 0.4, up: true },
      { x: 0.47, up: true },
      { x: 0.54, up: false },
      { x: 0.61, up: false },
    ],
    `fused, ${n.fusion.tokens} tokens`,
    n.fusion.fusedKb,
    true,
  );
  const pace = 1.1;
  const loop = 6 * pace + 2.5;
  let time = 0;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? loop - 0.5 : (time + dt) % loop;
      for (const l of [unfused, fused]) {
        l.trips.forEach((trip, k) => {
          const t = smooth(k * pace + 0.3, k * pace + 0.3 + pace * 0.85, time);
          const mesh = l.packets[k];
          mesh.visible = t > 0 && t < 1;
          const from = trip.up ? l.dramZ : l.kernelZ;
          const to = trip.up ? l.kernelZ : l.dramZ;
          mesh.position.set(trip.x, 0.03 + Math.sin(Math.PI * t) * 0.09, from + (to - from) * t);
          l.tallies[k].visible = t >= 1;
        });
        l.sum.scale.setScalar(0.8 + 0.2 * Math.sin(time * 2.2));
      }
    },
  };
}

export function createFoundationsModel(kit: Kit, fonts: Fonts, numbers: FoundationsNumbers): BookModel {
  const hierarchy = hierarchyExhibit(kit, fonts, numbers);
  const memory = memoryExhibit(kit, fonts, numbers);
  const bytes = bytesExhibit(kit, fonts, numbers);
  const roofline = rooflineExhibit(kit, fonts, numbers);
  const fusion = fusionExhibit(kit, fonts, numbers);
  return {
    exhibits: [hierarchy.root, memory.root, bytes.root, roofline.root, fusion.root],
    handles: [roofline.knob],
    dragStart: () => roofline.setHeld(true),
    drag: (x) => roofline.drag(x),
    dragEnd: () => roofline.setHeld(false),
    enter: (stop) => {
      if (stop === FOUNDATIONS_STOP.roofline) roofline.reset();
    },
    update: (dt, exhibit, _stop, reduced) => {
      if (exhibit === 0) hierarchy.update(dt, reduced);
      if (exhibit === 1) memory.update(dt, reduced);
      if (exhibit === 2) bytes.update(dt, reduced);
      if (exhibit === 3) roofline.update(dt, reduced);
      if (exhibit === 4) fusion.update(dt, reduced);
    },
  };
}
