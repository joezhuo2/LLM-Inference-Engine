import * as THREE from "three";
import { angle, MAX_POS, wavelength } from "@/components/kernels/rope/rotation";
import { createCard, type Kit, liquidMaterial, ribbonTube, washPlane } from "../ink";
import type { BookModel } from "../book-scene";
import type { Fonts } from "../typeset";

export type RopeNumbers = { naive: number; draft: number; fast: number };

export const ROPE_STOP = { job: 0, layout: 1, parallel: 2, bits: 3, serial: 4 };

const DIALS = [0, 1, 2, 3, 4, 8, 16, 31];
const placed = <T extends THREE.Object3D>(object: T, x: number, y: number, z: number): T => {
  object.position.set(x, y, z);
  return object;
};
const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const turnLabel = (pair: number) => `one turn: ${Math.round(wavelength(pair)).toLocaleString("en-US")} positions`;

function dialsExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  const arms: { pair: number; group: THREE.Group }[] = [];
  const radius = 0.082;
  root.add(placed(washPlane(kit.palette.blue, 1, 1.0, 0.14), 0.5, 0.002, -0.14));
  DIALS.forEach((pair, k) => {
    const x = 0.145 + (k % 4) * 0.237;
    const z = -0.42 + Math.floor(k / 4) * 0.34;
    const dial = new THREE.Group();
    dial.position.set(x, 0, z);
    const drum = kit.solid(new THREE.CylinderGeometry(radius, radius * 1.04, 0.055, 56), { hull: true, threshold: 35 });
    drum.position.y = 0.0275;
    const ticks: number[] = [];
    for (let t = 0; t < 16; t++) {
      const a = (t / 16) * Math.PI * 2;
      const inner = t % 4 === 0 ? radius * 0.78 : radius * 0.88;
      ticks.push(Math.cos(a) * inner, 0.0565, Math.sin(a) * inner, Math.cos(a) * radius * 0.96, 0.0565, Math.sin(a) * radius * 0.96);
    }
    const tickGeometry = new THREE.BufferGeometry();
    tickGeometry.setAttribute("position", new THREE.Float32BufferAttribute(ticks, 3));
    const tickLines = new THREE.LineSegments(tickGeometry, new THREE.LineBasicMaterial({ color: kit.palette.ink }));
    const arm = new THREE.Group();
    arm.position.y = 0.06;
    const bar = new THREE.Mesh(new THREE.BoxGeometry(radius * 0.86, 0.008, 0.01), kit.blue);
    bar.position.x = radius * 0.43;
    bar.castShadow = true;
    const tip = new THREE.Mesh(new THREE.SphereGeometry(0.014, 18, 14), kit.blue);
    tip.position.x = radius * 0.86;
    tip.castShadow = true;
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.009, 0.009, 0.016, 16), kit.pink);
    arm.add(bar, tip, hub);
    arms.push({ pair, group: arm });
    const card = createCard(kit, 0.2, 0.062, (ctx, w, h) => {
      ctx.fillStyle = kit.palette.ink;
      ctx.font = `${h * 0.46}px ${fonts.serif}`;
      ctx.fillText(`pair ${pair}`, w * 0.05, h * 0.44);
      ctx.font = `${h * 0.28}px ${fonts.sans}`;
      ctx.fillStyle = kit.palette.faint;
      ctx.fillText(turnLabel(pair), w * 0.05, h * 0.82);
    });
    card.group.position.set(0, 0, radius + 0.035);
    dial.add(drum, tickLines, arm, card.group);
    root.add(dial);
  });

  const railZ = 0.205;
  const rail = kit.solid(new THREE.BoxGeometry(0.8, 0.018, 0.022));
  rail.position.set(0.5, 0.009, railZ);
  const stops: number[] = [];
  for (let i = 0; i <= 8; i++) {
    const x = 0.1 + (i / 8) * 0.8;
    stops.push(x, 0.019, railZ - 0.018, x, 0.019, railZ + 0.018);
  }
  const stopGeometry = new THREE.BufferGeometry();
  stopGeometry.setAttribute("position", new THREE.Float32BufferAttribute(stops, 3));
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.03, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(0.1, 0.05, railZ);
  let position = 300;
  const readout = createCard(kit, 0.2, 0.07, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.28}px ${fonts.sans}`;
    ctx.fillText("position", w * 0.05, h * 0.3);
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.62}px ${fonts.serif}`;
    ctx.fillText(Math.floor(position).toLocaleString("en-US"), w * 0.05, h * 0.84);
  });
  readout.group.position.set(0.5, 0, railZ + 0.05);
  root.add(rail, new THREE.LineSegments(stopGeometry, new THREE.LineBasicMaterial({ color: kit.palette.ink })), knob, readout.group);

  let shown = -1;
  let held = false;
  const apply = () => {
    for (const { pair, group } of arms) group.rotation.y = angle(pair, position);
    knob.position.x = 0.1 + (position / MAX_POS) * 0.8;
    const whole = Math.floor(position);
    if (whole !== shown) {
      shown = whole;
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
      position = Math.min(MAX_POS, Math.max(0, ((x - 0.1) / 0.8) * MAX_POS));
      apply();
    },
    update: (dt: number, reduced: boolean) => {
      if (!held && !reduced) position = position + dt >= MAX_POS ? 0 : position + dt;
      apply();
    },
  };
}

function layoutExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.blue, 1, 0.8, 0.2), 0.5, 0.002, -0.08));
  const x0 = 0.06;
  const headW = 0.0218;
  const z = -0.43;
  const qMaterial = liquidMaterial(kit.palette.blue);
  for (let h = 0; h < 40; h++) {
    const material = h < 32 ? qMaterial : h < 36 ? kit.pink : kit.ink;
    const height = h === 0 ? 0.11 : 0.07;
    const head = kit.solid(new THREE.BoxGeometry(headW - 0.004, height, 0.06), { material, threshold: 40 });
    head.position.set(x0 + h * headW + headW / 2, height / 2, z);
    root.add(head);
  }
  const headCard = createCard(kit, 0.86, 0.05, (ctx, w, h) => {
    ctx.font = `${h * 0.48}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.blue;
    ctx.fillText("32 query heads, rotated", w * 0.01, h * 0.68);
    ctx.fillStyle = kit.palette.pink;
    ctx.fillText("4 key", w * 0.72, h * 0.68);
    ctx.fillStyle = kit.palette.faint;
    ctx.fillText("4 value, untouched", w * 0.82, h * 0.68);
  }, -0.5);
  headCard.group.position.set(0.5, 0, z + 0.06);
  root.add(headCard.group);

  const cellW = 0.0134;
  const stripX = 0.5 - (64 * cellW) / 2;
  const stripZ = 0.02;
  const cellGeometry = new THREE.BoxGeometry(cellW - 0.0018, 0.022, 0.05);
  const cells: { mesh: THREE.Mesh; base: number }[] = [];
  for (let i = 0; i < 64; i++) {
    const cell = new THREE.Mesh(cellGeometry, kit.ink.clone());
    cell.position.set(stripX + (i + 0.5) * cellW, 0.011, stripZ);
    cell.castShadow = true;
    cell.receiveShadow = true;
    cells.push({ mesh: cell, base: 0 });
    root.add(cell);
  }
  const connect = new THREE.BufferGeometry();
  connect.setAttribute("position", new THREE.Float32BufferAttribute([x0 + 0.002, 0.11, z + 0.03, stripX, 0.02, stripZ - 0.025, x0 + headW - 0.002, 0.11, z + 0.03, stripX + 64 * cellW, 0.02, stripZ - 0.025], 3));
  root.add(new THREE.LineSegments(connect, new THREE.LineDashedMaterial({ color: kit.palette.faint, dashSize: 0.01, gapSize: 0.008 })).computeLineDistances());

  const arcs: THREE.Mesh[] = [];
  const arcMaterial = kit.blue;
  for (let i = 0; i < 32; i++) {
    const a = stripX + (i + 0.5) * cellW;
    const b = stripX + (i + 32.5) * cellW;
    const top = 0.07 + (i / 31) * 0.2;
    const arc = ribbonTube(
      [new THREE.Vector3(a, 0.022, stripZ), new THREE.Vector3((a * 3 + b) / 4, top * 0.82, stripZ), new THREE.Vector3((a + b) / 2, top, stripZ), new THREE.Vector3((a + b * 3) / 4, top * 0.82, stripZ), new THREE.Vector3(b, 0.022, stripZ)],
      0.0017,
      arcMaterial,
    );
    arc.visible = false;
    arcs.push(arc);
    root.add(arc);
  }
  const formula = createCard(kit, 0.78, 0.17, (ctx, w, h) => {
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${h * 0.12}px ${fonts.sans}`;
    ctx.fillText("pair i = (xᵢ, xᵢ₊₃₂), as in Hugging Face's rotate_half", w * 0.03, h * 0.17);
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${h * 0.17}px ${fonts.serif}`;
    ctx.fillText("x′ᵢ = bf16( bf16(xᵢ · c) + bf16(−xᵢ₊₃₂ · s) )", w * 0.03, h * 0.46);
    ctx.fillText("x′ᵢ₊₃₂ = bf16( bf16(xᵢ₊₃₂ · c) + bf16(xᵢ · s) )", w * 0.03, h * 0.7);
    ctx.fillStyle = kit.palette.pink;
    ctx.font = `${h * 0.12}px ${fonts.sans}`;
    ctx.fillText("8 roundings per pair, in PyTorch's order", w * 0.03, h * 0.92);
  });
  formula.group.position.set(0.5, 0, 0.18);
  root.add(formula.group);

  let time = 0;
  const pink = new THREE.Color(kit.palette.pink);
  const paper = new THREE.Color(kit.palette.paper);
  const blue = new THREE.Color(kit.palette.blue);
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 9 : (time + dt) % 9.5;
      const shown = Math.min(32, Math.floor(time / 0.2));
      arcs.forEach((arc, i) => {
        arc.visible = i < shown;
      });
      cells.forEach(({ mesh }, i) => {
        const pair = i < 32 ? i : i - 32;
        const material = mesh.material as THREE.MeshLambertMaterial;
        if (pair === shown - 1 && shown < 32) material.color.copy(pink);
        else if (pair < shown) material.color.copy(blue);
        else material.color.copy(paper);
      });
    },
  };
}

function bitsExhibit(kit: Kit, fonts: Fonts) {
  const root = new THREE.Group();
  root.add(placed(washPlane(kit.palette.pink, 0.8, 0.8, 0.12), 0.62, 0.002, -0.2));
  const stairZ = -0.27;
  const stepW = 0.17;
  const stepHeights = [0.05, 0.095, 0.14];
  const stairX = 0.36;
  stepHeights.forEach((height, i) => {
    const step = kit.solid(new THREE.BoxGeometry(stepW, height, 0.2));
    step.position.set(stairX + stepW * i + stepW / 2, height / 2, stairZ);
    root.add(step);
  });
  const boundaryX = stairX + stepW * 1.5;
  const boundary = new THREE.Mesh(new THREE.PlaneGeometry(0.2, 0.3), new THREE.MeshBasicMaterial({ color: kit.palette.pink, transparent: true, opacity: 0.2, side: THREE.DoubleSide, depthWrite: false }));
  boundary.rotation.y = Math.PI / 2;
  boundary.position.set(boundaryX, 0.15, stairZ);
  root.add(boundary);

  const pivotX = 0.09;
  const pivotY = 0.34;
  const post = kit.solid(new THREE.CylinderGeometry(0.014, 0.022, pivotY, 18), { hull: true });
  post.position.set(pivotX, pivotY / 2, stairZ);
  root.add(post);
  const armLength = boundaryX - pivotX;
  const makeArm = (material: THREE.Material) => {
    const arm = new THREE.Group();
    arm.position.set(pivotX, pivotY, stairZ);
    const bar = new THREE.Mesh(new THREE.BoxGeometry(armLength, 0.006, 0.006), material);
    bar.position.x = armLength / 2;
    bar.castShadow = true;
    arm.add(bar);
    root.add(arm);
    return arm;
  };
  const hostArm = makeArm(kit.blue);
  const deviceArm = makeArm(kit.pink);
  const ball = (material: THREE.Material) => {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.02, 22, 16), material);
    mesh.castShadow = true;
    root.add(mesh);
    return mesh;
  };
  const hostBall = ball(kit.blue);
  const deviceBall = ball(kit.pink);
  const baseAngle = -0.33;
  const spread = 0.055;

  const writeCard = (w: number, h: number, draw: (ctx: CanvasRenderingContext2D, W: number, H: number) => void) => createCard(kit, w, h, draw);
  const bitsCard = writeCard(0.4, 0.105, (ctx, W, H) => {
    ctx.font = `${H * 0.2}px ${fonts.sans}`;
    ctx.fillStyle = kit.palette.blue;
    ctx.fillText("host std::pow", W * 0.04, H * 0.3);
    ctx.fillStyle = kit.palette.ink;
    ctx.fillText("…1010 0110", W * 0.56, H * 0.3);
    ctx.fillStyle = kit.palette.pink;
    ctx.fillText("device __powf", W * 0.04, H * 0.62);
    ctx.fillStyle = kit.palette.ink;
    ctx.fillText("…1010 0111", W * 0.56, H * 0.62);
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${H * 0.17}px ${fonts.sans}`;
    ctx.fillText("the same θᵢ, off in the last bit", W * 0.04, H * 0.9);
  });
  bitsCard.group.position.set(0.25, 0, 0.0);
  const leverCard = writeCard(0.4, 0.105, (ctx, W, H) => {
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${H * 0.38}px ${fonts.serif}`;
    ctx.fillText("× position 2,047", W * 0.04, H * 0.42);
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${H * 0.17}px ${fonts.sans}`;
    ctx.fillText("the angle carries the error that many times", W * 0.04, H * 0.76);
  });
  leverCard.group.position.set(0.72, 0, 0.0);
  const resultCard = writeCard(0.82, 0.12, (ctx, W, H) => {
    ctx.fillStyle = kit.palette.pink;
    ctx.font = `${H * 0.27}px ${fonts.serif}`;
    ctx.fillText("device powf: 57 of 1.4 million outputs off, up to 64 ulps", W * 0.02, H * 0.38);
    ctx.fillStyle = kit.palette.blue;
    ctx.fillText("θ from the host, passed by value: 0 off", W * 0.02, H * 0.78);
  });
  resultCard.group.position.set(0.5, 0, 0.2);
  root.add(bitsCard.group, leverCard.group, resultCard.group);

  const tipOf = (a: number) => new THREE.Vector3(pivotX + Math.cos(a) * armLength, pivotY + Math.sin(a) * armLength, stairZ);
  let time = 0;
  return {
    root,
    update: (dt: number, reduced: boolean) => {
      time = reduced ? 8.5 : (time + dt) % 10;
      const open = smooth(1.2, 3.2, time);
      const hostA = baseAngle + (spread / 2) * open;
      const deviceA = baseAngle - (spread / 2) * open;
      hostArm.rotation.z = hostA;
      deviceArm.rotation.z = deviceA;
      const drop = smooth(3.8, 4.8, time);
      const hostTip = tipOf(hostA);
      const deviceTip = tipOf(deviceA);
      const deviceLand = stepHeights[1] + 0.02;
      const hostX = Math.min(boundaryX - 0.03, hostTip.x);
      const deviceX = Math.max(boundaryX + 0.03, deviceTip.x);
      hostBall.position.set(hostX + (stairX + stepW * 0.5 - hostX) * smooth(4.8, 5.8, time), hostTip.y + (stepHeights[0] + 0.02 - hostTip.y) * smooth(3.8, 5.8, time), stairZ);
      deviceBall.position.set(deviceX + (stairX + stepW * 1.75 - deviceX) * smooth(4.8, 5.8, time), deviceTip.y + (deviceLand - deviceTip.y) * drop, stairZ);
    },
  };
}

function gridExhibit(kit: Kit, fonts: Fonts, numbers: RopeNumbers) {
  const root = new THREE.Group();
  const rows = 36;
  const columns = 32;
  const cellW = 0.0195;
  const cellD = 0.0132;
  const originX = 0.06;
  const originZ = -0.2;
  root.add(placed(washPlane(kit.palette.blue, 0.95, 0.75, 0.24), 0.38, 0.002, 0.0));
  const plate = kit.solid(new THREE.BoxGeometry(cellW * columns + 0.02, 0.012, cellD * rows + 0.02));
  plate.position.set(originX + (cellW * columns) / 2, 0.006, originZ + (cellD * rows) / 2);
  root.add(plate);
  const material = new THREE.MeshLambertMaterial({ color: "#ffffff" });
  const cells = new THREE.InstancedMesh(new THREE.BoxGeometry(cellW - 0.0028, 1, cellD - 0.0028), material, rows * columns);
  cells.castShadow = true;
  cells.receiveShadow = true;
  cells.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(rows * columns * 3), 3);
  root.add(cells);

  const pegs = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.0052, 0.0052, 1, 10), new THREE.MeshLambertMaterial({ color: "#ffffff" }), 256);
  pegs.castShadow = true;
  pegs.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(256 * 3), 3);
  root.add(pegs);

  const jarX = [0.76, 0.91];
  const jarR = 0.062;
  const jarH = 0.22;
  const jarZ = -0.05;
  const fills = jarX.map((x, i) => {
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
    const fill = new THREE.Mesh(new THREE.CylinderGeometry(jarR * 0.94, jarR * 0.94, 1, 32), i === 0 ? kit.pink : kit.blue);
    fill.castShadow = true;
    glass.add(fill, wall, new THREE.LineSegments(rimGeometry, new THREE.LineBasicMaterial({ color: kit.palette.ink })));
    glass.position.set(x, 0, jarZ);
    root.add(glass);
    return fill;
  });
  const jarCard = (x: number, name: string, count: string) => {
    const card = createCard(kit, 0.135, 0.075, (ctx, W, H) => {
      ctx.fillStyle = kit.palette.faint;
      ctx.font = `${H * 0.24}px ${fonts.sans}`;
      ctx.fillText(name, W * 0.06, H * 0.3);
      ctx.fillStyle = kit.palette.ink;
      ctx.font = `${H * 0.46}px ${fonts.serif}`;
      ctx.fillText(count, W * 0.06, H * 0.78);
    });
    card.group.position.set(x, 0, jarZ + jarR + 0.03);
    root.add(card.group);
  };
  jarCard(jarX[0], "draft: sin, cos", "1,152");
  jarCard(jarX[1], "fast: sin, cos", "256");

  let mode: "serial" | "parallel" = "serial";
  let clockValue = 0;
  const clockCard = createCard(kit, 0.27, 0.09, (ctx, W, H) => {
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${H * 0.22}px ${fonts.sans}`;
    ctx.fillText(mode === "serial" ? "first draft, one token" : "fast kernel, one token", W * 0.04, H * 0.28);
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${H * 0.5}px ${fonts.serif}`;
    ctx.fillText(`${clockValue.toFixed(1)} µs`, W * 0.04, H * 0.78);
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${H * 0.22}px ${fonts.sans}`;
    ctx.fillText(`naive: ${numbers.naive.toFixed(1)} µs`, W * 0.56, H * 0.78);
  });
  clockCard.group.position.set(0.845, 0, 0.17);
  root.add(clockCard.group);
  const threadCard = createCard(kit, 0.5, 0.05, (ctx, W, H) => {
    ctx.fillStyle = kit.palette.ink;
    ctx.font = `${H * 0.5}px ${fonts.sans}`;
    ctx.fillText(mode === "serial" ? "128 threads per token: 36 work, 92 have nothing to do" : "256 threads: 32 pairs × 8 head groups, all working", W * 0.01, H * 0.66);
  }, -0.5);
  threadCard.group.position.set(0.33, 0, -0.27);
  root.add(threadCard.group);

  const axis = new THREE.BufferGeometry();
  axis.setAttribute("position", new THREE.Float32BufferAttribute([originX, 0.013, originZ + cellD * rows + 0.016, originX + cellW * columns, 0.013, originZ + cellD * rows + 0.016], 3));
  root.add(new THREE.LineSegments(axis, new THREE.LineBasicMaterial({ color: kit.palette.ink })));
  const axisCard = createCard(kit, cellW * columns, 0.04, (ctx, W, H) => {
    ctx.fillStyle = kit.palette.faint;
    ctx.font = `${H * 0.56}px ${fonts.sans}`;
    ctx.fillText("pair 0", W * 0.0, H * 0.68);
    ctx.textAlign = "center";
    ctx.fillText("36 heads × 32 pairs", W * 0.5, H * 0.68);
    ctx.textAlign = "right";
    ctx.fillText("pair 31", W, H * 0.68);
    ctx.textAlign = "left";
  }, -0.7);
  axisCard.group.position.set(originX + (cellW * columns) / 2, 0, originZ + cellD * rows + 0.03);
  root.add(axisCard.group);

  const dummy = new THREE.Object3D();
  const color = new THREE.Color();
  const paper = new THREE.Color(kit.palette.paper);
  const blue = new THREE.Color(kit.palette.blue);
  const pink = new THREE.Color(kit.palette.pink);
  const faintPeg = new THREE.Color(kit.palette.faint);
  const secondsPerMicro = 0.28;
  let time = 0;

  const layoutPegs = () => {
    const count = mode === "serial" ? 128 : 256;
    const pegRows = mode === "serial" ? 4 : 8;
    for (let i = 0; i < 256; i++) {
      const c = i % 32;
      const r = Math.floor(i / 32);
      dummy.position.set(originX + (c + 0.5) * cellW, 0.012, -0.48 + r * 0.0135);
      dummy.scale.set(1, i < count ? 0.024 : 0.0001, 1);
      if (pegRows === 4) dummy.position.z = -0.46 + r * 0.02;
      dummy.updateMatrix();
      pegs.setMatrixAt(i, dummy.matrix);
      pegs.setColorAt(i, mode === "serial" ? (i < 36 ? blue : faintPeg) : blue);
    }
    pegs.instanceMatrix.needsUpdate = true;
    if (pegs.instanceColor) pegs.instanceColor.needsUpdate = true;
  };
  layoutPegs();

  const setMode = (next: "serial" | "parallel") => {
    mode = next;
    time = 0;
    layoutPegs();
    threadCard.redraw();
    clockCard.redraw();
  };

  let lastClock = -1;
  return {
    root,
    setMode,
    update: (dt: number, reduced: boolean) => {
      const us = mode === "serial" ? numbers.draft : numbers.fast;
      const duration = us * secondsPerMicro;
      const loop = duration + 3;
      time = reduced ? duration : (time + dt) % loop;
      const t = Math.min(1, time / duration);
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          let done: number;
          let hot = 0;
          if (mode === "serial") {
            const a = column / columns;
            const b = (column + 1) / columns;
            done = smooth(a, b, t);
            hot = t >= a && t < b ? 1 : 0;
          } else {
            const wave = Math.floor(row / 8);
            const a = 0.22 + (wave / 5) * 0.78;
            const b = 0.22 + ((wave + 1) / 5) * 0.78;
            done = smooth(a, b, t);
            hot = t >= a && t < b ? 1 : 0;
          }
          const height = 0.008 + done * 0.04;
          dummy.position.set(originX + (column + 0.5) * cellW, 0.012 + height / 2, originZ + (row + 0.5) * cellD);
          dummy.scale.set(1, height, 1);
          dummy.updateMatrix();
          const index = row * columns + column;
          cells.setMatrixAt(index, dummy.matrix);
          color.copy(paper).lerp(blue, done);
          if (hot) color.lerp(pink, 0.7);
          cells.setColorAt(index, color);
        }
      }
      cells.instanceMatrix.needsUpdate = true;
      if (cells.instanceColor) cells.instanceColor.needsUpdate = true;
      const draftLevel = mode === "serial" ? t : 1;
      const fastLevel = mode === "serial" ? 0 : smooth(0, 0.22, t);
      const setFill = (mesh: THREE.Mesh, level: number) => {
        const height = Math.max(0.0005, level * jarH * 0.9);
        mesh.scale.y = height;
        mesh.position.y = height / 2 + 0.002;
      };
      setFill(fills[0], draftLevel);
      setFill(fills[1], fastLevel * (256 / 1152));
      const clockNow = Math.round(t * us * 10) / 10;
      if (clockNow !== lastClock) {
        lastClock = clockNow;
        clockValue = clockNow;
        clockCard.redraw();
      }
    },
  };
}

export function createRopeModel(kit: Kit, fonts: Fonts, numbers: RopeNumbers): BookModel {
  const dials = dialsExhibit(kit, fonts);
  const layout = layoutExhibit(kit, fonts);
  const bits = bitsExhibit(kit, fonts);
  const grid = gridExhibit(kit, fonts, numbers);
  return {
    exhibits: [dials.root, layout.root, bits.root, grid.root],
    handles: [dials.knob],
    dragStart: () => dials.setHeld(true),
    drag: (x) => dials.drag(x),
    dragEnd: () => dials.setHeld(false),
    enter: (stop) => {
      if (stop === ROPE_STOP.serial) grid.setMode("serial");
      if (stop === ROPE_STOP.parallel) grid.setMode("parallel");
    },
    update: (dt, exhibit, _stop, reduced) => {
      if (exhibit === 0) dials.update(dt, reduced);
      if (exhibit === 1) layout.update(dt, reduced);
      if (exhibit === 2) bits.update(dt, reduced);
      if (exhibit === 3) grid.update(dt, reduced);
    },
  };
}
