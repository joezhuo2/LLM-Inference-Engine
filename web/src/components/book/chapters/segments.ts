import * as THREE from "three";
import type { BookModel } from "../book-scene";
import { createCard, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";
import { approach, lay, liquid, makeRail, pen, picker, unitBox } from "./where-the-time-went-model";

export type Tone = "ink" | "blue" | "glass";
export type Segment = { label: string; value: number; headline: string; lines: [string, string]; tone?: Tone };
export type Reference = { label: string; text: string; value: number; tone: Tone; count?: number };
export type SegmentOptions = { title: string; sub: string; refs: Reference[]; row: { label: string; text: string }; segments: Segment[]; total: number; rail: string; rest?: number };
export type Part = { root: THREE.Group; pick: ReturnType<typeof picker>; update: (dt: number, reduced: boolean) => void };

export const X0 = 0.08;
export const W = 0.84;

export function block(kit: Kit, tone: Tone): THREE.Object3D {
  if (tone === "blue") return liquid(kit.blue);
  if (tone === "glass") return kit.solid(unitBox, { material: kit.glass, shadow: false });
  return kit.solid(unitBox);
}

export function tagCard(kit: Kit, fonts: Fonts, root: THREE.Group, z: number, label: string, text: string, color: string) {
  const card = createCard(kit, 0.84, 0.034, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write(label, 0.02, 0.72, 0.58, color);
    ctx.textAlign = "right";
    write(text, 0.98, 0.72, 0.58, color);
    ctx.textAlign = "left";
  });
  card.group.position.set(0.5, 0, z + 0.042);
  root.add(card.group);
}

export function titleCard(kit: Kit, fonts: Fonts, root: THREE.Group, title: string, sub: string) {
  const card = createCard(kit, 0.86, 0.1, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write(title, 0.03, 0.4, 0.28, kit.palette.ink, true);
    write(sub, 0.03, 0.82, 0.2, kit.palette.ink);
  });
  card.group.position.set(0.5, 0, -0.59);
  root.add(card.group);
}

export function readoutCard(kit: Kit, fonts: Fonts, root: THREE.Group, z: number, lines: () => [string, string, string]) {
  const card = createCard(kit, 0.86, 0.12, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    const [a, b, c] = lines();
    write(a, 0.03, 0.34, 0.23, kit.palette.pink, true);
    write(b, 0.03, 0.62, 0.17, kit.palette.ink);
    write(c, 0.03, 0.88, 0.16, kit.palette.ink);
  });
  card.group.position.set(0.5, 0, z);
  root.add(card.group);
  return card;
}

export function segmentExhibit(kit: Kit, fonts: Fonts, o: SegmentOptions): Part {
  const root = new THREE.Group();
  const scale = W / o.total;
  const D = 0.05;
  const H = 0.035;
  const GAP = 0.003;
  const rowZ = (i: number) => -0.09 - 0.11 * (o.refs.length - i);
  const wash = washPlane(kit.palette.blue, 0.9, 0.5, 0.1);
  wash.position.set(0.5, 0.002, -0.25);
  root.add(wash);
  titleCard(kit, fonts, root, o.title, o.sub);

  o.refs.forEach((r, i) => {
    tagCard(kit, fonts, root, rowZ(i), r.label, r.text, r.tone === "blue" ? kit.palette.blue : kit.palette.ink);
    const count = r.count ?? 1;
    const w = (r.value * scale) / count;
    for (let k = 0; k < count; k++) {
      const bar = block(kit, r.tone);
      lay(bar, X0 + k * w, count > 1 ? w - GAP : w, H, rowZ(i), D);
      bar.visible = true;
      root.add(bar);
    }
  });

  const z = rowZ(o.refs.length);
  tagCard(kit, fonts, root, z, o.row.label, o.row.text, kit.palette.pink);
  const blocks = o.segments.map((s) => {
    const plain = block(kit, s.tone ?? "ink");
    const lit = liquid(kit.pink);
    root.add(plain, lit);
    return { plain, lit, height: H };
  });
  const starts: number[] = [];
  o.segments.reduce((x, s) => {
    starts.push(x);
    return x + s.value * scale;
  }, X0);

  let chosen = 0;
  const readout = readoutCard(kit, fonts, root, z + 0.19, () => {
    const s = o.segments[chosen];
    return [s.headline, s.lines[0], s.lines[1]];
  });
  const rail = makeRail(kit, fonts, root, z + 0.36, o.segments.map((_, i) => 0.2 + (0.6 * i) / Math.max(1, o.segments.length - 1)), o.rail, `${o.segments[0].label}, ${o.segments[1].label} …`);
  const pick = picker(rail, 0, o.rest ?? 0, 3, (i) => {
    chosen = i;
    readout.redraw();
  });

  return {
    root,
    pick,
    update: (dt, reduced) => {
      pick.update(dt, reduced);
      blocks.forEach((b, i) => {
        b.height = approach(b.height, i === chosen ? H * 2.6 : H, dt * 1.4, reduced);
        const w = Math.max(0.0012, o.segments[i].value * scale - GAP);
        lay(b.plain, starts[i], w, b.height, z, D);
        lay(b.lit, starts[i], w, b.height, z, D);
        b.plain.visible = i !== chosen;
        b.lit.visible = i === chosen;
      });
    },
  };
}

export function joinParts(parts: Part[]): BookModel {
  let held: Part | null = null;
  return {
    exhibits: parts.map((p) => p.root),
    handles: parts.map((p) => p.pick.knob),
    dragStart: (handle) => {
      held = parts.find((p) => p.pick.knob === handle) ?? null;
      held?.pick.start();
    },
    drag: (x) => held?.pick.drag(x),
    dragEnd: () => {
      held?.pick.end();
      held = null;
    },
    enter: () => parts.forEach((p) => p.pick.enter()),
    update: (dt, exhibit, _stop, reduced) => parts[exhibit]?.update(dt, reduced),
  };
}
