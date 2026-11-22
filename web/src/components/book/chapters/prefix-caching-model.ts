import * as THREE from "three";
import type { BookModel } from "../book-scene";
import { createCard, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";
import { block, joinParts, type Part, readoutCard, titleCard, type Tone, W, X0 } from "./segments";
import { approach, lay, liquid, makeRail, pen, picker } from "./where-the-time-went-model";

type Lines = [string, string, string];
export type PcTone = Tone | "pink";
export type PcLane = { label: string; text: string; parts: { value: number; tone: PcTone }[] };
export type PcChoice = { lanes: PcLane[]; lines: Lines };
export type PcLanes = { title: string; sub: string; rail: string; range: string; full: number; choices: PcChoice[]; rest: number };
export type PcSlot = "blank" | "held" | "shared" | "cached" | "taken";
export type PcEvent = { slots: PcSlot[]; lines: Lines };
export type PcPool = { title: string; sub: string; rail: string; range: string; events: PcEvent[] };
export type PcNumbers = { lead: PcLanes; tables: PcPool; pool: PcPool; ttft: PcLanes; burst: PcLanes };

const H = 0.035;
const D = 0.05;
const GAP = 0.003;
const railXs = (count: number) => Array.from({ length: count }, (_, i) => 0.2 + (0.6 * i) / Math.max(1, count - 1));

function wash(root: THREE.Group, kit: Kit) {
  const plane = washPlane(kit.palette.blue, 0.9, 0.5, 0.1);
  plane.position.set(0.5, 0.002, -0.25);
  root.add(plane);
}

const solid = (kit: Kit, tone: PcTone) => (tone === "pink" ? liquid(kit.pink) : block(kit, tone));

function laneCard(kit: Kit, fonts: Fonts, root: THREE.Group, z: number, text: () => [string, string]) {
  const card = createCard(kit, 0.84, 0.032, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    const [left, right] = text();
    write(left, 0.02, 0.72, 0.58, kit.palette.ink);
    ctx.textAlign = "right";
    write(right, 0.98, 0.72, 0.58, kit.palette.blue);
    ctx.textAlign = "left";
  });
  card.group.position.set(0.5, 0, z);
  root.add(card.group);
  return card;
}

export function lanesExhibit(kit: Kit, fonts: Fonts, o: PcLanes): Part {
  const root = new THREE.Group();
  wash(root, kit);
  titleCard(kit, fonts, root, o.title, o.sub);
  const scale = W / o.full;
  const laneZ = (k: number) => -0.42 + k * 0.15;
  let chosen = o.rest;
  const lanes = o.choices[0].lanes.map((lane, k) => {
    const blocks = lane.parts.map((p) => {
      const variants = (["ink", "blue", "glass", "pink"] as const).map((tone) => solid(kit, tone));
      root.add(...variants);
      return { variants, width: p.value * scale };
    });
    const card = laneCard(kit, fonts, root, laneZ(k) + 0.042, () => [o.choices[chosen].lanes[k].label, o.choices[chosen].lanes[k].text]);
    return { blocks, card };
  });
  const readout = readoutCard(kit, fonts, root, 0.1, () => o.choices[chosen].lines);
  const rail = makeRail(kit, fonts, root, 0.27, railXs(o.choices.length), o.rail, o.range);
  const pick = picker(rail, 0, o.rest, 3.4, (i) => {
    chosen = i;
    readout.redraw();
    lanes.forEach((l) => l.card.redraw());
  });
  return {
    root,
    pick,
    update: (dt, reduced) => {
      pick.update(dt, reduced);
      lanes.forEach((l, k) => {
        const parts = o.choices[chosen].lanes[k].parts;
        let at = X0;
        l.blocks.forEach((b, i) => {
          b.width = approach(b.width, parts[i].value * scale, dt * 1.2, reduced);
          const tones = ["ink", "blue", "glass", "pink"];
          b.variants.forEach((v, t) => {
            lay(v, at, Math.max(0, b.width - GAP), H, laneZ(k), D);
            v.visible = v.visible && tones[t] === parts[i].tone;
          });
          at += b.width;
        });
      });
    },
  };
}

export function poolExhibit(kit: Kit, fonts: Fonts, o: PcPool): Part {
  const root = new THREE.Group();
  wash(root, kit);
  titleCard(kit, fonts, root, o.title, o.sub);
  const count = o.events[0].slots.length;
  const pitch = W / count;
  const Z = -0.36;
  const tone: Record<PcSlot, PcTone> = { blank: "glass", held: "blue", shared: "blue", cached: "ink", taken: "pink" };
  const height: Record<PcSlot, number> = { blank: H * 0.25, held: H, shared: H * 2.2, cached: H, taken: H };
  let chosen = 0;
  const slots = Array.from({ length: count }, (_, i) => {
    const variants = (["ink", "blue", "glass", "pink"] as const).map((t) => solid(kit, t));
    root.add(...variants);
    const tag = createCard(kit, pitch * 0.8, 0.026, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      ctx.textAlign = "center";
      write(String(i + 1), 0.5, 0.75, 0.62, kit.palette.ink);
      ctx.textAlign = "left";
    });
    tag.group.position.set(X0 + pitch * (i + 0.5), 0, Z + 0.07);
    root.add(tag.group);
    return { variants, h: height[o.events[0].slots[i]] };
  });
  const legend = createCard(kit, 0.86, 0.042, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("blue: held · tall: held by 2 · ink: cached · glass: blank · pink: just handed out", 0.02, 0.72, 0.5, kit.palette.faint);
  });
  legend.group.position.set(0.5, 0, -0.2);
  root.add(legend.group);
  const readout = readoutCard(kit, fonts, root, 0.1, () => o.events[chosen].lines);
  const rail = makeRail(kit, fonts, root, 0.27, railXs(o.events.length), o.rail, o.range);
  const pick = picker(rail, 0, 0, 3.4, (i) => {
    chosen = i;
    readout.redraw();
  });
  return {
    root,
    pick,
    update: (dt, reduced) => {
      pick.update(dt, reduced);
      const tones = ["ink", "blue", "glass", "pink"];
      slots.forEach((s, i) => {
        const state = o.events[chosen].slots[i];
        s.h = approach(s.h, height[state], dt * 1.4, reduced);
        s.variants.forEach((v, t) => {
          lay(v, X0 + pitch * i + GAP, pitch - 2 * GAP, s.h, Z, D * 1.4);
          v.visible = tones[t] === tone[state];
        });
      });
    },
  };
}

export function createPrefixModel(kit: Kit, fonts: Fonts, n: PcNumbers): BookModel {
  return joinParts([poolExhibit(kit, fonts, n.tables), poolExhibit(kit, fonts, n.pool), lanesExhibit(kit, fonts, n.ttft), lanesExhibit(kit, fonts, n.burst)]);
}
