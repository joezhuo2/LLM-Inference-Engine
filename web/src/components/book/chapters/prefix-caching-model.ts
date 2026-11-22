import * as THREE from "three";
import type { BookModel } from "../book-scene";
import { type PoolPhase, poolRun } from "@/components/kernels/prefix-caching/pool";
import { createCard, inkSegments, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";
import { block, type Part, readoutCard, titleCard, type Tone, W, X0 } from "./segments";
import { approach, lay, liquid, makeRail, pen, picker } from "./where-the-time-went-model";

type Lines = [string, string, string];
export type PcTone = Tone | "pink";
export type PcLane = { label: string; text: string; parts: { value: number; tone: PcTone }[] };
export type PcChoice = { lanes: PcLane[]; lines: Lines };
export type PcLanes = { title: string; sub: string; rail: string; range: string; full: number; choices: PcChoice[]; rest: number };
export type PcSlot = "blank" | "held" | "shared" | "cached" | "taken";
export type PcEvent = { slots: PcSlot[]; lines: Lines };
export type PcPool = { title: string; sub: string; rail: string; range: string; events: PcEvent[] };
export type PcKnob = { label: string; values: number[]; start: number; text: (v: number) => string };
export type PcSandbox = { title: string; sub: string; shared: PcKnob; requests: PcKnob; pool: PcKnob; steps: string[]; restStep: number; lines: (phase: PoolPhase, step: number, knobs: { shared: number; requests: number; pool: number }) => Lines };
export type PcNumbers = { lead: PcLanes; tables: PcPool; pool: PcPool; ttft: PcLanes; burst: PcLanes; sandbox: PcSandbox };

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

type Rail = ReturnType<typeof makeRail>;
type Multi = { root: THREE.Group; picks: ReturnType<typeof picker>[]; update: (dt: number, reduced: boolean) => void };

function miniRail(kit: Kit, fonts: Fonts, root: THREE.Group, center: number, z: number, count: number, label: string, value: () => string) {
  const span = 0.15;
  const xs = Array.from({ length: count }, (_, i) => center - span / 2 + (span * i) / Math.max(1, count - 1));
  const bar = kit.solid(new THREE.BoxGeometry(span + 0.03, 0.016, 0.02));
  bar.position.set(center, 0.008, z);
  const ticks: number[] = [];
  for (const x of xs) ticks.push(x, 0.017, z - 0.018, x, 0.017, z + 0.018);
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.024, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(xs[0], 0.04, z);
  const card = createCard(kit, 0.205, 0.05, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    ctx.textAlign = "center";
    write(label, 0.5, 0.42, 0.34, kit.palette.faint);
    write(value(), 0.5, 0.86, 0.38, kit.palette.pink);
    ctx.textAlign = "left";
  });
  card.group.position.set(center, 0, z + 0.065);
  root.add(bar, inkSegments(ticks, kit.thin), knob, card.group);
  const nearest = (px: number) => xs.reduce((best, x, i) => (Math.abs(x - px) < Math.abs(xs[best] - px) ? i : best), 0);
  const rail: Rail = { knob, xs, a: xs[0], b: xs[xs.length - 1], nearest };
  return { rail, card };
}

export function poolSandbox(kit: Kit, fonts: Fonts, o: PcSandbox): Multi {
  const root = new THREE.Group();
  wash(root, kit);
  titleCard(kit, fonts, root, o.title, o.sub);
  const most = Math.max(...o.pool.values);
  const pitch = W / most;
  const Z = -0.44;
  const rowZ = (r: number) => -0.33 + r * 0.036;
  const knobs = { shared: o.shared.values[o.shared.start], requests: o.requests.values[o.requests.start], pool: o.pool.values[o.pool.start] };
  let phases = poolRun(knobs);
  let step = o.restStep;
  const phase = () => phases[Math.min(step, phases.length - 1)];
  const tone = { blank: "glass", held: "blue", cached: "ink", evicted: "pink" } as const;
  const height = (s: PoolPhase["slots"][number]) => (s.state === "blank" ? H * 0.25 : s.state === "cached" ? H : H * (0.6 + 0.4 * s.count));
  const slots = Array.from({ length: most }, (_, i) => {
    const variants = (["ink", "blue", "glass", "pink"] as const).map((t) => solid(kit, t));
    root.add(...variants);
    const tag = createCard(kit, pitch * 0.92, 0.036, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      const s = phase().slots[i];
      ctx.textAlign = "center";
      write(String(i + 1), 0.5, 0.44, 0.4, kit.palette.ink);
      if (s && s.count > 1) write(`×${s.count}`, 0.5, 0.9, 0.38, kit.palette.blue);
      ctx.textAlign = "left";
    });
    tag.group.position.set(X0 + pitch * (i + 0.5), 0, Z + 0.055);
    root.add(tag.group);
    return { variants, tag, h: H * 0.25 };
  });
  const rows = Array.from({ length: Math.max(...o.requests.values) + 2 }, (_, r) => {
    const card = createCard(kit, 0.86, 0.032, (ctx, w, h) => {
      const row = phase().rows[r];
      if (!row) return;
      const write = pen(ctx, fonts, w, h);
      const faint = row.state !== "running";
      write(row.label, 0.015, 0.74, 0.56, faint ? kit.palette.faint : kit.palette.ink);
      row.blocks.forEach((id, k) => write(String(id), 0.25 + k * 0.028, 0.74, 0.56, faint ? kit.palette.faint : k < row.shared ? kit.palette.blue : kit.palette.ink));
      ctx.textAlign = "right";
      write(row.state === "waiting" ? `waits: needs ${row.need} free ${row.need === 1 ? "block" : "blocks"}` : `${row.state === "finished" ? "finished · " : ""}reused ${row.reused} · new ${row.prefilled}`, 0.985, 0.74, 0.56, row.state === "waiting" ? kit.palette.pink : faint ? kit.palette.faint : kit.palette.ink);
      ctx.textAlign = "left";
    });
    card.group.position.set(0.5, 0, rowZ(r));
    root.add(card.group);
    return card;
  });
  const readout = readoutCard(kit, fonts, root, 0.045, () => o.lines(phase(), step, knobs));
  const cards: ReturnType<typeof createCard>[] = [];
  const redraw = () => {
    readout.redraw();
    rows.forEach((c) => c.redraw());
    slots.forEach((s) => s.tag.redraw());
    cards.forEach((c) => c.redraw());
  };
  const knob = (k: PcKnob, center: number, key: keyof typeof knobs) => {
    const { rail, card } = miniRail(kit, fonts, root, center, 0.215, k.values.length, k.label, () => k.text(knobs[key]));
    cards.push(card);
    return picker(rail, k.start, k.start, Number.POSITIVE_INFINITY, (i) => {
      if (knobs[key] === k.values[i]) return;
      knobs[key] = k.values[i];
      phases = poolRun(knobs);
      redraw();
    });
  };
  const picks = [knob(o.shared, 0.165, "shared"), knob(o.requests, 0.385, "requests"), knob(o.pool, 0.605, "pool")];
  const { rail: stepRail, card: stepCard } = miniRail(kit, fonts, root, 0.835, 0.215, o.steps.length, "what happens", () => o.steps[step]);
  cards.push(stepCard);
  picks.push(
    picker(stepRail, 0, o.restStep, 4.2, (i) => {
      step = i;
      redraw();
    }),
  );
  return {
    root,
    picks,
    update: (dt, reduced) => {
      picks.forEach((p) => p.update(dt, reduced));
      const now = phase();
      rows.forEach((c, r) => {
        c.group.visible = r < now.rows.length;
      });
      slots.forEach((s, i) => {
        const slot = now.slots[i];
        const visible = i < knobs.pool && slot !== undefined;
        s.tag.group.visible = visible;
        s.h = approach(s.h, slot ? height(slot) : H * 0.25, dt * 1.4, reduced);
        s.variants.forEach((v, t) => {
          lay(v, X0 + pitch * i + GAP, pitch - 2 * GAP, s.h, Z, D);
          v.visible = visible && (["ink", "blue", "glass", "pink"] as const)[t] === tone[slot.state];
        });
      });
    },
  };
}

function joinMulti(parts: Multi[]): BookModel {
  let held: ReturnType<typeof picker> | null = null;
  return {
    exhibits: parts.map((p) => p.root),
    handles: parts.flatMap((p) => p.picks.map((k) => k.knob)),
    dragStart: (handle) => {
      held = parts.flatMap((p) => p.picks).find((k) => k.knob === handle) ?? null;
      held?.start();
    },
    drag: (x) => held?.drag(x),
    dragEnd: () => {
      held?.end();
      held = null;
    },
    enter: () => parts.forEach((p) => p.picks.forEach((k) => k.enter())),
    update: (dt, exhibit, _stop, reduced) => parts[exhibit]?.update(dt, reduced),
  };
}

const multi = (p: Part): Multi => ({ root: p.root, picks: [p.pick], update: p.update });

export function createPrefixModel(kit: Kit, fonts: Fonts, n: PcNumbers): BookModel {
  return joinMulti([multi(poolExhibit(kit, fonts, n.tables)), multi(poolExhibit(kit, fonts, n.pool)), multi(lanesExhibit(kit, fonts, n.ttft)), multi(lanesExhibit(kit, fonts, n.burst)), poolSandbox(kit, fonts, n.sandbox)]);
}
