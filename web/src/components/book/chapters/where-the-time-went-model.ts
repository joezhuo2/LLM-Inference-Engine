import * as THREE from "three";
import type { BookModel } from "../book-scene";
import { createCard, inkSegments, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";

export type TimeStage = { date: string; title: string; what: string; commit: string; kernels: number; span: number; tpot: number | null; idle: number; replay: boolean; groups: { gemm: number; attention: number; norm: number; rope: number; swiglu: number; rest: number } };
export type TimePrefill = { lengths: string[]; paths: { id: string; label: string; ms: number[] }[]; perToken: number[][] };
export type TimeLoad = { rates: number[]; date: string; commit: string; ttftMs: number; tpotMs: number; series: { id: string; label: string; goodput: number[]; ttftP99: number[] }[] };
export type TimeNumbers = { stages: TimeStage[]; prefill: TimePrefill; load: TimeLoad };

export const fmt = (v: number, digits = 0) => v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
export const approach = (value: number, goal: number, dt: number, reduced: boolean) => (reduced ? goal : value + (goal - value) * Math.min(1, dt * 2.2));

export type Pen = (text: string, x: number, y: number, size: number, color: string, serif?: boolean) => void;
export const pen =
  (ctx: CanvasRenderingContext2D, fonts: Fonts, w: number, h: number): Pen =>
  (text, x, y, size, color, serif = false) => {
    ctx.fillStyle = color;
    ctx.font = `${h * size}px ${serif ? fonts.serif : fonts.sans}`;
    ctx.fillText(text, w * x, h * y);
  };

export const unitBox = new THREE.BoxGeometry(1, 1, 1);

export function lay(object: THREE.Object3D, x: number, w: number, h: number, z: number, d: number) {
  object.visible = w > 0.0006;
  object.scale.set(Math.max(0.0001, w), h, d);
  object.position.set(x + w / 2, h / 2 + 0.003, z);
}

export function liquid(material: THREE.Material) {
  const mesh = new THREE.Mesh(unitBox, material);
  mesh.castShadow = true;
  return mesh;
}

export function makeRail(kit: Kit, fonts: Fonts, root: THREE.Group, z: number, xs: number[], label: string, range: string) {
  const a = Math.min(...xs);
  const b = Math.max(...xs);
  const rail = kit.solid(new THREE.BoxGeometry(b - a + 0.04, 0.016, 0.02));
  rail.position.set((a + b) / 2, 0.008, z);
  const ticks: number[] = [];
  for (const x of xs) ticks.push(x, 0.017, z - 0.018, x, 0.017, z + 0.018);
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.028, 28, 22), kit.pink);
  knob.castShadow = true;
  knob.position.set(xs[0], 0.045, z);
  const card = createCard(kit, 0.6, 0.045, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write(label, 0.02, 0.7, 0.5, kit.palette.faint);
    ctx.textAlign = "right";
    write(range, 0.98, 0.7, 0.5, kit.palette.pink);
    ctx.textAlign = "left";
  });
  card.group.position.set(0.5, 0, z + 0.06);
  root.add(rail, inkSegments(ticks, kit.thin), knob, card.group);
  const nearest = (px: number) => xs.reduce((best, x, i) => (Math.abs(x - px) < Math.abs(xs[best] - px) ? i : best), 0);
  return { knob, xs, a, b, nearest };
}

export function picker(rail: ReturnType<typeof makeRail>, start: number, rest: number, dwell: number, onChange: (index: number) => void) {
  const count = rail.xs.length;
  let index = -1;
  let held = false;
  let chosen = false;
  let clock = 0;
  let knobX = rail.xs[rest];
  const show = (i: number) => {
    if (i === index) return;
    index = i;
    onChange(i);
  };
  show(rest);
  rail.knob.position.x = rail.xs[rest];
  return {
    knob: rail.knob,
    index: () => index,
    start: () => {
      held = true;
      chosen = true;
    },
    drag: (px: number) => {
      knobX = Math.min(rail.b, Math.max(rail.a, px));
      show(rail.nearest(knobX));
    },
    end: () => {
      held = false;
    },
    enter: () => {
      clock = 0;
      chosen = false;
    },
    update: (dt: number, reduced: boolean) => {
      if (!chosen) {
        if (reduced) show(rest);
        else {
          clock += dt;
          show((start + Math.floor(clock / dwell)) % count);
        }
      }
      rail.knob.position.x = held ? knobX : approach(rail.knob.position.x, rail.xs[index], dt * 0.7, reduced);
    },
  };
}

const smallOf = (s: TimeStage) => s.groups.norm + s.groups.rope + s.groups.swiglu + s.groups.rest;

function stagesExhibit(kit: Kit, fonts: Fonts, stages: TimeStage[]) {
  const root = new THREE.Group();
  const X0 = 0.08;
  const W = 0.84;
  const Z1 = -0.31;
  const Z2 = -0.08;
  const D = 0.07;
  const H = 0.05;
  const GAP = 0.004;
  const total = (s: TimeStage) => s.groups.gemm + s.groups.attention + smallOf(s) + s.idle;
  const rest = (s: TimeStage) => s.groups.attention + smallOf(s) + s.idle;
  const whole = Math.max(...stages.map(total));
  const zoomed = Math.max(...stages.map(rest));
  const s1 = W / whole;
  const s2 = (W - 5 * GAP) / zoomed;
  const wash = washPlane(kit.palette.blue, 0.9, 0.5, 0.1);
  wash.position.set(0.5, 0.002, -0.2);
  root.add(wash);

  let shown = stages[stages.length - 1];
  const title = createCard(kit, 0.86, 0.1, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write(`${shown.replay ? "" : `${shown.date}: `}${shown.title}`, 0.03, 0.42, 0.32, kit.palette.ink, true);
    write(`${shown.what} · ${shown.kernels} kernels · commit ${shown.commit}`, 0.03, 0.84, 0.24, kit.palette.ink);
  });
  title.group.position.set(0.5, 0, -0.57);

  const note = (z: number, lines: [string, string][]) => {
    const card = createCard(kit, 0.84, 0.062, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      lines.forEach(([text, color], i) => write(text, 0.02, 0.42 + i * 0.44, 0.37, color));
    });
    card.group.position.set(0.5, 0, z);
    root.add(card.group);
  };
  note(-0.43, [
    ["one decode step to scale: GEMMs ink, attention blue, small pink, gaps glass", kit.palette.ink],
    ["the outline on the page: the length of the first profile", kit.palette.faint],
  ]);
  note(-0.2, [
    [`the same step without the GEMMs, drawn ${fmt(s2 / s1, 1)} times as large`, kit.palette.ink],
    ["blue: attention · pink: norm and add, RoPE, SwiGLU, cache write · glass: gaps", kit.palette.faint],
  ]);

  const firstLength = total(stages[0]) * s1;
  const zb = Z1 - D / 2 - 0.008;
  const zf = Z1 + D / 2 + 0.008;
  root.add(inkSegments([X0, 0.004, zb, X0 + firstLength, 0.004, zb, X0 + firstLength, 0.004, zb, X0 + firstLength, 0.004, zf, X0 + firstLength, 0.004, zf, X0, 0.004, zf, X0, 0.004, zf, X0, 0.004, zb], kit.thin));

  const glass = () => kit.solid(unitBox, { material: kit.glass, shadow: false });
  const row1 = [kit.solid(unitBox), liquid(kit.blue), liquid(kit.pink), glass()];
  const row2 = [liquid(kit.blue), liquid(kit.pink), liquid(kit.pink), liquid(kit.pink), liquid(kit.pink), glass()];
  root.add(title.group, ...row1, ...row2);

  const values = (s: TimeStage) => [s.groups.gemm, s.groups.attention, smallOf(s), s.idle, s.groups.norm, s.groups.rope, s.groups.swiglu, s.groups.rest];
  const live = values(shown);

  const readout = createCard(kit, 0.86, 0.12, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write(`GEMMs ${fmt(shown.groups.gemm)} µs · attention ${fmt(shown.groups.attention)} µs · small kernels ${fmt(smallOf(shown))} µs · gaps ${fmt(shown.idle)} µs`, 0.03, 0.33, 0.21, kit.palette.ink, true);
    write(`span ${fmt(shown.span)} µs${shown.tpot === null ? "" : `, ${fmt(shown.tpot, 2)} ms per token in the same run`}`, 0.03, 0.62, 0.2, kit.palette.blue);
    write(`norm and add ${fmt(shown.groups.norm)} · RoPE ${fmt(shown.groups.rope)} · SwiGLU ${fmt(shown.groups.swiglu)} · cache write, embedding ${fmt(shown.groups.rest)} µs`, 0.03, 0.89, 0.18, kit.palette.pink);
  });
  readout.group.position.set(0.5, 0, 0.1);
  root.add(readout.group);

  const rail = makeRail(kit, fonts, root, 0.27, stages.map((_, i) => 0.2 + (0.6 * i) / (stages.length - 1)), "which profile", `first to today, ${stages.length} of them`);
  const pick = picker(rail, 0, stages.length - 1, 3.4, (i) => {
    shown = stages[i];
    title.redraw();
    readout.redraw();
  });

  return {
    root,
    pick,
    update: (dt: number, reduced: boolean) => {
      pick.update(dt, reduced);
      const goal = values(shown);
      for (let i = 0; i < live.length; i++) live[i] = approach(live[i], goal[i], dt * 0.9, reduced);
      let x = X0;
      [live[0], live[1], live[2], live[3]].forEach((us, i) => {
        lay(row1[i], x, us * s1, i === 3 ? H * 0.4 : H, Z1, D);
        x += us * s1;
      });
      x = X0;
      [live[1], live[4], live[5], live[6], live[7], live[3]].forEach((us, i) => {
        lay(row2[i], x, us * s2, i === 5 ? H * 0.4 : H, Z2, D);
        x += us * s2 + GAP;
      });
    },
  };
}

function prefillExhibit(kit: Kit, fonts: Fonts, p: TimePrefill) {
  const root = new THREE.Group();
  const xs = p.paths.map((_, i) => 0.17 + (0.66 * i) / (p.paths.length - 1));
  const Z = -0.27;
  const SIDE = 0.1;
  const HMAX = 0.26;
  const top = Math.max(...p.perToken.flat());
  const wash = washPlane(kit.palette.blue, 0.9, 0.36, 0.1);
  wash.position.set(0.5, 0.002, Z);
  root.add(wash);

  let k = p.lengths.length - 1;
  const title = createCard(kit, 0.86, 0.1, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("one prompt alone: the time to its first token, four ways", 0.03, 0.42, 0.32, kit.palette.ink, true);
    write("a bar's height is ms per prompt token; the cards give the ms to the first token", 0.03, 0.84, 0.24, kit.palette.ink);
  });
  title.group.position.set(0.5, 0, -0.57);
  root.add(title.group);

  const material = (id: string) => (id === "tiled" ? kit.blue : id === "first" ? kit.pink : null);
  const color = (id: string) => (id === "tiled" ? kit.palette.blue : id === "first" ? kit.palette.pink : kit.palette.ink);
  const bars = p.paths.map((path) => {
    const m = material(path.id);
    const bar = m ? liquid(m) : kit.solid(unitBox, path.id === "llama" ? { material: kit.glass } : {});
    root.add(bar);
    return bar;
  });
  const tags = p.paths.map((path, i) => {
    const card = createCard(kit, 0.21, 0.11, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      const words = path.label.split(", ");
      write(words[0], 0.06, 0.3, 0.23, color(path.id), true);
      if (words[1]) write(words[1], 0.06, 0.55, 0.19, kit.palette.faint);
      write(`${fmt(path.ms[k], 1)} ms`, 0.06, 0.88, 0.28, color(path.id));
    });
    card.group.position.set(xs[i], 0, Z + 0.1);
    root.add(card.group);
    return card;
  });

  const at = (id: string) => p.paths.find((x) => x.id === id)?.ms[k] ?? 0;
  const readout = createCard(kit, 0.86, 0.1, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write(`${p.lengths[k]} prompt tokens: ${fmt(at("paged"), 1)} ms with one walk per row, ${fmt(at("tiled"), 1)} ms tiled, ${fmt(at("paged") / at("tiled"), 1)} times less`, 0.03, 0.42, 0.26, kit.palette.ink, true);
    write(`llama.cpp ${fmt(at("llama"), 1)} ms in the same session · the first flash kernel ${fmt(at("first"), 1)} ms`, 0.03, 0.84, 0.21, kit.palette.ink);
  });
  readout.group.position.set(0.5, 0, 0.11);
  root.add(readout.group);

  const rail = makeRail(kit, fonts, root, 0.27, p.lengths.map((_, i) => 0.2 + (0.6 * i) / (p.lengths.length - 1)), "prompt tokens", `${p.lengths[0]} to ${p.lengths[p.lengths.length - 1]}`);
  const heights = p.paths.map((_, i) => (HMAX * p.perToken[i][k]) / top);
  const pick = picker(rail, p.lengths.length - 1, p.lengths.length - 1, 3.6, (i) => {
    k = i;
    tags.forEach((t) => t.redraw());
    readout.redraw();
  });

  return {
    root,
    pick,
    update: (dt: number, reduced: boolean) => {
      pick.update(dt, reduced);
      bars.forEach((bar, i) => {
        heights[i] = approach(heights[i], (HMAX * p.perToken[i][k]) / top, dt * 0.9, reduced);
        lay(bar, xs[i] - SIDE / 2, SIDE, heights[i], Z, SIDE);
      });
    },
  };
}

function loadExhibit(kit: Kit, fonts: Fonts, l: TimeLoad) {
  const root = new THREE.Group();
  const top = l.rates[l.rates.length - 1];
  const xOf = (q: number) => 0.14 + (0.72 * (q - l.rates[0])) / (top - l.rates[0]);
  const Z = -0.25;
  const HMAX = 0.24;
  const peak = Math.max(top, ...l.series.flatMap((s) => s.goodput));
  const hOf = (v: number) => (HMAX * v) / peak;
  const BAR = 0.022;
  const wash = washPlane(kit.palette.pink, 0.9, 0.3, 0.08);
  wash.position.set(0.5, 0.002, Z);
  root.add(wash);

  const title = createCard(kit, 0.86, 0.12, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write(`goodput against offered load, measured ${l.date} at commit ${l.commit}`, 0.03, 0.33, 0.26, kit.palette.ink, true);
    write(`a bar: requests/s with the first token in ${fmt(l.ttftMs)} ms, then ${fmt(l.tpotMs)} ms per token at most`, 0.03, 0.63, 0.19, kit.palette.ink);
    write("the ink line: every offered request served in time", 0.03, 0.9, 0.19, kit.palette.faint);
  });
  title.group.position.set(0.5, 0, -0.57);
  root.add(title.group);

  const material = (id: string) => (id === "vllm" ? kit.pink : id === "chunked" ? kit.blue : null);
  const color = (id: string) => (id === "vllm" ? kit.palette.pink : id === "chunked" ? kit.palette.blue : kit.palette.ink);
  const offset = (i: number) => (i - (l.series.length - 1) / 2) * (BAR + 0.004);
  l.series.forEach((s, i) => {
    l.rates.forEach((q, r) => {
      const m = material(s.id);
      const bar = m ? liquid(m) : kit.solid(unitBox);
      lay(bar, xOf(q) + offset(i) - BAR / 2, BAR, Math.max(0.002, hOf(s.goodput[r])), Z, 0.05);
      root.add(bar);
    });
  });
  const zl = Z - 0.04;
  root.add(inkSegments([xOf(l.rates[0]), hOf(l.rates[0]), zl, xOf(top), hOf(top), zl, xOf(l.rates[0]), 0.004, zl, xOf(l.rates[0]), hOf(l.rates[0]), zl, xOf(top), 0.004, zl, xOf(top), hOf(top), zl, 0.1, 0.004, Z + 0.035, 0.9, 0.004, Z + 0.035], kit.thin));
  const marker = kit.solid(new THREE.BoxGeometry(l.series.length * (BAR + 0.004) + 0.02, 0.004, 0.09), { shadow: false });
  marker.position.set(xOf(l.rates[0]), 0.003, Z);
  root.add(marker);

  let r = l.rates.length - 1;
  const readout = createCard(kit, 0.86, 0.2, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write(`offered: ${l.rates[r]} requests per second`, 0.03, 0.2, 0.15, kit.palette.ink, true);
    l.series.forEach((s, i) => write(`${s.label}: ${fmt(s.goodput[r], 2)} req/s in time · TTFT p99 ${fmt(s.ttftP99[r])} ms`, 0.03, 0.43 + i * 0.22, 0.115, color(s.id)));
  });
  readout.group.position.set(0.5, 0, 0.04);
  root.add(readout.group);

  const rail = makeRail(kit, fonts, root, 0.28, l.rates.map(xOf), "offered load, requests per second", `${l.rates[0]} to ${top}`);
  const pick = picker(rail, 0, l.rates.length - 1, 2.8, (i) => {
    r = i;
    readout.redraw();
  });

  return {
    root,
    pick,
    update: (dt: number, reduced: boolean) => {
      pick.update(dt, reduced);
      marker.position.x = approach(marker.position.x, xOf(l.rates[r]), dt * 1.4, reduced);
    },
  };
}

export function createTimeModel(kit: Kit, fonts: Fonts, n: TimeNumbers): BookModel {
  const parts = [stagesExhibit(kit, fonts, n.stages), prefillExhibit(kit, fonts, n.prefill), loadExhibit(kit, fonts, n.load)];
  let held: (typeof parts)[number] | null = null;
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
