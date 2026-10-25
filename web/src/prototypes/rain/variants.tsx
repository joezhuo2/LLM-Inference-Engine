"use client";

import { useRef } from "react";
import { type Frame, useCanvasLoop } from "../canvas";
import { createReplayer, loopTime, stepsDoneBy, type Trace, traceDuration } from "../trace";
import { HeroFrame, PlayButton } from "./hero";

type Sim = { advance: (t: number) => void };

function clean(text: string) {
  return text.replace(/\n/g, "↵").replace(/\s+/g, " ");
}

function hash(a: number, b: number) {
  let h = (a * 2654435761) ^ (b * 1597334677);
  h = Math.imul(h ^ (h >>> 15), 2246822519);
  return (h ^ (h >>> 13)) >>> 0;
}

function RainCanvas({
  trace,
  makeSim,
  render,
  caption,
}: {
  trace: Trace;
  makeSim: () => Sim;
  render: (frame: Frame, t: number) => void;
  caption: string;
}) {
  const duration = traceDuration(trace);
  const sim = useRef<Sim | null>(null);
  const { canvasRef, playing, play } = useCanvasLoop((frame) => {
    const t = loopTime(frame.elapsedS, duration);
    sim.current ??= makeSim();
    sim.current.advance(t);
    frame.ctx.clearRect(0, 0, frame.width, frame.height);
    render(frame, t);
  }, duration * 0.6);
  return (
    <HeroFrame caption={caption}>
      <canvas ref={canvasRef} aria-hidden="true" className="absolute inset-0 h-full w-full" />
      {!playing && <PlayButton onPlay={play} />}
    </HeroFrame>
  );
}

type Drop = { col: number; spawn: number; chars: string; first: boolean };

export function Cascade({ trace }: { trace: Trace }) {
  const drops = useRef<Drop[]>([]);
  const counts = useRef(new Map<number, number>());
  const cell = 16;
  const speed = 420;
  return (
    <RainCanvas
      trace={trace}
      caption="every falling word is a token the engine generated; heavier rain means more tokens per second, amber marks a request's first token"
      makeSim={() => ({ advance: createReplayerAdvance(trace, drops, counts) })}
      render={({ ctx, width, height, palette }, t) => {
        const ncols = Math.max(1, Math.floor(width / cell));
        ctx.font = `14px ${palette.mono}`;
        ctx.textBaseline = "top";
        const life = (height + 12 * cell) / speed;
        drops.current = drops.current.filter((d) => t - d.spawn < life && t >= d.spawn);
        for (const d of drops.current) {
          const x = (d.col % ncols) * cell;
          const head = (t - d.spawn) * speed;
          for (let i = d.chars.length - 1, k = 0; i >= 0; i--, k++) {
            const y = head - k * cell;
            if (y < -cell || y > height) continue;
            ctx.globalAlpha = Math.max(0, 1 - k * 0.14);
            ctx.fillStyle = k === 0 ? (d.first ? palette.warn : palette.text) : palette.accent;
            ctx.fillText(d.chars[i], x, y);
          }
        }
        ctx.globalAlpha = 1;
      }}
    />
  );
}

function createReplayerAdvance(
  trace: Trace,
  drops: { current: Drop[] },
  counts: { current: Map<number, number> },
) {
  return createReplayer(
    trace,
    (step) => {
      step.seqs.forEach((seq, j) => {
        const n = counts.current.get(seq) ?? 0;
        counts.current.set(seq, n + 1);
        const chars = step.tokens[j].text.replace(/\s+/g, "") || "·";
        drops.current.push({ col: hash(seq, n), spawn: step.end_s, chars, first: step.kind === "prefill" });
      });
    },
    () => {
      drops.current = [];
      counts.current.clear();
    },
  );
}

type Lane = { seq: number; text: string; started: number; finished?: number };

function useLaneSim(trace: Trace, slotCount: number) {
  const slots = useRef<(Lane | null)[]>([]);
  const make = (): Sim => {
    slots.current = Array.from({ length: slotCount }, () => null);
    const advance = createReplayer(
      trace,
      (step) => {
        step.seqs.forEach((seq, j) => {
          let lane = slots.current.find((l) => l?.seq === seq);
          if (!lane) {
            let i = slots.current.findIndex((l) => l === null);
            if (i < 0) i = slots.current.findIndex((l) => l?.finished !== undefined);
            if (i < 0) return;
            lane = { seq, text: "", started: step.end_s };
            slots.current[i] = lane;
          }
          lane.text = (lane.text + clean(step.tokens[j].text)).slice(-600);
        });
        for (const id of step.finished) {
          const lane = slots.current.find((l) => l?.seq === id);
          if (lane) lane.finished = step.end_s;
        }
      },
      () => {
        slots.current = Array.from({ length: slotCount }, () => null);
      },
    );
    return {
      advance: (t: number) => {
        advance(t);
        slots.current = slots.current.map((l) => (l && l.finished !== undefined && t - l.finished > 1.6 ? null : l));
      },
    };
  };
  return { slots, make };
}

export function Lanes({ trace }: { trace: Trace }) {
  const { slots, make } = useLaneSim(trace, 8);
  const waiting = useRef(0);
  return (
    <RainCanvas
      trace={trace}
      caption="one lane per request being served, its answer streaming in token by token; lanes free up when a request finishes"
      makeSim={() => {
        const sim = make();
        return {
          advance: (t) => {
            sim.advance(t);
            const done = stepsDoneBy(trace, t);
            waiting.current = done ? trace.steps[done - 1].waiting : 0;
          },
        };
      }}
      render={({ ctx, width, height, palette }, t) => {
        const top = 20;
        const area = height * 0.5;
        const rowH = Math.min(40, area / slots.current.length);
        ctx.font = `13px ${palette.mono}`;
        ctx.textBaseline = "middle";
        const charW = ctx.measureText("M").width;
        const left = 20;
        const textLeft = left + 7 * charW;
        const right = width - 20;
        const visible = Math.max(4, Math.floor((right - textLeft) / charW));
        slots.current.forEach((lane, i) => {
          const y = top + i * rowH + rowH / 2;
          ctx.globalAlpha = 1;
          ctx.fillStyle = palette.border;
          ctx.fillRect(left, y + rowH / 2 - 1, right - left, 1);
          if (!lane) return;
          const fade = lane.finished === undefined ? 1 : Math.max(0, 1 - (t - lane.finished) / 1.2);
          ctx.globalAlpha = fade;
          ctx.fillStyle = palette.muted;
          ctx.fillText(`#${String(lane.seq).padStart(2, "0")}`, left, y);
          const shown = lane.text.slice(-visible);
          const x0 = right - shown.length * charW;
          const fresh = lane.finished === undefined ? Math.min(shown.length, 24) : 0;
          ctx.fillStyle = palette.muted;
          ctx.fillText(shown.slice(0, shown.length - fresh), x0, y);
          ctx.fillStyle = palette.text;
          ctx.fillText(shown.slice(shown.length - fresh, shown.length - 2), x0 + (shown.length - fresh) * charW, y);
          ctx.fillStyle = palette.accent;
          if (fresh) ctx.fillText(shown.slice(-2), right - 2 * charW, y);
          if (t - lane.started < 0.5) {
            ctx.globalAlpha = 1 - (t - lane.started) / 0.5;
            ctx.fillStyle = palette.warn;
            ctx.fillRect(left - 8, y - rowH / 3, 3, (rowH * 2) / 3);
          }
        });
        ctx.globalAlpha = 1;
        ctx.textAlign = "right";
        ctx.fillStyle = palette.muted;
        ctx.fillText(`waiting ${waiting.current}`, right, top + slots.current.length * rowH + 14);
        ctx.textAlign = "left";
      }}
    />
  );
}

type Column = { seq: number; words: string[]; started: number; prompt: number; finished?: number };

export function Pulse({ trace }: { trace: Trace }) {
  const columns = useRef<(Column | null)[]>([]);
  const prompts = new Map(trace.requests.map((r) => [r.id, r.prompt_tokens]));
  const maxPrompt = Math.max(1, ...trace.requests.map((r) => r.prompt_tokens));
  const slotCount = 12;
  return (
    <RainCanvas
      trace={trace}
      caption="one column per request in the batch, newest token on top; the amber bar is its prompt being prefilled, grey means finished"
      makeSim={() => {
        columns.current = Array.from({ length: slotCount }, () => null);
        const advance = createReplayer(
          trace,
          (step) => {
            step.seqs.forEach((seq, j) => {
              let col = columns.current.find((c) => c?.seq === seq);
              if (!col) {
                const i = columns.current.findIndex(
                  (c) => c === null || (c.finished !== undefined && step.end_s - c.finished > 1.5),
                );
                if (i < 0) return;
                col = { seq, words: [], started: step.end_s, prompt: prompts.get(seq) ?? 0 };
                columns.current[i] = col;
              }
              const word = clean(step.tokens[j].text).trim();
              if (word) col.words.unshift(word);
              col.words.length = Math.min(col.words.length, 20);
            });
            for (const id of step.finished) {
              const col = columns.current.find((c) => c?.seq === id);
              if (col) col.finished = step.end_s;
            }
          },
          () => {
            columns.current = Array.from({ length: slotCount }, () => null);
          },
        );
        return {
          advance: (t) => {
            advance(t);
            columns.current = columns.current.map((c) =>
              c && c.finished !== undefined && t - c.finished > 1.5 ? null : c,
            );
          },
        };
      }}
      render={({ ctx, width, height, palette }, t) => {
        const shownSlots = Math.max(3, Math.min(slotCount, Math.floor(width / 84)));
        const colW = width / shownSlots;
        const lineH = 18;
        ctx.textBaseline = "top";
        columns.current.slice(0, shownSlots).forEach((col, i) => {
          if (!col) return;
          const x = i * colW + 12;
          const fade = col.finished === undefined ? 1 : Math.max(0, 1 - (t - col.finished) / 1.5);
          ctx.save();
          ctx.beginPath();
          ctx.rect(i * colW, 0, colW - 8, height);
          ctx.clip();
          ctx.globalAlpha = fade;
          ctx.font = `11px ${palette.mono}`;
          ctx.fillStyle = palette.muted;
          ctx.fillText(`#${String(col.seq).padStart(2, "0")}`, x, 16);
          const age = t - col.started;
          if (age < 0.6) {
            ctx.globalAlpha = fade * (1 - age / 0.6);
            ctx.fillStyle = palette.warn;
            ctx.fillRect(x, 32, ((colW - 28) * col.prompt) / maxPrompt, 4);
          }
          ctx.font = `13px ${palette.mono}`;
          col.words.forEach((w, k) => {
            ctx.globalAlpha = fade * Math.max(0, 1 - k / 16);
            ctx.fillStyle = col.finished !== undefined ? palette.muted : k === 0 ? palette.accent : palette.text;
            ctx.fillText(w, x, 44 + k * lineH);
          });
          ctx.restore();
        });
      }}
    />
  );
}
