"use client";

import { useEffect, useRef } from "react";
import { Toggle } from "@/components/kernels/toggle";
import type { Feed } from "./feed";
import type { ReplayPlayer } from "./replay-player";

const speeds = ["1x", "2x", "4x"];

export function Controls({ player, feed }: { player: ReplayPlayer | null; feed: Feed }) {
  const range = useRef<HTMLInputElement>(null);
  const counter = useRef<HTMLSpanElement>(null);
  const dragging = useRef(false);

  useEffect(() => {
    if (!player) return;
    let raf = 0;
    const paint = () => {
      raf = 0;
      const done = player.done;
      if (range.current && !dragging.current) range.current.value = String(done);
      if (counter.current) counter.current.textContent = `step ${done} of ${player.total} · ${player.t.toFixed(2)} of ${player.duration.toFixed(2)} s`;
    };
    const unsubscribe = feed.subscribe(() => {
      if (!raf) raf = requestAnimationFrame(paint);
    });
    paint();
    return () => {
      unsubscribe();
      cancelAnimationFrame(raf);
    };
  }, [player, feed]);

  const disabled = !player;
  const playing = player?.playing ?? false;
  const nudge = (d: number) => {
    if (!player) return;
    player.pause();
    player.seekStep(player.done + d);
  };

  return (
    <section aria-label="Replay controls" className="sticky bottom-0 z-10 flex flex-wrap items-center gap-x-4 gap-y-2 rounded border border-border bg-bg/90 px-3 py-2 backdrop-blur md:static md:bg-surface md:backdrop-blur-none">
      <div className="flex gap-1.5">
        <button
          type="button"
          disabled={disabled}
          onClick={() => (playing ? player?.pause() : player?.play())}
          aria-pressed={playing}
          className="pressable w-16 rounded border border-border px-2.5 py-1 text-accent disabled:opacity-40"
        >
          {playing ? "pause" : player && player.done >= player.total ? "replay" : "play"}
        </button>
        <button type="button" disabled={disabled} onClick={() => nudge(-1)} aria-label="previous step" className="pressable rounded border border-border px-2 py-1 text-text disabled:opacity-40">
          ‹
        </button>
        <button type="button" disabled={disabled} onClick={() => nudge(1)} aria-label="next step" className="pressable rounded border border-border px-2 py-1 text-text disabled:opacity-40">
          ›
        </button>
      </div>
      <Toggle label="speed" choices={speeds} value={`${player?.speed ?? 1}x`} onChange={(v) => player?.setSpeed(Number(v.replace("x", "")))} />
      <input
        ref={range}
        type="range"
        min={0}
        max={player?.total ?? 0}
        step={1}
        defaultValue={0}
        disabled={disabled}
        aria-label="scrub through the steps"
        onPointerDown={() => {
          dragging.current = true;
        }}
        onPointerUp={() => {
          dragging.current = false;
        }}
        onChange={(e) => player?.seekStep(Number(e.currentTarget.value))}
        className="min-w-40 flex-1 accent-[var(--accent)]"
      />
      <span ref={counter} className="shrink-0 tabular-nums text-muted" />
    </section>
  );
}
