"use client";

import clsx from "clsx";
import { useEffect, useMemo, useRef, useState } from "react";
import { emptyTiles, type TileValues } from "@/lib/dashboard/stats";
import { useIntroDone } from "@/lib/intro";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import { type Trace, useTrace } from "@/lib/trace";
import { emptyModel, Feed } from "./feed";
import { ReplayPlayer } from "./replay-player";
import { Tiles } from "./tiles";
import { KvPool, poolWindow } from "./kv-pool";
import { Timeline } from "./timeline";

const replayWindows = { rate: "last 1 s of the run", latency: "requests finished so far" };

function useReplay(trace: Trace | null, feed: Feed, rootRef: React.RefObject<HTMLElement | null>) {
  const player = useMemo(() => (trace ? new ReplayPlayer(trace) : null), [trace]);
  const [tiles, setTiles] = useState<TileValues>(emptyTiles);
  const [, setTick] = useState(0);
  const introDone = useIntroDone();
  const reduced = usePrefersReducedMotion();
  const autoplayed = useRef(false);

  useEffect(() => player?.attach(feed, setTiles, () => setTick((n) => n + 1)), [player, feed]);

  useEffect(() => {
    const root = rootRef.current;
    if (!player || !root) return;
    let inView = true;
    const update = () => player.setVisible(inView && !document.hidden);
    const seen = new IntersectionObserver(([entry]) => {
      inView = entry.isIntersecting;
      update();
    });
    seen.observe(root);
    document.addEventListener("visibilitychange", update);
    return () => {
      seen.disconnect();
      document.removeEventListener("visibilitychange", update);
    };
  }, [player, rootRef]);

  useEffect(() => {
    if (!player || !introDone || autoplayed.current) return;
    autoplayed.current = true;
    if (reduced) player.seekStep(Math.round(player.total / 2));
    else player.play();
  }, [player, introDone, reduced]);

  return { player, tiles };
}

function Panel({ title, legend, children, className }: { title: string; legend?: React.ReactNode; children: React.ReactNode; className?: string }) {
  return (
    <section aria-label={title} className={clsx("flex min-w-0 flex-col rounded border border-border bg-surface p-3", className)}>
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h2 className="text-text">{title}</h2>
        {legend}
      </div>
      {children}
    </section>
  );
}

function Key({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5 text-muted">
      <span aria-hidden className="h-1 w-3 rounded-sm" style={{ background: color }} />
      {label}
    </span>
  );
}

export function Dashboard({ label }: { label: string }) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [feed] = useState(() => new Feed(emptyModel()));
  const trace = useTrace();
  const { player, tiles } = useReplay(trace, feed, rootRef);
  const maxBatch = Number(trace?.meta.max_batch ?? 16);

  return (
    <div ref={rootRef} className="space-y-3 font-mono text-xs">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="rounded border border-warn/40 px-1.5 py-0.5 text-warn">recording</span>
        <span className="text-muted">{label}</span>
      </div>
      <Tiles tiles={tiles} windows={replayWindows} />
      <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-3">
        <Panel
          title="batch timeline"
          legend={
            <span className="flex gap-3">
              <Key color="var(--series-2)" label="prefill" />
              <Key color="var(--series-1)" label="decode" />
            </span>
          }
        >
          <div className="h-72 lg:h-[22rem]">
            <Timeline feed={feed} maxBatch={maxBatch} />
          </div>
        </Panel>
        <Panel title="KV block pool" legend={<span className="text-muted">a color per request</span>}>
          <div className="h-72 lg:h-[22rem]">
            <KvPool feed={feed} minWindow={poolWindow(player?.highWater ?? 0)} />
          </div>
        </Panel>
      </div>
    </div>
  );
}
