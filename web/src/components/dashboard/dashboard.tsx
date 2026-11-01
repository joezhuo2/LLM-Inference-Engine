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
import { Toggle } from "@/components/kernels/toggle";
import { ConnectBar, ConnectHelp, DEFAULT_URL, LiveBar, useLive } from "./connect";
import { Controls } from "./controls";
import { Playground } from "./playground";
import { KvPool, poolWindow } from "./kv-pool";
import { Requests } from "./requests";
import { Timeline } from "./timeline";

const replayWindows = { rate: "last 1 s of the run", latency: "requests finished so far" };

const liveWindows = { rate: "engine, last 10 s", latency: "engine, last 60 s" };

const modes = ["replay", "live"];

function useReplay(trace: Trace | null, feed: Feed, active: boolean, setTiles: (t: TileValues) => void, rootRef: React.RefObject<HTMLElement | null>) {
  const player = useMemo(() => (trace ? new ReplayPlayer(trace) : null), [trace]);
  const [, setTick] = useState(0);
  const introDone = useIntroDone();
  const reduced = usePrefersReducedMotion();
  const autoplayed = useRef(false);

  useEffect(() => (active ? player?.attach(feed, setTiles, () => setTick((n) => n + 1)) : undefined), [player, feed, active, setTiles]);

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
    if (!player || !active || !introDone || autoplayed.current) return;
    autoplayed.current = true;
    if (reduced) player.seekStep(Math.round(player.total / 2));
    else player.play();
  }, [player, active, introDone, reduced]);

  return player;
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
  const [mode, setMode] = useState("replay");
  const [tiles, setTiles] = useState<TileValues>(emptyTiles);
  const trace = useTrace();
  const replaying = mode === "replay";
  const player = useReplay(trace, feed, replaying, setTiles, rootRef);
  const { live, status } = useLive(feed, !replaying, setTiles);
  const maxBatch = replaying ? Number(trace?.meta.max_batch ?? 16) : Number(status.kind === "connected" ? (status.meta?.max_batch ?? 16) : 16);
  const url = status.kind === "idle" ? DEFAULT_URL : status.url;

  const switchMode = (next: string) => {
    if (next === mode) return;
    feed.switchTo(next === "live", emptyModel());
    setTiles(emptyTiles);
    setMode(next);
  };

  return (
    <div ref={rootRef} className="space-y-3 font-mono text-xs">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <Toggle label="mode" choices={modes} value={mode} onChange={switchMode} />
        {replaying ? (
          <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
            <span className="rounded border border-warn/40 px-1.5 py-0.5 text-warn">recording</span>
            <span className="text-muted">{label}</span>
          </div>
        ) : (
          <ConnectBar live={live} status={status} />
        )}
      </div>
      {!replaying && <ConnectHelp status={status} />}
      <Tiles tiles={tiles} windows={replaying ? replayWindows : liveWindows} />
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
          <div className="h-72 lg:h-[20rem]">
            <Timeline feed={feed} maxBatch={maxBatch} />
          </div>
        </Panel>
        <Panel title="KV block pool" legend={<span className="text-muted">a color per request</span>}>
          <div className="h-72 lg:h-[20rem]">
            <KvPool key={mode} feed={feed} minWindow={replaying ? poolWindow(player?.highWater ?? 0) : poolWindow(0)} />
          </div>
        </Panel>
        {replaying ? (
          <Panel title="requests" legend={<span className="text-muted">one lane per color</span>} className="md:col-span-2 lg:col-span-1">
            <div className="lg:h-[20rem]">
              <Requests feed={feed} total={trace?.requests.length ?? 0} />
            </div>
          </Panel>
        ) : (
          <Panel title="chat" legend={<span className="text-muted">streams from the engine</span>} className="md:col-span-2 lg:col-span-1">
            <div className="h-[22rem] lg:h-[20rem]">
              <Playground feed={feed} url={url} connected={status.kind === "connected"} />
            </div>
          </Panel>
        )}
      </div>
      {replaying ? <Controls player={player} feed={feed} /> : <LiveBar feed={feed} status={status} />}
    </div>
  );
}
