"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { emptyTiles, type TileValues } from "@/lib/dashboard/stats";
import { useIntroDone } from "@/lib/intro";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import { type Trace, useTrace } from "@/lib/trace";
import { emptyModel, Feed } from "./feed";
import { ReplayPlayer } from "./replay-player";
import { Tiles } from "./tiles";

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

export function Dashboard({ label }: { label: string }) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [feed] = useState(() => new Feed(emptyModel()));
  const trace = useTrace();
  const { tiles } = useReplay(trace, feed, rootRef);

  return (
    <div ref={rootRef} className="space-y-3 font-mono text-xs">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="rounded border border-warn/40 px-1.5 py-0.5 text-warn">recording</span>
        <span className="text-muted">{label}</span>
      </div>
      <Tiles tiles={tiles} windows={replayWindows} />
    </div>
  );
}
