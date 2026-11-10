"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Controls } from "@/components/dashboard/controls";
import { Key, Panel } from "@/components/dashboard/dashboard";
import { emptyModel, Feed } from "@/components/dashboard/feed";
import { KvPool, poolWindow } from "@/components/dashboard/kv-pool";
import { ReplayPlayer } from "@/components/dashboard/replay-player";
import { Timeline } from "@/components/dashboard/timeline";
import { preferredTrace } from "@/lib/data";
import { useTrace } from "@/lib/trace";

const info = preferredTrace();

export const replayLabel = info ? `${String(info.meta.date ?? "").slice(0, 10)} on the ${String(info.meta.gpu).replace(/^NVIDIA (GeForce )?/, "")}: ${info.requests} chat requests at ${String(info.meta.request_rate)} per second, max_batch ${String(info.meta.max_batch)}` : "no recorded run in results/traces";
export const replayFacts = info ? { requests: info.requests, steps: info.steps, rate: Number(info.meta.request_rate), maxBatch: Number(info.meta.max_batch), seen: info.max_batch_seen, blocks: Number(info.meta.kv_blocks) } : null;

export function ServingReplay({ pages }: { pages?: { replay: HTMLElement | null } }) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [feed] = useState(() => new Feed(emptyModel()));
  const [, setTick] = useState(0);
  const trace = useTrace();
  const player = useMemo(() => (trace ? new ReplayPlayer(trace) : null), [trace]);
  const started = useRef(false);

  useEffect(() => {
    if (!player) return;
    const detach = player.attach(
      feed,
      () => {},
      () => setTick((n) => n + 1),
    );
    if (!started.current) {
      started.current = true;
      player.seekStep(Math.round(player.total / 2));
    }
    return detach;
  }, [player, feed]);

  const target = pages ? pages.replay : undefined;

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
  }, [player, target]);

  const maxBatch = Number(trace?.meta.max_batch ?? 16);
  const body = (
    <div ref={rootRef} className="space-y-3 tabular-nums text-xs">
      <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
        <span className="rounded border border-warn/40 px-1.5 py-0.5 text-warn">recording</span>
        <span className="text-muted">{replayLabel}</span>
      </div>
      <div className={pages ? "grid gap-2" : "grid gap-2 md:grid-cols-2"}>
        <Panel
          title="batch timeline"
          legend={
            <span className="flex gap-3">
              <Key color="var(--series-2)" label="prefill" />
              <Key color="var(--series-1)" label="decode" />
            </span>
          }
        >
          <div className={pages ? "h-52" : "h-72"}>
            <Timeline feed={feed} maxBatch={maxBatch} />
          </div>
        </Panel>
        <Panel title="KV block pool" legend={<span className="text-muted">a color per request</span>}>
          <div className={pages ? "h-60" : "h-72"}>
            <KvPool feed={feed} minWindow={poolWindow(player?.highWater ?? 0)} />
          </div>
        </Panel>
      </div>
      <Controls player={player} feed={feed} />
    </div>
  );
  if (pages) return pages.replay ? createPortal(body, pages.replay) : null;
  return body;
}
