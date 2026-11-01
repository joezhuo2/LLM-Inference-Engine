"use client";

import clsx from "clsx";
import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import type { TileValues } from "@/lib/dashboard/stats";
import type { Feed } from "./feed";
import { LiveConnection, type LiveStatus } from "./live";

export const DEFAULT_URL = "http://localhost:8000";

const noSubscribe = () => () => {};

export function useLive(feed: Feed, active: boolean, setTiles: (t: TileValues) => void) {
  const [status, setStatus] = useState<LiveStatus>({ kind: "idle" });
  const live = useMemo(() => (active ? new LiveConnection(feed, setStatus, setTiles) : null), [active, feed, setTiles]);

  useEffect(() => () => live?.stop(), [live]);

  return { live, status: active ? status : ({ kind: "idle" } as LiveStatus) };
}

const tone: Record<LiveStatus["kind"], string> = {
  idle: "text-muted",
  connecting: "text-muted",
  connected: "text-accent",
  unreachable: "text-warn",
  blocked: "text-warn",
  "not-engine": "text-warn",
  lost: "text-warn",
};

function statusLine(s: LiveStatus): string {
  switch (s.kind) {
    case "idle":
      return "not connected";
    case "connecting":
      return `connecting to ${s.url}`;
    case "connected":
      return `live: ${String(s.meta?.model ?? "engine")} on ${String(s.meta?.gpu ?? "an unknown GPU")}`;
    case "unreachable":
      return `nothing answered at ${s.url}`;
    case "blocked":
      return s.reason === "cors" ? `${s.url} answered, but the browser blocked the response (CORS)` : s.reason === "mixed-content" ? "this HTTPS page cannot load plain http:// from another machine" : `the browser blocked this page from reaching ${s.url}, or nothing is running there`;
    case "not-engine":
      return `${s.url} does not look like the engine: ${s.detail}`;
    case "lost":
      return `the engine stopped answering after step ${s.atStep}; retrying every 3 s`;
  }
}

export function ConnectBar({ live, status }: { live: LiveConnection | null; status: LiveStatus }) {
  const [url, setUrl] = useState(DEFAULT_URL);
  const busy = status.kind === "connecting" || status.kind === "connected" || status.kind === "lost";
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (busy) live?.stop();
        else live?.connect(url);
      }}
      className="flex min-w-0 flex-1 flex-wrap items-center gap-2"
    >
      <input
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        disabled={busy}
        spellCheck={false}
        aria-label="Engine address"
        className="w-56 min-w-0 rounded border border-border bg-bg px-2 py-1 text-text disabled:opacity-60"
      />
      <button type="submit" disabled={!live} className={clsx("pressable rounded border border-border px-2.5 py-1 disabled:opacity-40", busy ? "text-text" : "text-accent")}>
        {busy ? "disconnect" : "connect"}
      </button>
      <span role="status" className={clsx("min-w-0", tone[status.kind])}>
        {statusLine(status)}
      </span>
    </form>
  );
}

function Code({ children }: { children: string }) {
  return <code className="whitespace-pre-wrap break-words rounded bg-bg px-1.5 py-0.5 text-text">{children}</code>;
}

export function ConnectHelp({ status }: { status: LiveStatus }) {
  const origin = useSyncExternalStore(
    noSubscribe,
    () => location.origin,
    () => "",
  );
  if (status.kind === "connected" || status.kind === "connecting") return null;
  const blocked = status.kind === "blocked" ? status.reason : null;
  return (
    <section aria-label="How to connect" className="space-y-2 rounded border border-border bg-surface p-3 leading-relaxed">
      <p className="text-text">Live mode reads an engine running on your own machine. Nothing leaves your computer: this page talks to the engine directly.</p>
      <ol className="list-decimal space-y-1.5 pl-5 text-muted">
        <li>
          Start the server with this page&apos;s origin allowed: <Code>{`build/release/bin/engine serve --model models/tinyllama --cors-origin ${origin || "<this page's origin>"}`}</Code>
        </li>
        <li>Wait for it to load the model, then press connect. The default address is {DEFAULT_URL}.</li>
      </ol>
      {blocked === "cors" && <p className="text-warn">The engine is running but did not allow {origin}. Restart it with that exact origin in --cors-origin.</p>}
      {(blocked === "private-network" || status.kind === "unreachable") && (
        <p className="text-warn">
          {status.kind === "unreachable" ? "Check that the engine is running and listening on that port. " : ""}
          From an HTTPS page, Chrome asks before letting a site reach your local network: allow it if asked. Other browsers may block it outright; then run the site yourself (npm run dev in web/) and connect from http://localhost:3000.
        </p>
      )}
      {blocked === "mixed-content" && <p className="text-warn">Use the engine on this machine (localhost), or run the site yourself over http.</p>}
      <p className="text-muted">The server has no authentication. Keep it on localhost; do not expose the GPU to the internet.</p>
    </section>
  );
}

export function LiveBar({ feed, status }: { feed: Feed; status: LiveStatus }) {
  const text = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    let raf = 0;
    const paint = () => {
      raf = 0;
      const m = feed.model;
      if (text.current) text.current.textContent = m.lastIndex < 0 ? "no steps yet: the engine sends one when it serves a request" : `step ${m.lastIndex.toLocaleString("en-US")} · ${m.missed ? `${m.missed.toLocaleString("en-US")} steps missed` : "no steps missed"} · server time ${m.last ? m.last.end_s.toFixed(1) : "–"} s`;
    };
    const unsubscribe = feed.subscribe(() => {
      if (!raf) raf = requestAnimationFrame(paint);
    });
    paint();
    return () => {
      unsubscribe();
      cancelAnimationFrame(raf);
    };
  }, [feed]);
  return (
    <section aria-label="Live status" className="sticky bottom-0 z-10 flex flex-wrap items-center gap-x-4 gap-y-1 rounded border border-border bg-bg/90 px-3 py-2 backdrop-blur md:static md:bg-surface md:backdrop-blur-none">
      <span className={clsx(status.kind === "connected" ? "text-accent" : "text-muted")}>{status.kind === "connected" ? "live" : status.kind === "lost" ? "reconnecting" : "offline"}</span>
      <span ref={text} className="tabular-nums text-muted" />
      <span className="text-muted">metrics every 1 s, steps as they happen</span>
    </section>
  );
}
