import { DashModel } from "@/lib/dashboard/model";
import { readSse } from "@/lib/dashboard/sse";
import type { TileValues } from "@/lib/dashboard/stats";
import type { TraceStep } from "@/lib/trace";
import { type Feed, SLOTS } from "./feed";

export type Metrics = {
  version: number;
  time_s: number;
  meta?: Record<string, string | number | boolean>;
  requests: { waiting: number; running: number; finished_total: number; preemptions_total: number };
  kv: { blocks_total: number; blocks_free: number; block_size: number };
  tokens: { output_total: number; output_per_s: number };
  latency: { ttft_s: { p50: number; p99: number } | null; tpot_s: { p50: number; p99: number } | null };
  steps_total: number;
};

export type Blocked = "cors" | "private-network" | "mixed-content";

export type LiveStatus =
  | { kind: "idle" }
  | { kind: "connecting"; url: string }
  | { kind: "connected"; url: string; meta: Metrics["meta"] }
  | { kind: "unreachable"; url: string; privateNetwork: boolean }
  | { kind: "blocked"; url: string; reason: Blocked }
  | { kind: "not-engine"; url: string; detail: string }
  | { kind: "lost"; url: string; atStep: number };

const KEEP_STEPS = 3000;
const RETRY_MS = 3000;
const POLL_MS = 1000;
const TIMEOUT_MS = 4000;

export function normalizeUrl(raw: string): string | null {
  try {
    const url = new URL(raw.trim().includes("://") ? raw.trim() : `http://${raw.trim()}`);
    if (url.protocol !== "http:" && url.protocol !== "https:") return null;
    return url.origin + url.pathname.replace(/\/+$/, "");
  } catch {
    return null;
  }
}

export function isPrivateHost(url: string): boolean {
  const host = new URL(url).hostname.replace(/^\[|\]$/g, "");
  return host === "localhost" || host.endsWith(".localhost") || host === "::1" || /^127\./.test(host) || /^10\./.test(host) || /^192\.168\./.test(host) || /^172\.(1[6-9]|2\d|3[01])\./.test(host) || host.endsWith(".local");
}

function isLoopback(url: string): boolean {
  const host = new URL(url).hostname.replace(/^\[|\]$/g, "");
  return host === "localhost" || host.endsWith(".localhost") || host === "::1" || /^127\./.test(host);
}

async function getJson<T>(url: string, signal: AbortSignal): Promise<T> {
  const res = await fetch(url, { signal: AbortSignal.any([signal, AbortSignal.timeout(TIMEOUT_MS)]), cache: "no-store" });
  if (!res.ok) throw new HttpError(res.status, await res.text().catch(() => ""));
  return (await res.json()) as T;
}

class HttpError extends Error {
  readonly status: number;
  constructor(status: number, body: string) {
    super(`HTTP ${status}${body ? `: ${body.slice(0, 120)}` : ""}`);
    this.status = status;
  }
}

export function metricsTiles(m: Metrics): TileValues {
  return {
    tokensPerS: m.tokens.output_per_s,
    running: m.requests.running,
    waiting: m.requests.waiting,
    kvUsed: m.kv.blocks_total - 1 - m.kv.blocks_free,
    kvTotal: m.kv.blocks_total,
    ttft: m.latency.ttft_s,
    tpot: m.latency.tpot_s,
  };
}

export class LiveConnection {
  status: LiveStatus = { kind: "idle" };
  metricsAt = 0;
  private readonly feed: Feed;
  private readonly onStatus: (s: LiveStatus) => void;
  private readonly onTiles: (t: TileValues) => void;
  private abort = new AbortController();
  private poll = 0;
  private retry = 0;
  private raf = 0;
  private url = "";

  constructor(feed: Feed, onStatus: (s: LiveStatus) => void, onTiles: (t: TileValues) => void) {
    this.feed = feed;
    this.onStatus = onStatus;
    this.onTiles = onTiles;
  }

  private set(status: LiveStatus) {
    this.status = status;
    this.onStatus(status);
  }

  async connect(raw: string) {
    this.stop();
    const url = normalizeUrl(raw);
    if (!url) return this.set({ kind: "not-engine", url: raw, detail: "that is not an http:// or https:// address" });
    this.url = url;
    this.abort = new AbortController();
    const signal = this.abort.signal;
    if (location.protocol === "https:" && url.startsWith("http:") && !isLoopback(url)) return this.set({ kind: "blocked", url, reason: "mixed-content" });
    this.set({ kind: "connecting", url });
    const problem = await this.probe(url, signal);
    if (signal.aborted) return;
    if (problem) return this.set(problem);
    await this.start(url, signal, true);
  }

  private async probe(url: string, signal: AbortSignal): Promise<LiveStatus | null> {
    try {
      const health = await getJson<{ status?: string }>(`${url}/health`, signal);
      if (health.status !== "ok") return { kind: "not-engine", url, detail: `/health answered status "${String(health.status)}"` };
      return null;
    } catch (e) {
      if (signal.aborted) return null;
      if (e instanceof HttpError) return { kind: "not-engine", url, detail: `/health answered ${e.message}` };
      if (e instanceof SyntaxError) return { kind: "not-engine", url, detail: "/health did not answer JSON" };
    }
    const privateNetwork = location.protocol === "https:" && isPrivateHost(url);
    try {
      await fetch(`${url}/health`, { mode: "no-cors", signal: AbortSignal.any([signal, AbortSignal.timeout(TIMEOUT_MS)]), cache: "no-store" });
      return { kind: "blocked", url, reason: "cors" };
    } catch {
      return privateNetwork ? { kind: "blocked", url, reason: "private-network" } : { kind: "unreachable", url, privateNetwork };
    }
  }

  private async start(url: string, signal: AbortSignal, fresh: boolean) {
    let metrics: Metrics;
    try {
      metrics = await getJson<Metrics>(`${url}/metrics`, signal);
    } catch (e) {
      if (!signal.aborted) this.set({ kind: "not-engine", url, detail: `/metrics: ${e instanceof Error ? e.message : String(e)}` });
      return;
    }
    if (fresh || this.feed.model.totalBlocks !== metrics.kv.blocks_total) this.feed.reset(this.newModel(metrics));
    this.publish(metrics);
    this.set({ kind: "connected", url, meta: metrics.meta });
    this.poll = window.setInterval(() => this.pollMetrics(url, signal), POLL_MS);
    this.stream(url, signal);
  }

  private newModel(m: Metrics) {
    return new DashModel({ totalBlocks: m.kv.blocks_total, blockSize: m.kv.block_size, slots: SLOTS, keepSteps: KEEP_STEPS });
  }

  private publish(m: Metrics) {
    this.metricsAt = performance.now();
    this.onTiles(metricsTiles(m));
  }

  private async pollMetrics(url: string, signal: AbortSignal) {
    try {
      this.publish(await getJson<Metrics>(`${url}/metrics`, signal));
    } catch {
      if (!signal.aborted) this.lose();
    }
  }

  private async stream(url: string, signal: AbortSignal) {
    try {
      const res = await fetch(`${url}/v1/steps`, { signal, cache: "no-store", headers: { Accept: "text/event-stream" } });
      if (!res.ok || !res.body) throw new HttpError(res.status, "");
      await readSse(
        res.body,
        (e) => {
          if (e.event === "step") this.receive(JSON.parse(e.data) as TraceStep);
        },
        signal,
      );
    } catch {
      if (signal.aborted) return;
    }
    if (!signal.aborted) this.lose();
  }

  private receive(step: TraceStep) {
    const model = this.feed.model;
    if (model.lastIndex >= 0 && step.index <= model.lastIndex) this.feed.model = new DashModel({ totalBlocks: model.totalBlocks, blockSize: model.blockSize, slots: SLOTS, keepSteps: KEEP_STEPS });
    this.feed.model.apply(step);
    if (!this.raf) {
      this.raf = requestAnimationFrame(() => {
        this.raf = 0;
        this.feed.emit();
      });
    }
  }

  private lose() {
    if (this.status.kind !== "connected") return;
    const url = this.url;
    this.halt();
    this.abort = new AbortController();
    const signal = this.abort.signal;
    this.set({ kind: "lost", url, atStep: this.feed.model.lastIndex });
    const attempt = async () => {
      if (signal.aborted) return;
      const problem = await this.probe(url, signal);
      if (signal.aborted) return;
      if (problem) {
        this.retry = window.setTimeout(attempt, RETRY_MS);
        return;
      }
      await this.start(url, signal, false);
    };
    this.retry = window.setTimeout(attempt, RETRY_MS);
  }

  private halt() {
    this.abort.abort();
    clearInterval(this.poll);
    clearTimeout(this.retry);
    cancelAnimationFrame(this.raf);
    this.raf = 0;
  }

  stop() {
    this.halt();
    if (this.status.kind !== "idle") this.set({ kind: "idle" });
  }
}
