import { DashModel } from "@/lib/dashboard/model";
import { replayTiles, stepsDoneBy, type TileValues } from "@/lib/dashboard/stats";
import type { Trace } from "@/lib/trace";
import { type Feed, SLOTS } from "./feed";

const tileEveryMs = 1000;

export class ReplayPlayer {
  readonly trace: Trace;
  readonly duration: number;
  readonly highWater: number;
  t = 0;
  speed = 1;
  playing = false;
  private feed: Feed | null = null;
  private readonly prompts: Map<number, number>;
  private onTiles: (tiles: TileValues) => void = () => {};
  private onState: () => void = () => {};
  private raf = 0;
  private last = 0;
  private lastTiles = -Infinity;
  private visible = true;

  constructor(trace: Trace) {
    this.trace = trace;
    this.duration = trace.steps.length ? trace.steps[trace.steps.length - 1].end_s : 0;
    this.prompts = new Map(trace.requests.map((r) => [r.id, r.prompt_tokens]));
    const full = this.freshModel();
    trace.steps.forEach((s) => full.apply(s));
    this.highWater = full.highWater;
  }

  attach(feed: Feed, onTiles: (tiles: TileValues) => void, onState: () => void) {
    this.feed = feed;
    this.onTiles = onTiles;
    this.onState = onState;
    feed.reset(this.freshModel());
    this.seekTime(this.t);
    return () => {
      this.dispose();
      this.feed = null;
    };
  }

  private get fed(): Feed {
    if (!this.feed) throw new Error("replay player is not attached to a feed");
    return this.feed;
  }

  get done(): number {
    return this.feed ? this.fed.model.lastIndex + 1 : 0;
  }

  get total(): number {
    return this.trace.steps.length;
  }

  private freshModel() {
    const { kv_blocks, block_size } = this.trace.meta;
    return new DashModel({ totalBlocks: Number(kv_blocks), blockSize: Number(block_size), slots: SLOTS, prompts: this.prompts });
  }

  play() {
    if (this.t >= this.duration) this.seekTime(0);
    this.playing = true;
    this.fed.playing = true;
    this.onState();
    this.start();
  }

  pause() {
    this.playing = false;
    this.fed.playing = false;
    cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.publishTiles();
    this.fed.emit();
    this.onState();
  }

  setSpeed(speed: number) {
    this.speed = speed;
    this.onState();
  }

  setVisible(visible: boolean) {
    this.visible = visible;
    if (visible) this.start();
  }

  seekStep(done: number) {
    const k = Math.max(0, Math.min(this.total, done));
    this.seekTime(k === 0 ? 0 : this.trace.steps[k - 1].end_s);
  }

  seekTime(t: number) {
    this.t = Math.max(0, Math.min(this.duration, t));
    const target = stepsDoneBy(this.trace.steps, this.t);
    if (target < this.done) this.fed.model = this.freshModel();
    this.advanceTo(target);
    this.publishTiles();
    this.fed.emit();
    this.onState();
  }

  dispose() {
    cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.playing = false;
  }

  private advanceTo(target: number) {
    const steps = this.trace.steps;
    for (let i = this.done; i < target; i++) this.fed.model.apply(steps[i]);
    const next = steps[target];
    if (next && this.t > next.start_s) {
      this.fed.pending = next;
      this.fed.progress = (this.t - next.start_s) / Math.max(1e-6, next.end_s - next.start_s);
    } else {
      this.fed.pending = null;
      this.fed.progress = 0;
    }
  }

  private publishTiles() {
    this.lastTiles = performance.now();
    this.onTiles(replayTiles(this.trace.steps, this.trace.requests, this.done, this.t, Number(this.trace.meta.kv_blocks)));
  }

  private start() {
    if (!this.feed || !this.playing || this.raf || !this.visible || document.hidden) return;
    this.last = performance.now();
    this.raf = requestAnimationFrame(this.tick);
  }

  private tick = (now: number) => {
    this.raf = 0;
    if (!this.feed || !this.playing || !this.visible || document.hidden) return;
    const dt = Math.min((now - this.last) / 1000, 0.1);
    this.last = now;
    this.t = Math.min(this.duration, this.t + dt * this.speed);
    this.advanceTo(stepsDoneBy(this.trace.steps, this.t));
    if (now - this.lastTiles >= tileEveryMs) this.publishTiles();
    this.fed.emit();
    if (this.t >= this.duration) {
      this.pause();
      return;
    }
    this.raf = requestAnimationFrame(this.tick);
  };
}
