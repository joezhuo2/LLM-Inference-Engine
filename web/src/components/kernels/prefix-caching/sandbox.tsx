"use client";

import clsx from "clsx";
import { useState } from "react";
import NumberFlow from "@/components/number-flow";
import { pcFmt as fmt, prefix, settings } from "@/content/kernels/prefix-caching";
import { MirrorRange } from "../mirror";
import { Toggle } from "../toggle";
import { shareRun } from "./share";

const ease = "transition-all duration-700 ease-in-out motion-reduce:transition-none";
const owns = ["16", "64", "256"];
const arrivals = ["one at a time", "all at once"];

export function PrefixSandbox() {
  const [shared, setShared] = useState(prefix.largest.shared);
  const [own, setOwn] = useState(String(prefix.own));
  const [arrive, setArrive] = useState(arrivals[0]);
  const atOnce = arrive === arrivals[1];
  const ownTokens = Number(own);
  const run = shareRun({ shared, own: ownTokens, requests: prefix.requests, atOnce, budget: prefix.budget, maxBatch: prefix.maxBatch, block: prefix.block });
  const prompt = shared + ownTokens;
  const odd = shared % prefix.block;
  const measured = settings.find((s) => s.cache === "on" && s.shared === shared && s.own === ownTokens && (s.rate === 0) === atOnce);
  const tiles = [
    { label: "cached tokens per hit", value: run.perHit },
    { label: "requests that find nothing", value: run.firstStep },
    { label: "hit rate, %", value: 100 * run.hitRate, digits: 1 },
    { label: "tokens to prefill, all requests", value: run.prefill },
  ];
  return (
    <div className="space-y-4 text-xs tabular-nums">
      <div className="space-y-2">
        <MirrorRange id="prefix-shared" label="shared" min={0} max={1024} step={8} value={shared} onValue={setShared} text={fmt(shared)} />
        <div className="flex flex-wrap gap-x-6 gap-y-2">
          <Toggle label="own tokens" choices={owns} value={own} onChange={setOwn} />
          <Toggle label="arrivals" choices={arrivals} value={arrive} onChange={setArrive} />
        </div>
      </div>
      <div className="space-y-1">
        <div className="flex justify-between gap-2 text-muted">
          <span>{`a later request's ${fmt(prompt)} prompt tokens`}</span>
          <span>{`${fmt(run.perHit)} from the cache`}</span>
        </div>
        <div role="img" aria-label={`${fmt(run.perHit)} of ${fmt(prompt)} prompt tokens come from the cache and ${fmt(prompt - run.perHit)} are prefilled`} className="flex h-5 w-full overflow-hidden rounded-sm border border-border">
          <span className={clsx("block h-full bg-text/35", ease)} style={{ width: `${(run.perHit / prompt) * 100}%` }} />
          <span className={clsx("block h-full bg-warn/70", ease)} style={{ width: `${((shared - run.perHit) / prompt) * 100}%` }} />
          <span className={clsx("block h-full bg-accent", ease)} style={{ width: `${(ownTokens / prompt) * 100}%` }} />
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-muted">
          <span><span className="mr-1 inline-block size-2 bg-text/35" />shared, cached</span>
          <span><span className="mr-1 inline-block size-2 bg-warn/70" />shared, but in a block with its own tokens</span>
          <span><span className="mr-1 inline-block size-2 bg-accent" />its own</span>
        </div>
        <p className="min-h-8 text-muted">
          {odd ? `${fmt(shared)} is not a whole number of ${prefix.block}-token blocks: the last ${odd} shared tokens sit in a block with the request's own, so that block's key differs and it is computed.` : `${fmt(shared)} is ${fmt(shared / prefix.block)} whole blocks, all of them shared.`}
        </p>
      </div>
      <div className="space-y-1">
        <div className="flex justify-between gap-2 text-muted">
          <span>{`${prefix.requests} requests, in arrival order`}</span>
          <span>{atOnce ? `the first step holds ${run.firstStep}` : "the first finds nothing"}</span>
        </div>
        <div role="img" aria-label={`${run.firstStep} of ${prefix.requests} requests find nothing in the cache and ${prefix.requests - run.firstStep} find ${fmt(run.perHit)} tokens`} className="grid grid-cols-16 gap-0.5">
          {run.cached.map((c, i) => (
            <span key={i} className="flex h-3 overflow-hidden rounded-[1px] border border-border">
              <span className={clsx("block h-full bg-text/35", ease)} style={{ width: `${(c / prompt) * 100}%` }} />
              <span className={clsx("block h-full flex-1 bg-accent", ease)} />
            </span>
          ))}
        </div>
      </div>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-4">
        {tiles.map((t) => (
          <div key={t.label}>
            <dt className="text-muted">{t.label}</dt>
            <dd className="font-serif text-xl text-text">
              <NumberFlow value={t.value} format={{ minimumFractionDigits: t.digits ?? 0, maximumFractionDigits: t.digits ?? 0 }} />
            </dd>
          </div>
        ))}
      </dl>
      <p className="min-h-8 leading-relaxed text-muted">
        {measured ? `Measured at this setting: a hit rate of ${fmt(100 * measured.hit_rate, 1)}%, ${fmt(measured.cached_tokens)} cached tokens, the same as the arithmetic.` : `Without the cache all ${fmt(run.total)} tokens are prefilled. The measured settings are ${prefix.own} own tokens with ${settings.filter((s) => s.cache === "on" && s.rate === 0).map((s) => fmt(s.shared)).join(", ")} shared.`}
      </p>
    </div>
  );
}
