"use client";

import { pcFmt as fmt, prefix } from "@/content/kernels/prefix-caching";
import { StickyStory, type StoryStep } from "../sticky-story";
import { ShareFigure } from "./figures";

const p = prefix.largest;

export const prefixAbout: StoryStep = {
  id: "about",
  title: "compute a shared start once",
  body: (
    <>
      <p>
        Many requests begin with the same tokens. A chat service puts the same system prompt in front of every user message, a second chat turn repeats the whole first turn, and several questions about one document all start with that document. Each of them pays the prefill of those tokens again.
      </p>
      <p>
        It does not have to. A token&apos;s keys and values depend only on that token and the ones before it, so the same start gives the same keys and values in every request. Keep them after the first request, and <span className="text-accent">a later request that starts the same way can use them</span> and prefill only the tokens that are its own.
      </p>
      <p>
        That is prefix caching. In this engine, at a shared start of {fmt(p.shared)} tokens and {prefix.own} tokens of its own, a request waited {fmt(p.quiet.off.later_ttft_ms)} ms for its first token without the cache and {fmt(p.quiet.on.later_ttft_ms)} ms with it (measured).
      </p>
    </>
  ),
};

export const prefixAboutCaption = `measured ${prefix.date} on the ${prefix.gpu}: ${p.quiet.on.requests} requests sharing a ${fmt(p.shared)}-token start, arriving ${prefix.quietArrivals.replace("/s", " a second")}; the median later request, median of ${p.quiet.on.runs} runs (results.md, Prefix caching); the blocks above are illustrative`;

export function PrefixAboutStory() {
  return <StickyStory kicker="the idea" steps={[prefixAbout]} caption={() => prefixAboutCaption} figure={() => <ShareFigure />} />;
}
