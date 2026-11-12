"use client";

import type { ReactNode } from "react";
import { createPortal } from "react-dom";
import { speculativeAbout, speculativeAboutCaption } from "@/components/kernels/speculative-decoding/about";
import { RowsFigure, TruncateFigure } from "@/components/kernels/speculative-decoding/figures";
import { speculativeCaptions, speculativeSteps } from "@/components/kernels/speculative-decoding/story";
import { chapterNumber } from "@/content/book-order";
import { piece, repeatRows, spec, specFmt, speculativeDecoding } from "@/content/kernels/speculative-decoding";
import type { SpeculativeStep } from "@/lib/data";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview, withLead } from "../chapter";
import { joinParts } from "./segments";
import { createSpeculativeModel, type SpecNumbers, type SpecStop, stepsExhibit } from "./speculative-decoding-model";

const fmt = specFmt;
const example = spec.example;
const all = [...example.prompt_tokens, ...example.output_tokens];
const prompt = example.prompt_tokens.length;
const text = (tokens: string[], most = 44) => {
  const joined = tokens.map(piece).join("").replace(/↵/g, " ").trim();
  return joined.length > most ? `${joined.slice(0, most - 1)}…` : joined;
};
const cell = (token: string) => piece(token).trim() || "␣";
const count = (n: number, word: string) => `${fmt(n)} ${word}${n === 1 ? "" : "s"}`;

const withDraft = example.steps.filter((s) => s.draft.length > 0);
const spread = Array.from({ length: Math.min(9, withDraft.length) }, (_, i) => withDraft[Math.round((i * (withDraft.length - 1)) / Math.max(1, Math.min(9, withDraft.length) - 1))]);
const picked = [...new Set([...spread, spec.bestStep, spec.missStep])].sort((a, b) => a.index - b.index);

const stop = (s: SpeculativeStep): SpecStop => {
  const k = s.draft.length;
  const lead = Math.min(2, s.match);
  const wrote = example.output_tokens.slice(s.at - prompt, s.at - prompt + s.kept + 1);
  return {
    index: s.index,
    rows: 1 + k,
    kept: s.kept,
    ngram: s.ngram,
    back: s.at - s.ngram - s.match,
    found: all.slice(s.match - lead, s.match + s.ngram + k).map(cell),
    lead,
    tail: all.slice(s.at - s.ngram - 1, s.at).map(cell),
    draft: s.draft.map(cell),
    from: s.at - prompt,
    width: s.kept + 1,
    lines: {
      steps: [`step ${s.index}: ${count(s.kept + 1, "token")} from one step`, `${count(k, "token")} guessed, the model agreed with ${fmt(s.kept)}`, `it wrote “${text(wrote)}”`],
      guess: [`step ${s.index}: the last ${count(s.ngram, "token")} appeared ${count(s.at - s.ngram - s.match, "token")} back`, `what followed is the guess: “${text(s.draft)}”`, `the model agreed with ${fmt(s.kept)} of ${fmt(k)}: the step wrote ${count(s.kept + 1, "token")}`],
      check: [
        `step ${s.index}: 1 real row and ${count(k, "guessed row")} of ${spec.maxBatch}`,
        `the model agreed with ${fmt(s.kept)}: ${count(s.kept + 1, "token")} written`,
        s.kept < k ? `${count(k - s.kept, "row")} dropped in pink, the cache slots given back` : "every guess was right: nothing to give back",
      ],
    },
  };
};

const numbers: SpecNumbers = {
  tokens: example.output_tokens.length,
  maxBatch: spec.maxBatch,
  rest: Math.max(0, picked.indexOf(spec.bestStep)),
  blocks: [{ from: 0, width: 1, kept: 0 }, ...example.steps.map((s) => ({ from: s.at - prompt, width: s.kept + 1, kept: s.kept }))],
  stops: picked.map(stop),
  bars: repeatRows.map((r) => {
    const chat = spec.chatAt(r.k);
    return {
      k: r.k,
      tokensPerStep: r.tokensPerStep,
      ms: r.ms,
      chatMs: chat?.ms ?? null,
      lines: [
        r.k === 0 ? "no guesses: plain decoding, one token per step" : `up to ${r.k === 1 ? "1 guess" : `${r.k} guesses`} per step`,
        `repeat answers: ${fmt(r.tokensPerStep, 2)} tokens per step, ${fmt(r.ms, 2)} ms per token${r.k > 0 ? `, ${fmt(r.speedup, 2)}×, ${fmt(r.acceptedPct)}% kept` : ""}`,
        chat ? `chat answers: ${fmt(chat.ms, 2)} ms per token${chat.k > 0 ? `, ${fmt(chat.acceptedPct)}% of the guesses kept` : ""}` : "chat answers: not measured at this setting",
      ],
    };
  }),
};

const beats: Beat[][] = [
  [flat("What a step costs, by its rows")],
  [
    overview("The last tokens, found earlier in the context"),
    close("Close up: the match in blue, what followed in pink", [0.5, 0.05, -0.25], 0.96, 0.44, -12, 36),
    close("From the side: the guess carried to the end", [0.56, 0.06, -0.2], 0.8, 0.44, 24, 26),
    close("The rail: drag to another step", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
  [
    overview("One step, sixteen rows"),
    close("Close up: kept rows in blue, dropped rows in pink", [0.5, 0.03, -0.24], 0.96, 0.52, -14, 38),
    close("From the side: the tokens that came out", [0.74, 0.04, -0.24], 0.5, 0.5, 28, 26),
    close("The rail: drag to another step", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
  [flat("The cache slots of a step with a wrong guess")],
  [
    overview("Tokens per step, and time per token"),
    close("Close up: tokens per step rise with the guesses", [0.5, 0.06, -0.34], 0.92, 0.3, -12, 34),
    close("From the side: repeat answers against chat answers", [0.5, 0.04, -0.12], 0.9, 0.3, 22, 28),
    close("The rail: drag to change the most guesses", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
];

const instructions = [
  "Compare the two bars at the top: sixteen rows cost little more than one.",
  "Drag the pink ball to another step: the blue tokens are the match, the pink ones the guess copied to the end.",
  "Drag the pink ball to another step: blue rows were right, pink rows were computed from a wrong guess and dropped.",
  "Find the dashed slots: they were reserved for guesses the model did not agree with.",
  "Drag the pink ball to allow more guesses per step: the blue bars are tokens per step.",
];

const figures: Record<string, () => ReactNode> = {
  rows: () => <RowsFigure />,
  truncate: () => <TruncateFigure />,
};

function SpeculativeOverlay({ kind, target, narrow }: OverlayProps) {
  if (!target || !kind) return null;
  const figure = figures[kind];
  if (!figure) return null;
  return createPortal(<div className={narrow ? "aspect-[560/400] w-full" : "h-[34rem]"}>{figure()}</div>, target);
}

const chapter: ChapterDef = {
  slug: "speculative-decoding",
  number: chapterNumber("speculative-decoding"),
  title: "speculative decoding",
  formula: speculativeDecoding.formula,
  steps: speculativeSteps,
  captions: speculativeCaptions,
  exhibits: [null, 0, 1, null, 2],
  overlays: ["rows", null, null, "truncate", null],
  instructions,
  beats,
  workbench: [],
  buildModel: (kit, fonts) => createSpeculativeModel(kit, fonts, numbers),
  Overlay: SpeculativeOverlay,
};

export const speculativeDecodingChapter = withLead(chapter, {
  step: speculativeAbout,
  caption: speculativeAboutCaption,
  instruction: "Drag the pink ball along the answer: each block is one step, and a wide, tall block wrote several tokens at once.",
  beats: [
    overview("The same answer, in fewer steps"),
    close("Close up: one token per step above, several below", [0.5, 0.04, -0.26], 0.94, 0.4, -12, 36),
    close("From the side: tall blocks kept more guesses", [0.4, 0.05, -0.15], 0.6, 0.3, 26, 22),
    close("The rail: drag along the answer", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
  buildModel: (kit, fonts) => joinParts([stepsExhibit(kit, fonts, numbers)]),
});
