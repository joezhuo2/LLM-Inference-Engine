"use client";

import type { ReactNode } from "react";
import { createPortal } from "react-dom";
import { timeAbout, timeAboutCaption } from "@/components/kernels/where-the-time-went/about";
import { FloorFigure, NextFigure } from "@/components/kernels/where-the-time-went/figures";
import { timeCaptions, timeSteps } from "@/components/kernels/where-the-time-went/story";
import { chapterNumber } from "@/content/book-order";
import { eagerStage, load, prefillLengths, prefillPaths, prefillPerToken, stepGroups, stepStages, whereTheTimeWent } from "@/content/kernels/where-the-time-went";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview, withLead } from "../chapter";
import { createTimeAboutModel } from "./where-the-time-went-about-model";
import { createTimeModel } from "./where-the-time-went-model";

const beats: Beat[][] = [
  [
    overview("One decode step, profile by profile"),
    close("Close up: the GEMMs keep their length, the blue and pink ends shrink", [0.62, 0.03, -0.33], 0.7, 0.36, -16, 42),
    close("From the side: the step without the GEMMs, drawn larger", [0.4, 0.03, -0.1], 0.74, 0.36, 22, 32),
    close("The rail: drag from the first profile to today", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
  [flat("One token today, and the part no kernel can remove")],
  [
    overview("One prompt, four ways to its first token"),
    close("Close up: the tall bars walk every key once per row", [0.34, 0.08, -0.24], 0.6, 0.46, -18, 36),
    close("From the side: the tiled kernel beside llama.cpp", [0.7, 0.05, -0.2], 0.56, 0.44, 24, 30),
    close("The rail: drag the prompt from short to long", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
  [
    overview("Requests served in time against requests offered"),
    close("Close up: where the bars leave the ink line", [0.62, 0.08, -0.26], 0.66, 0.42, -16, 36),
    close("From the side: the systems at the highest rates", [0.68, 0.08, -0.24], 0.62, 0.44, 26, 30),
    close("The rail: drag the offered load", [0.5, 0.03, 0.14], 0.9, 0.44, 0, 46),
  ],
  [flat("Three next steps, and what each could move")],
];

const instructions = [
  "Drag the pink ball from the first profile to today: watch which blocks shrink and which stay.",
  "Read the bar from the left: the blue part is the read of the weights.",
  "Drag the pink ball to change the prompt length: two bars grow, two stay level.",
  "Drag the pink ball along the offered load and read the card.",
  "Compare the three bars, then read what each would move.",
];

const figures: Record<string, () => ReactNode> = {
  floor: () => <FloorFigure />,
  next: () => <NextFigure />,
};

function TimeOverlay({ kind, target, narrow }: OverlayProps) {
  if (!target || !kind) return null;
  const figure = figures[kind];
  if (!figure) return null;
  return createPortal(<div className={narrow ? "aspect-[560/400] w-full" : "h-[34rem]"}>{figure()}</div>, target);
}

const chapter: ChapterDef = {
  slug: "where-the-time-went",
  number: chapterNumber("where-the-time-went"),
  title: "where the time went",
  formula: whereTheTimeWent.formula,
  steps: timeSteps,
  captions: timeCaptions,
  exhibits: [0, null, 1, 2, null],
  overlays: [null, "floor", null, null, "next"],
  instructions,
  beats,
  workbench: [],
  buildModel: (kit, fonts) => createTimeModel(kit, fonts, { stages: stepStages, prefill: { lengths: prefillLengths, paths: prefillPaths, perToken: prefillPerToken }, load: { rates: load.rates, date: load.date, commit: load.commit, ttftMs: load.slo.ttft_ms, tpotMs: load.slo.tpot_ms, series: load.series } }),
  Overlay: TimeOverlay,
};

export const whereTheTimeWentChapter = withLead(chapter, {
  step: timeAbout,
  caption: timeAboutCaption,
  instruction: "Drag the pink ball to pick an operation: its block rises, and the card gives its share of the kernel time.",
  beats: [
    overview("One decode step: the token, the span, the busy time, the ops"),
    close("Close up: the token is longer than the span, the span longer than the busy time", [0.72, 0.03, -0.3], 0.5, 0.4, -16, 42),
    close("From the side: the kernel time split by operation", [0.4, 0.03, -0.1], 0.74, 0.34, 22, 32),
    close("The rail: drag to pick an operation", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
  buildModel: (kit, fonts) => createTimeAboutModel(kit, fonts, { tpotMs: eagerStage.tpot ?? 0, span: eagerStage.span, busy: eagerStage.busy, kernels: eagerStage.kernels, idlePct: eagerStage.idlePct, groups: stepGroups.map((g) => ({ label: g.label, us: eagerStage.groups[g.id], ops: [...g.ops] })) }),
});
