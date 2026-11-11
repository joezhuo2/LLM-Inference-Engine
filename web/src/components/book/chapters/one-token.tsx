"use client";

import type { ReactNode } from "react";
import { createPortal } from "react-dom";
import { tokenAbout, tokenAboutCaption } from "@/components/kernels/one-token/about";
import { HeadFigure, TokenizeFigure } from "@/components/kernels/one-token/figures";
import { tokenCaptions, tokenSteps } from "@/components/kernels/one-token/story";
import { chapterNumber } from "@/content/book-order";
import { hostTimes, journey, layerOps, layerUs, layersUs, oneToken, shareText, stations, stepProfile, tokenFmt, tokenUs, trip, tripSum } from "@/content/kernels/one-token";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview, withLead } from "../chapter";
import { createStationsExhibit } from "./one-token-about-model";
import { createTokenModel } from "./one-token-model";
import { joinParts } from "./segments";

const fmt = tokenFmt;
const us = (v: number) => `${fmt(v, v < 100 ? 1 : 0)} µs`;

const beats: Beat[][] = [
  [flat("The question, cut into the tokens the model knows")],
  [
    overview("One layer, and the 22 it is one of"),
    close("Close up: four matrix products hold most of the layer", [0.5, 0.03, -0.14], 0.9, 0.36, -14, 40),
    close("From the side: the small kernels between them", [0.42, 0.03, -0.1], 0.6, 0.3, 24, 30),
    close("The rail: drag to pick an operation", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
  [flat("The largest of 32,000 scores becomes the ID")],
  [
    overview("One token, stage by stage, to scale"),
    close("Close up: the layers take almost the whole bar", [0.5, 0.03, -0.14], 0.9, 0.4, -14, 40),
    close("From the side: the slivers at the end of the step", [0.74, 0.03, -0.1], 0.5, 0.3, 24, 30),
    close("The rail: drag to pick a stage", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
];

const instructions = [
  "Read the pieces in order: the question sits between the template's markers.",
  "Drag the pink ball to pick an operation: its block rises, and the card gives its time in one layer and in all of them.",
  "Read the bars from the top: the first one wins by a wide margin.",
  "Drag the pink ball along the stages of a token, and compare each with the bar above.",
];

const figures: Record<string, () => ReactNode> = {
  tokenize: () => <TokenizeFigure />,
  head: () => <HeadFigure />,
};

function TokenOverlay({ kind, target, narrow }: OverlayProps) {
  if (!target || !kind) return null;
  const figure = figures[kind];
  if (!figure) return null;
  return createPortal(<div className={narrow ? "aspect-[560/400] w-full" : "h-[34rem]"}>{figure()}</div>, target);
}

const chapter: ChapterDef = {
  slug: "one-token",
  number: chapterNumber("one-token"),
  title: "one token, end to end",
  formula: oneToken.formula,
  steps: tokenSteps,
  captions: tokenCaptions,
  exhibits: [null, 0, null, 1],
  overlays: ["tokenize", null, "head", null],
  instructions,
  beats,
  workbench: [],
  buildModel: (kit, fonts) =>
    createTokenModel(kit, fonts, {
      layers: journey.layers,
      layersText: us(layersUs),
      layerUs,
      layerText: us(layerUs),
      ops: layerOps.map((o) => ({ label: o.label, value: o.layer, headline: `${o.label}: ${us(o.layer)} in one layer, ${us(o.step)} in ${journey.layers}`, lines: [o.does, `${fmt((o.step / layersUs) * 100, o.step / layersUs < 0.02 ? 1 : 0)}% of the layers' kernel time; in the profile: ${o.ops.join(", ")}`] })),
      tokenUs,
      tokenText: `${fmt(stepProfile.tpotMs, 2)} ms`,
      encodeUs: hostTimes.encode,
      encodeText: us(hostTimes.encode),
      sumText: us(tripSum),
      trip: trip.map((t) => ({ label: t.label, value: t.us, headline: `${t.label}: ${us(t.us)}, ${shareText(t.us)} of the token`, lines: [t.does, `runs on the ${t.where}`], tone: t.where === "host" ? "glass" : "ink" })),
    }),
  Overlay: TokenOverlay,
};

export const oneTokenChapter = withLead(chapter, {
  step: tokenAbout,
  caption: tokenAboutCaption,
  instruction: "Drag the pink ball from station to station: the card says what each takes in, what it gives and how long it takes.",
  beats: [
    overview("Seven stations between a question and one token of its answer"),
    close("Close up: the five stations on the GPU", [0.5, 0.03, -0.26], 0.74, 0.4, -14, 40),
    close("From the side: the token over its station", [0.4, 0.04, -0.24], 0.7, 0.36, 24, 30),
    close("The rail: drag to move the token", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
  buildModel: (kit, fonts) => joinParts([createStationsExhibit(kit, fonts, { token: journey.chosenText.trim(), tpotMs: fmt(stepProfile.tpotMs, 2), stations })]),
});
