"use client";

import { createPortal } from "react-dom";
import { chunkScores } from "@/components/kernels/attention-v2/figures";
import { attentionV2Captions, attentionV2Numbers, attentionV2Steps, AttentionV2Profile, AttentionV2Step, V1V2Picker } from "@/components/kernels/attention-v2/story";
import { attentionV2 } from "@/content/kernels/attention-v2";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview } from "../chapter";
import { createAttentionV2Model } from "./attention-v2-model";

const beats: Beat[][] = [
  [
    overview("v1: one block, three passes over 2,048 tokens"),
    close("Close up: every score waits in shared memory", [0.5, 0.05, -0.4], 0.9, 0.34, -18, 40),
    close("From the side: in pass 3 only 4 of 16 pegs stand", [0.4, 0.04, -0.16], 0.7, 0.4, 28, 40),
    close("The output: 64 values from 64 threads", [0.5, 0.03, 0.08], 0.9, 0.34, 0, 44),
  ],
  [
    overview("Shared memory per block against the context"),
    close("Close up: the cap post at 2,048 tokens", [0.32, 0.04, -0.4], 0.56, 0.36, -22, 40),
    close("From the side: both bars stop far short of the 48 KB wall", [0.62, 0.04, -0.3], 0.7, 0.46, 28, 38),
    close("The rail: drag the context from 128 to 8,192", [0.5, 0.03, 0.2], 0.9, 0.34, 0, 46),
  ],
  [
    overview("Four chunks, one pass, three running values"),
    close("Close up: the pink line is the running max", [0.46, 0.08, -0.38], 0.84, 0.4, -16, 34),
    close("From the side: l and acc shrink when m rises", [0.48, 0.06, 0.04], 0.84, 0.42, 24, 36),
  ],
  [
    overview("One key, eight lanes, three shuffles"),
    close("Close up: each row is one 16-byte read", [0.36, 0.03, -0.32], 0.62, 0.42, -22, 36),
    close("From the side: the shuffle tree, 8 to 4 to 2 to 1", [0.4, 0.03, 0.15], 0.8, 0.42, 24, 38),
  ],
  [flat("Nsight Compute, one launch each")],
  [flat("The decode step, before and after")],
  [flat("Pick a size")],
];

const instructions = [
  "Watch the three passes, and how many pegs stay up in the last one.",
  "Drag the pink ball along the rail to change the context.",
  "Watch m rise, then l and acc shrink before each chunk is added.",
  "Watch the lanes read one row at a time, then the shuffles add their sums.",
  "Compare each pair of bars: v1 above, v2 below.",
  "Compare each pair of bars: before the switch above, after below.",
  "Pick a row count and a context to compare the measured times.",
];

function AttentionV2Overlay({ kind, target, narrow }: OverlayProps) {
  if (!target) return null;
  if (kind === "profile") return createPortal(<AttentionV2Profile />, target);
  if (kind === "step") return createPortal(<AttentionV2Step />, target);
  if (kind === "picker")
    return createPortal(
      <div className={narrow ? undefined : "h-[34rem]"}>
        <V1V2Picker />
      </div>,
      target,
    );
  return null;
}

export const attentionV2Chapter: ChapterDef = {
  slug: "attention-v2",
  number: 6,
  title: "attention v2",
  formula: attentionV2.formula,
  steps: attentionV2Steps,
  captions: attentionV2Captions,
  exhibits: [0, 1, 2, 3, null, null, null],
  overlays: [null, null, null, null, "profile", "step", "picker"],
  instructions,
  beats,
  workbench: [],
  buildModel: (kit, fonts) => createAttentionV2Model(kit, fonts, { ...attentionV2Numbers, scores: chunkScores }),
  Overlay: AttentionV2Overlay,
};
