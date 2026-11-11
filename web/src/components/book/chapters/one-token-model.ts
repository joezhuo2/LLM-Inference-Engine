import type { BookModel } from "../book-scene";
import type { Kit } from "../ink";
import type { Fonts } from "../typeset";
import { joinParts, type Segment, segmentExhibit } from "./segments";

export type TokenNumbers = {
  layers: number;
  layersText: string;
  layerUs: number;
  layerText: string;
  ops: Segment[];
  tokenUs: number;
  tokenText: string;
  encodeUs: number;
  encodeText: string;
  sumText: string;
  trip: Segment[];
};

export function createTokenModel(kit: Kit, fonts: Fonts, n: TokenNumbers): BookModel {
  const layer = segmentExhibit(kit, fonts, {
    title: "one layer of the decode step, operation by operation",
    sub: "every block is to scale by its kernel time at batch 1 (measured)",
    refs: [{ label: `${n.layers} layers, one after another, each the same ten operations`, text: n.layersText, value: n.layerUs, tone: "ink", count: n.layers }],
    row: { label: "one layer, to scale", text: n.layerText },
    segments: n.ops,
    total: n.layerUs,
    rail: "which operation",
  });
  const trip = segmentExhibit(kit, fonts, {
    title: "one token, from the ID going in to the ID coming out",
    sub: "every bar is to the same scale: the time a user waits for one token (measured)",
    refs: [
      { label: "time per token", text: n.tokenText, value: n.tokenUs, tone: "glass" },
      { label: "once per request, before the first step: encode the prompt", text: n.encodeText, value: n.encodeUs, tone: "blue" },
    ],
    row: { label: "the stages of one decode step, in order", text: n.sumText },
    segments: n.trip,
    total: n.tokenUs,
    rail: "which stage",
    rest: 1,
  });
  return joinParts([layer, trip]);
}
