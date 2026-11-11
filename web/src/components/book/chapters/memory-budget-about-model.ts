import type { Kit } from "../ink";
import type { Fonts } from "../typeset";
import { type Part, type Segment, segmentExhibit } from "./segments";

export type TankNumbers = { vram: number; vramText: string; sumText: string; parts: Segment[] };

export function createTankExhibit(kit: Kit, fonts: Fonts, n: TankNumbers): Part {
  return segmentExhibit(kit, fonts, {
    title: "what the GPU's memory holds while the engine serves",
    sub: "every block is to scale by its bytes; the pool is as the engine allocated it (measured)",
    refs: [{ label: "the GPU's memory", text: n.vramText, value: n.vram, tone: "glass" }],
    row: { label: "what is in it, part by part", text: n.sumText },
    segments: n.parts,
    total: n.vram,
    rail: "which part",
    rest: 1,
  });
}
