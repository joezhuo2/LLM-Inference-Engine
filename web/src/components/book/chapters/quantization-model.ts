import * as THREE from "three";
import type { BookModel } from "../book-scene";
import { type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";
import { block, joinParts, type Part, readoutCard, type Reference, type Segment, segmentExhibit, tagCard, titleCard, W, X0 } from "./segments";
import { approach, lay, liquid, makeRail, picker } from "./where-the-time-went-model";

export type SegmentSet = { title: string; sub: string; refs: Reference[]; row: { label: string; text: string }; segments: Segment[]; rail: string; rest?: number };
export type RidgeStop = { label: string; intensity: number; lines: [string, string, string] };
export type QuantNumbers = { floor: SegmentSet; ridge: { value: number; text: string; stops: RidgeStop[] }; ladder: SegmentSet; round: SegmentSet; pool: SegmentSet; plan: SegmentSet };

export function setExhibit(kit: Kit, fonts: Fonts, s: SegmentSet): Part {
  const total = Math.max(s.segments.reduce((a, b) => a + b.value, 0), ...s.refs.map((r) => r.value));
  return segmentExhibit(kit, fonts, { title: s.title, sub: s.sub, refs: s.refs, row: s.row, segments: s.segments, total, rail: s.rail, rest: s.rest });
}

function ridgeExhibit(kit: Kit, fonts: Fonts, n: QuantNumbers["ridge"]): Part {
  const root = new THREE.Group();
  const Z1 = -0.3;
  const Z2 = -0.12;
  const D = 0.05;
  const H = 0.04;
  const scale = W / n.value;
  const wash = washPlane(kit.palette.blue, 0.9, 0.4, 0.1);
  wash.position.set(0.5, 0.002, -0.22);
  root.add(wash);
  titleCard(kit, fonts, root, "FLOP per byte of weights, against the ridge", "the bar is the work each byte of weights does; past the ridge the math would set the time (calculated)");
  tagCard(kit, fonts, root, Z1, "the ridge: measured peak ÷ measured bandwidth", n.text, kit.palette.ink);
  const ridge = block(kit, "glass");
  lay(ridge, X0, W, H, Z1, D);
  root.add(ridge);
  tagCard(kit, fonts, root, Z2, "this width and batch", "FLOP per byte", kit.palette.blue);
  const bar = liquid(kit.blue);
  root.add(bar);
  let chosen = 0;
  let width = Math.max(0.004, n.stops[0].intensity * scale);
  const readout = readoutCard(kit, fonts, root, 0.1, () => n.stops[chosen].lines);
  const rail = makeRail(kit, fonts, root, 0.27, n.stops.map((_, i) => 0.2 + (0.6 * i) / Math.max(1, n.stops.length - 1)), "width and batch", `${n.stops[0].label} …`);
  const pick = picker(rail, 0, n.stops.length - 1, 3, (i) => {
    chosen = i;
    readout.redraw();
  });
  return {
    root,
    pick,
    update: (dt, reduced) => {
      pick.update(dt, reduced);
      width = approach(width, Math.max(0.004, n.stops[chosen].intensity * scale), dt * 1.2, reduced);
      lay(bar, X0, width, H, Z2, D);
    },
  };
}

export function createQuantModel(kit: Kit, fonts: Fonts, n: QuantNumbers): BookModel {
  return joinParts([setExhibit(kit, fonts, n.floor), ridgeExhibit(kit, fonts, n.ridge), setExhibit(kit, fonts, n.ladder), setExhibit(kit, fonts, n.round), setExhibit(kit, fonts, n.pool), setExhibit(kit, fonts, n.plan)]);
}
