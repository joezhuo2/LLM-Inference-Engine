import type { ComponentType } from "react";
import type { StoryStep } from "@/components/kernels/sticky-story";
import type { Beat, BookModel, Frame, Stop } from "./book-scene";
import type { Kit } from "./ink";
import type { Fonts, PageText, Run } from "./typeset";

export type WorkbenchPage = { title: string; overlay: string; caption: string; paragraphs: string[] };
export type OverlayProps = { kind: string | null; target: HTMLElement | null; narrow: boolean };

export type ChapterDef = {
  slug: string;
  number: number;
  title: string;
  formula?: string;
  steps: StoryStep[];
  captions: string[];
  exhibits: (number | null)[];
  overlays: (string | null)[];
  instructions: (string | undefined)[];
  beats: Beat[][];
  workbench: WorkbenchPage[];
  buildModel: (kit: Kit, fonts: Fonts) => BookModel;
  Overlay?: ComponentType<OverlayProps>;
  lead?: number;
};

export type LeadDef = {
  step: StoryStep;
  caption: string;
  instruction: string;
  beats: Beat[];
  buildModel: (kit: Kit, fonts: Fonts) => BookModel;
};

export type ChapterText = { title: string; paragraphs: Run[][] };

export const frames = {
  spread: { center: [0, 0, 0.02], width: 2.3, depth: 1.62, az: 0, el: 62 } as Frame,
  page: { center: [0.5, 0.03, -0.02], width: 1.06, depth: 1.42, az: 0, el: 62 } as Frame,
  flatRight: { center: [0, 0, 0], width: 2.08, depth: 1.42, az: 0, el: 89.6 } as Frame,
  flatNarrow: { center: [0.5, 0, 0], width: 1.02, depth: 1.42, az: 0, el: 89.6 } as Frame,
};

export const overview = (caption: string): Beat => ({ caption, frame: frames.spread, narrow: frames.page });
export const flat = (caption: string): Beat => ({ caption, frame: frames.flatRight, narrow: frames.flatNarrow });
export const close = (caption: string, center: [number, number, number], width: number, depth: number, az: number, el: number): Beat => {
  const frame: Frame = { center, width, depth, az, el };
  return { caption, frame, narrow: { ...frame, width: width * 0.9 } };
};

export const plainRuns = (paragraphs: string[]): Run[][] => paragraphs.map((text) => [{ text, accent: false }]);

export function chapterTitles(chapter: ChapterDef): string[] {
  return [...chapter.steps.map((s) => s.title), ...chapter.workbench.map((w) => w.title)];
}

export function stepNumber(chapter: ChapterDef, k: number): string {
  const lead = chapter.lead ?? 0;
  return k >= lead && k < chapter.steps.length ? String(k + 1 - lead) : "";
}

function joinLead(lead: BookModel, base: BookModel): BookModel {
  const own = new Set(lead.handles);
  let held: BookModel | null = null;
  return {
    exhibits: [lead.exhibits[0], ...base.exhibits],
    handles: [...lead.handles, ...base.handles],
    dragStart: (handle) => {
      held = own.has(handle) ? lead : base;
      held.dragStart?.(handle);
    },
    drag: (x) => held?.drag?.(x),
    dragEnd: () => {
      held?.dragEnd?.();
      held = null;
    },
    update: (dt, exhibit, stop, reduced) => (exhibit === 0 ? lead.update(dt, 0, 0, reduced) : base.update(dt, exhibit - 1, stop - 1, reduced)),
    enter: (stop) => (stop === 0 ? lead.enter?.(0) : base.enter?.(stop - 1)),
  };
}

export function withLead(chapter: ChapterDef, lead: LeadDef): ChapterDef {
  return {
    ...chapter,
    lead: 1,
    steps: [lead.step, ...chapter.steps],
    captions: [lead.caption, ...chapter.captions],
    exhibits: [0, ...chapter.exhibits.map((e) => (e === null ? null : e + 1))],
    overlays: [null, ...chapter.overlays],
    instructions: [lead.instruction, ...chapter.instructions],
    beats: [lead.beats, ...chapter.beats],
    buildModel: (kit, fonts) => joinLead(lead.buildModel(kit, fonts), chapter.buildModel(kit, fonts)),
  };
}

export function chapterStops(chapter: ChapterDef, text: ChapterText[]): Stop[] {
  const kicker = `chapter ${chapter.number} · ${chapter.title}`;
  const steps = text.map((t, i): Stop => {
    const page: PageText = { kicker, title: t.title, note: i === 0 ? chapter.formula : undefined, paragraphs: t.paragraphs, folio: "" };
    return {
      exhibit: chapter.exhibits[i] ?? null,
      spread: i,
      page,
      figure: `Fig. ${chapter.number}.${i + 1 - (chapter.lead ?? 0)}`,
      caption: chapter.captions[i] ?? "",
      instruction: chapter.instructions[i],
      folio: "",
      overlay: chapter.overlays[i] ?? null,
      beats: chapter.beats[i] ?? [overview(t.title)],
    };
  });
  const bench = chapter.workbench.map((w, k): Stop => {
    const i = text.length + k;
    return {
      exhibit: null,
      spread: i,
      page: { kicker: `${kicker} · try it yourself`, title: w.title, paragraphs: plainRuns(w.paragraphs), folio: "" },
      figure: "",
      caption: "",
      folio: "",
      overlay: w.overlay,
      beats: chapter.beats[i] ?? [flat(w.caption)],
    };
  });
  return [...steps, ...bench];
}
