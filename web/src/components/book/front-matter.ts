import { type DeadEnd, deadEnds, kindLabel } from "@/content/dead-ends";
import { compareCommand, type Exercise, exerciseCommands, chapterExercises, cloneCommand, implLabel, numpyCommand, numpyOf, numpyUrl, runsLabel } from "@/content/exercises";
import { kernelStory, layerOps } from "@/content/kernels";
import { chapterReview } from "@/content/reviews";
import type { ChapterReview } from "@/content/reviews/types";
import { PAGE_D, PAGE_W, type PageLink, type Stop } from "./book-scene";
import { type ChapterDef, chapterTitles, flat, frames, stepNumber } from "./chapter";
import type { BookPalette } from "./ink";
import { bookHref } from "./paths";
import type { Fonts } from "./typeset";

export const bookParts = [
  { numeral: "I", title: "the machine", slugs: ["foundations"] },
  { numeral: "II", title: "one layer", slugs: ["gemm", "rope", "paged-kv", "paged-attention", "attention-v2", "split-attention", "add-rmsnorm", "swiglu"] },
  { numeral: "III", title: "one step", slugs: ["sampling", "cuda-graphs"] },
  { numeral: "IV", title: "many requests", slugs: ["chunked-prefill"] },
];

export const bookOrder = bookParts.flatMap((p) => p.slugs);

export const chapterTitle = (slug: string) => kernelStory(slug)?.title ?? slug;
const firstPage: Record<string, number> = {};
export const chapterPage = (key: string): number | undefined => firstPage[key];

export function numberPages(spreads: Record<string, number>): number {
  let page = 2;
  for (const key of ["start", "contents", ...bookOrder.filter((slug) => spreads[slug] !== undefined), "errata"]) {
    firstPage[key] = page;
    page += key === "start" ? startSpreads : key === "contents" ? 1 : key === "errata" ? errataSpreads.length : spreads[key];
  }
  return page - 1;
}

export const errataTitle = "what didn't work";
export const errataIntro = `${deadEnds.length} changes that did not do what they were meant to, from the profiling notes, the changelog and the PRs. Each one says what was tried, what the measurement or the test showed, and what was kept. Numbers are as measured at the time.`;

const W = 1200;
const H = (W * PAGE_D) / PAGE_W;
type Side = "left" | "right";
type Ctx = CanvasRenderingContext2D;

function begin(canvas: HTMLCanvasElement, side: Side) {
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const unit = canvas.width / W;
  ctx.setTransform(unit, 0, 0, unit, 0, 0);
  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "left";
  const x0 = side === "left" ? 130 : 170;
  const x1 = side === "left" ? W - 170 : W - 130;
  return { ctx, x0, x1 };
}

function folio(ctx: Ctx, text: string, side: Side, x0: number, x1: number, palette: BookPalette, fonts: Fonts) {
  ctx.fillStyle = palette.faint;
  ctx.font = `26px ${fonts.sans}`;
  ctx.textAlign = side === "left" ? "left" : "right";
  ctx.fillText(text, side === "left" ? x0 : x1, H - 90);
  ctx.textAlign = "left";
}

function lines(ctx: Ctx, text: string, width: number): string[] {
  const out: string[] = [];
  let line = "";
  for (const word of text.split(/\s+/)) {
    const next = line ? `${line} ${word}` : word;
    if (line && ctx.measureText(next).width > width) {
      out.push(line);
      line = word;
    } else line = next;
  }
  if (line) out.push(line);
  return out;
}

function paragraph(ctx: Ctx, text: string, x: number, y: number, width: number, lineHeight: number, draw = true): number {
  for (const line of lines(ctx, text, width)) {
    if (draw) ctx.fillText(line, x, y);
    y += lineHeight;
  }
  return y;
}

const ROW = 92;
const toBox = (x0: number, x1: number, y: number): [number, number, number, number] => [x0 / W, (y - 56) / H, x1 / W, (y + 36) / H];

function contentsRows(parts: typeof bookParts, withErrata: boolean) {
  const rows: { kind: "part" | "entry"; y: number; text: string; number?: string; page?: number; href?: string }[] = [];
  let y = 380;
  for (const part of parts) {
    rows.push({ kind: "part", y, text: `Part ${part.numeral} · ${part.title}` });
    y += 78;
    for (const slug of part.slugs) {
      rows.push({ kind: "entry", y, text: chapterTitle(slug), number: String(bookOrder.indexOf(slug) + 1), page: chapterPage(slug), href: bookHref(slug) });
      y += ROW;
    }
    y += 40;
  }
  if (withErrata) rows.push({ kind: "entry", y: y + 20, text: `Errata: ${errataTitle}`, page: chapterPage("errata"), href: "#errata" });
  return rows;
}

function contentsPage(side: Side, parts: typeof bookParts, withErrata: boolean) {
  const paint = (canvas: HTMLCanvasElement, palette: BookPalette, fonts: Fonts) => {
    const { ctx, x0, x1 } = begin(canvas, side);
    if (side === "left") {
      ctx.fillStyle = palette.faint;
      ctx.font = `26px ${fonts.sans}`;
      ctx.fillText("Every Byte", x0, 150);
      ctx.fillStyle = palette.ink;
      ctx.font = `92px ${fonts.serif}`;
      ctx.fillText("Contents", x0, 260);
    }
    for (const row of contentsRows(parts, withErrata)) {
      if (row.kind === "part") {
        ctx.fillStyle = palette.faint;
        ctx.font = `26px ${fonts.sans}`;
        ctx.fillText(row.text, x0, row.y);
        continue;
      }
      ctx.fillStyle = palette.faint;
      ctx.font = `40px ${fonts.serif}`;
      if (row.number) ctx.fillText(row.number, x0, row.y);
      ctx.fillStyle = palette.ink;
      ctx.font = `46px ${fonts.serif}`;
      const tx = x0 + 70;
      ctx.fillText(row.text, tx, row.y);
      const end = tx + ctx.measureText(row.text).width + 18;
      if (row.page === undefined) continue;
      ctx.font = `30px ${fonts.sans}`;
      const page = String(row.page);
      const pw = ctx.measureText(page).width;
      ctx.fillStyle = palette.faint;
      ctx.fillText(page, x1 - pw, row.y);
      for (let x = end; x < x1 - pw - 22; x += 16) ctx.fillRect(x, row.y - 4, 3, 3);
    }
    if (side === "right") {
      ctx.fillStyle = palette.blue;
      ctx.font = `40px ${fonts.serif}`;
      paragraph(ctx, "Click a chapter to open it.", x0, H - 250, x1 - x0, 48);
    }
    if (side === "left") folio(ctx, String(chapterPage("contents")), side, x0, x1, palette, fonts);
  };
  const x0 = side === "left" ? 130 : 170;
  const x1 = side === "left" ? W - 170 : W - 130;
  const links: PageLink[] = contentsRows(parts, withErrata)
    .filter((r) => r.href)
    .map((r) => ({ side, box: toBox(x0, x1, r.y), href: r.href!, label: r.number ? `chapter ${r.number}, ${r.text}` : r.text }));
  return { paint, links };
}

function errataTitlePage(canvas: HTMLCanvasElement, palette: BookPalette, fonts: Fonts) {
  const { ctx, x0, x1 } = begin(canvas, "left");
  ctx.fillStyle = palette.faint;
  ctx.font = `26px ${fonts.sans}`;
  ctx.fillText("Every Byte · errata", x0, 150);
  ctx.fillStyle = palette.ink;
  ctx.font = `92px ${fonts.serif}`;
  ctx.fillText("Errata", x0, 260);
  ctx.fillStyle = palette.faint;
  ctx.font = `44px ${fonts.serif}`;
  ctx.fillText(errataTitle, x0, 330);
  ctx.fillStyle = palette.ink;
  ctx.font = `36px ${fonts.sans}`;
  paragraph(ctx, errataIntro, x0, 450, x1 - x0, 54);
  folio(ctx, String(chapterPage("errata")), "left", x0, x1, palette, fonts);
}

function deadEndPage(d: DeadEnd, side: Side, offset: number) {
  const link: PageLink = { side, box: [0, 0, 0, 0], href: bookOrder.includes(d.slug) ? bookHref(d.slug) : `/kernels/${d.slug}`, label: d.kernel };
  const paint = (canvas: HTMLCanvasElement, palette: BookPalette, fonts: Fonts) => {
    const { ctx, x0, x1 } = begin(canvas, side);
    const width = x1 - x0;
    ctx.fillStyle = palette.faint;
    ctx.font = `26px ${fonts.sans}`;
    ctx.fillText(`errata · ${kindLabel[d.kind]}`, x0, 150);
    ctx.fillStyle = palette.ink;
    ctx.font = `68px ${fonts.serif}`;
    let y = paragraph(ctx, d.title, x0, 240, width, 74) + 6;
    ctx.font = `28px ${fonts.sans}`;
    const lead = `${d.date} · `;
    ctx.fillStyle = palette.faint;
    ctx.fillText(lead, x0, y);
    const kx = x0 + ctx.measureText(lead).width;
    ctx.fillStyle = palette.blue;
    ctx.fillText(d.kernel, kx, y);
    link.box = [(kx - 8) / W, (y - 40) / H, (kx + ctx.measureText(d.kernel).width + 8) / W, (y + 14) / H];
    y += 40;
    const parts: [string, string][] = [
      ["tried", d.tried],
      ["happened", d.happened],
      ["kept", d.kept],
    ];
    const bottom = H - 170;
    let size = 36;
    const layout = (draw: boolean) => {
      let cy = y + 30;
      for (const [label, text] of parts) {
        ctx.fillStyle = palette.faint;
        ctx.font = `24px ${fonts.sans}`;
        cy += 34;
        if (draw) ctx.fillText(label, x0, cy);
        cy += size * 1.3;
        ctx.fillStyle = palette.ink;
        ctx.font = `${size}px ${fonts.sans}`;
        cy = paragraph(ctx, text, x0, cy, width, size * 1.42, draw) + size * 0.3;
      }
      return cy;
    };
    while (size > 24 && layout(false) > bottom) size -= 1;
    layout(true);
    if (side === "left") folio(ctx, String((chapterPage("errata") ?? 0) + offset), side, x0, x1, palette, fonts);
  };
  return { paint, link };
}

const pageText = (title: string, paragraphs: string[]) => ({ kicker: "Every Byte", title, paragraphs: paragraphs.map((text) => [{ text, accent: false }]), folio: "" });

const flatStop = (spread: number, caption: string, title: string, paragraphs: string[]): Stop => ({
  exhibit: null,
  spread,
  page: pageText(title, paragraphs),
  figure: "",
  caption: "",
  folio: "",
  overlay: null,
  beats: [{ caption, frame: frames.spread }],
});

export function contentsStop(spread = 0): Stop {
  const left = contentsPage("left", bookParts.slice(0, 2), false);
  const right = contentsPage("right", bookParts.slice(2), true);
  return {
    ...flatStop(spread, "Contents: click a chapter to open it", "Contents", bookOrder.map(chapterTitle)),
    paint: { left: left.paint, right: right.paint },
    links: [...left.links, ...right.links],
  };
}

export type ErrataSpread = { left: DeadEnd | null; right: DeadEnd | null };

export const errataSpreads: ErrataSpread[] = Array.from({ length: Math.ceil((deadEnds.length + 1) / 2) }, (_, k) => ({
  left: k === 0 ? null : deadEnds[2 * k - 1],
  right: deadEnds[2 * k] ?? null,
}));

export function errataStops(firstSpread: number): Stop[] {
  return errataSpreads.map(({ left, right }, k) => {
    const l = left ? deadEndPage(left, "left", k) : null;
    const r = right ? deadEndPage(right, "right", k) : null;
    const shown = [left, right].filter((d): d is DeadEnd => d !== null);
    return {
      ...flatStop(firstSpread + k, k === 0 ? `Errata: ${errataTitle}` : `Errata: ${shown.map((d) => d.kernel).join(" and ")}`, left?.title ?? "Errata", left ? [left.tried, left.happened, left.kept] : [errataIntro]),
      paint: { left: l?.paint ?? errataTitlePage, right: r?.paint ?? ((canvas) => void begin(canvas, "right")) },
      links: [l?.link, r?.link].filter((x): x is PageLink => !!x),
    };
  });
}

export const reviewTitles = ["what to remember", "answers"];

export const reviewCount = (slug: string) => (chapterReview(slug) ? reviewTitles.length : 0);

export const bookTitles = (def: ChapterDef) => [...chapterTitles(def), ...(chapterReview(def.slug) ? reviewTitles : []), ...chapterExercises(def.slug).map((e) => e.title)];

function introRows(def: ChapterDef) {
  const titles = bookTitles(def);
  const steps = def.steps.length;
  const reviewed = steps + def.workbench.length;
  const built = reviewed + reviewCount(def.slug);
  const starts: Record<number, string> = {};
  if (def.workbench.length > 0) starts[steps] = "try it yourself";
  if (built > reviewed) starts[reviewed] = "review";
  if (titles.length > built) starts[built] = "build it yourself";
  const step = Math.min(ROW, (H - 700) / (titles.length + Object.keys(starts).length * 1.5));
  const rows: { y: number; text: string; number?: string; kind?: "bench" | "review" | "build"; k?: number }[] = [];
  let y = 380;
  titles.forEach((text, k) => {
    if (starts[k]) {
      rows.push({ y: y + step * 0.2, text: starts[k] });
      y += step * 1.5;
    }
    rows.push({ y, text, number: stepNumber(def, k) || undefined, kind: k >= built ? "build" : k >= reviewed ? "review" : k >= steps ? "bench" : undefined, k });
    y += step;
  });
  return { rows, step };
}

export function introStop(def: ChapterDef): Stop {
  const part = bookParts.find((p) => p.slugs.includes(def.slug));
  const title = chapterTitle(def.slug);
  const summary = kernelStory(def.slug)?.short ?? "";
  const { rows, step } = introRows(def);
  const page = (k: number) => (chapterPage(def.slug) ?? 0) + k;
  const left = (canvas: HTMLCanvasElement, palette: BookPalette, fonts: Fonts) => {
    const { ctx, x0, x1 } = begin(canvas, "left");
    ctx.fillStyle = palette.faint;
    ctx.font = `26px ${fonts.sans}`;
    ctx.fillText(part ? `Part ${part.numeral} · ${part.title}` : "Every Byte", x0, 150);
    ctx.font = `44px ${fonts.serif}`;
    ctx.fillText(`Chapter ${def.number}`, x0, 420);
    ctx.fillStyle = palette.ink;
    ctx.font = `92px ${fonts.serif}`;
    const y = paragraph(ctx, title, x0, 530, x1 - x0, 100);
    ctx.font = `36px ${fonts.sans}`;
    paragraph(ctx, summary, x0, y + 50, x1 - x0, 54);
    folio(ctx, String(page(0)), "left", x0, x1, palette, fonts);
  };
  const right = (canvas: HTMLCanvasElement, palette: BookPalette, fonts: Fonts) => {
    const { ctx, x0, x1 } = begin(canvas, "right");
    ctx.fillStyle = palette.faint;
    ctx.font = `26px ${fonts.sans}`;
    ctx.fillText(`chapter ${def.number} · ${title}`, x0, 150);
    ctx.fillStyle = palette.ink;
    ctx.font = `92px ${fonts.serif}`;
    ctx.fillText("in this chapter", x0, 260);
    for (const row of rows) {
      if (row.k === undefined) {
        ctx.fillStyle = palette.faint;
        ctx.font = `26px ${fonts.sans}`;
        ctx.fillText(row.text, x0, row.y);
        continue;
      }
      ctx.fillStyle = palette.faint;
      ctx.font = `${Math.round(step * 0.43)}px ${fonts.serif}`;
      if (row.number) ctx.fillText(row.number, x0, row.y);
      const target = String(page(row.k + 1));
      ctx.font = `30px ${fonts.sans}`;
      const pw = ctx.measureText(target).width;
      ctx.fillText(target, x1 - pw, row.y);
      const tx = x0 + 70;
      let size = Math.round(step * 0.5);
      ctx.font = `${size}px ${fonts.serif}`;
      while (size > 26 && ctx.measureText(row.text).width > x1 - pw - 40 - tx) ctx.font = `${--size}px ${fonts.serif}`;
      ctx.fillStyle = palette.ink;
      ctx.fillText(row.text, tx, row.y);
      ctx.fillStyle = palette.faint;
      for (let x = tx + ctx.measureText(row.text).width + 18; x < x1 - pw - 22; x += 16) ctx.fillRect(x, row.y - 4, 3, 3);
    }
    ctx.fillStyle = palette.blue;
    ctx.font = `40px ${fonts.serif}`;
    paragraph(ctx, "Click a step to turn to it.", x0, H - 250, x1 - x0, 48);
  };
  const entries = rows.filter((r) => r.k !== undefined);
  const links: PageLink[] = entries.map((r, i) => ({
      side: "right",
      box: [170 / W, (r.y - step * 0.63) / H, (W - 130) / W, (r.y + step * (i === entries.length - 1 ? 0.75 : 0.37)) / H],
      href: `${bookHref(def.slug)}#${r.k}`, label: r.number ? `step ${r.number}, ${r.text}` : r.kind === "bench" ? `try it yourself, ${r.text}` : r.kind === "build" ? `build it yourself, ${r.text}` : r.kind === "review" ? `review, ${r.text}` : r.text }));
  return {
    ...flatStop(0, `Chapter ${def.number}: ${title}`, title, [summary]),
    paint: { left, right },
    links,
  };
}

const MONO_INDENT = 28;

function monoLines(ctx: Ctx, command: string, width: number): string[] {
  const out: string[] = [];
  let line = "";
  for (const word of command.split(" ")) {
    const next = line ? `${line} ${word}` : word;
    if (line && ctx.measureText(next).width > width - (out.length ? MONO_INDENT : 0)) {
      out.push(line);
      line = word;
    } else line = next;
  }
  if (line) out.push(line);
  return out;
}

type ExerciseStep = { title: string; text?: string; commands?: string[] };

function exerciseSteps(e: Exercise): ExerciseStep[] {
  if (e.kind === "open") {
    return [
      { title: "Write it from scratch", text: "A single .cu file built with nvcc is enough; nothing in the engine changes.", commands: ["nvcc -O3 -arch=native copy.cu -o copy && ./copy"] },
      { title: "Check it", text: e.check },
      { title: "Compare", text: e.compare.map((c) => `${c.path}: ${c.label}.`).join(" ") },
    ];
  }
  const [checkout, build, test] = exerciseCommands(e);
  return [
    { title: "Go back to the day the stub landed", text: `Commit ${e.start}: the function throws, the tests are written.`, commands: [cloneCommand, checkout, build] },
    { title: "Run the tests and watch them fail", commands: [test] },
    { title: "Make them pass", text: `Write it in ${e.file}, from the spec and the tests in ${e.tests}.` },
    { title: "Compare", text: `Then read ${implLabel(e)} in PR ${e.implPrs.join(", PR ")}, or diff it against the stub:`, commands: [compareCommand(e)] },
  ];
}

const NUMPY_Y = H - 260;

function exerciseLeft(def: ChapterDef, e: Exercise, folioPage: () => number) {
  return (canvas: HTMLCanvasElement, palette: BookPalette, fonts: Fonts) => {
    const { ctx, x0, x1 } = begin(canvas, "left");
    const width = x1 - x0;
    ctx.fillStyle = palette.faint;
    ctx.font = `26px ${fonts.sans}`;
    ctx.fillText(`chapter ${def.number} · ${chapterTitle(def.slug)} · build it yourself`, x0, 150);
    ctx.fillStyle = palette.ink;
    ctx.font = `92px ${fonts.serif}`;
    ctx.font = `${Math.min(92, (92 * width) / ctx.measureText(e.title).width)}px ${fonts.serif}`;
    ctx.fillText(e.title, x0, 260);
    ctx.fillStyle = palette.faint;
    ctx.font = `28px ${fonts.sans}`;
    const tags = [runsLabel[e.runs], e.kind === "stub" && e.model ? "some tests load TinyLlama" : e.kind === "open" ? "no tests, open ended" : ""].filter(Boolean);
    ctx.fillText(tags.join(" · "), x0, 320);
    ctx.fillStyle = palette.ink;
    ctx.font = `36px ${fonts.sans}`;
    let y = paragraph(ctx, e.task, x0, 410, width, 54) + 40;
    ctx.fillStyle = palette.faint;
    ctx.font = `24px ${fonts.sans}`;
    ctx.fillText("a nudge, if you are stuck", x0, y);
    ctx.fillStyle = palette.ink;
    ctx.font = `32px ${fonts.sans}`;
    y = paragraph(ctx, e.nudge, x0, y + 48, width, 46);
    const numpy = numpyOf(e);
    if (numpy) {
      ctx.fillStyle = palette.faint;
      ctx.font = `24px ${fonts.sans}`;
      ctx.fillText("try it in NumPy first, on any machine", x0, NUMPY_Y);
      ctx.fillStyle = palette.blue;
      ctx.font = `25px ${fonts.mono}`;
      ctx.fillText(numpyCommand(numpy), x0, NUMPY_Y + 48);
    }
    folio(ctx, String(folioPage()), "left", x0, x1, palette, fonts);
  };
}

const LINK_TEXT = "The full spec, with commands to copy →";

function exerciseRight(e: Exercise) {
  return (canvas: HTMLCanvasElement, palette: BookPalette, fonts: Fonts) => {
    const { ctx, x0, x1 } = begin(canvas, "right");
    const width = x1 - x0;
    ctx.fillStyle = palette.faint;
    ctx.font = `26px ${fonts.sans}`;
    ctx.fillText(e.kind === "stub" ? "four steps" : "three steps", x0, 150);
    const steps = exerciseSteps(e);
    const bottom = H - 300;
    const layout = (scale: number, draw: boolean) => {
      let y = 230;
      steps.forEach((st, i) => {
        const head = 40 * scale;
        ctx.font = `${head}px ${fonts.serif}`;
        if (draw) {
          ctx.fillStyle = palette.faint;
          ctx.fillText(String(i + 1), x0, y);
          ctx.fillStyle = palette.ink;
          ctx.fillText(st.title, x0 + 50, y);
        }
        y += head * 0.5;
        if (st.text) {
          ctx.font = `${28 * scale}px ${fonts.sans}`;
          ctx.fillStyle = palette.ink;
          y = paragraph(ctx, st.text, x0 + 50, y + 34 * scale, width - 50, 40 * scale, draw) - 40 * scale;
        }
        for (const command of st.commands ?? []) {
          ctx.font = `${23 * scale}px ${fonts.mono}`;
          const lines = monoLines(ctx, command, width - 74);
          const top = y + 22 * scale;
          const lh = 34 * scale;
          if (draw) {
            ctx.fillStyle = palette.blue;
            ctx.fillRect(x0 + 50, top, 3, lines.length * lh + 6 * scale);
            ctx.fillStyle = palette.ink;
            lines.forEach((line, k) => ctx.fillText(line, x0 + 70 + (k ? MONO_INDENT : 0), top + lh * (k + 0.8)));
          }
          y = top + lines.length * lh;
        }
        y += 70 * scale;
      });
      return y;
    };
    let scale = 1;
    while (scale > 0.6 && layout(scale, false) > bottom) scale -= 0.02;
    layout(scale, true);
    ctx.fillStyle = palette.blue;
    ctx.font = `38px ${fonts.serif}`;
    ctx.fillText(LINK_TEXT, x0, H - 200);
  };
}

export function exerciseStops(def: ChapterDef): Stop[] {
  const first = def.steps.length + def.workbench.length + reviewCount(def.slug);
  return chapterExercises(def.slug).map((e, j) => {
    const local = first + j;
    const left = exerciseLeft(def, e, () => (chapterPage(def.slug) ?? 0) + local + 1);
    const numpy = numpyOf(e);
    const links: PageLink[] = [{ side: "right", box: [170 / W, (H - 250) / H, (W - 130) / W, (H - 180) / H], href: `/exercises#${e.id}`, label: `${e.title}: ${LINK_TEXT}` }];
    if (numpy) links.push({ side: "left", box: [130 / W, (NUMPY_Y + 48 - 40) / H, (W - 170) / W, (NUMPY_Y + 48 + 16) / H], href: numpyUrl(numpy), label: `${e.title}: the NumPy version on GitHub, ${numpyCommand(numpy)}` });
    return {
      exhibit: null,
      spread: local,
      page: { kicker: `chapter ${def.number} · build it yourself`, title: e.title, paragraphs: [[{ text: e.task, accent: false }]], folio: "" },
      figure: "",
      caption: "",
      folio: "",
      overlay: null,
      beats: [flat(`Build it yourself: ${e.title}`)],
      paint: { left, right: exerciseRight(e) },
      links,
    };
  });
}

const startSpreads = 2;

const forWho = [
  "This book follows one token through an inference engine written from scratch in C++ and CUDA, kernel by kernel. The model is TinyLlama 1.1B and the hardware is one laptop GPU, an RTX 5060, and every number in the book was measured on it.",
  "It is for you if you have used a language model through PyTorch or an API and want to know what happens underneath: why generating text is slow, where the time goes, and what engines like vLLM and llama.cpp do about it.",
  "It is not about training, about how transformers were designed, or about serving across many GPUs. It covers one small engine in depth and says when something is left out.",
];

const needFirst = [
  "Python and a little PyTorch: tensors, shapes and a matrix multiply.",
  "A transformer at the level of a diagram: tokens become vectors, attention mixes information between them, and an MLP transforms each one. Chapter 1 and the first spread of every chapter explain the rest as they go.",
  "Neither C++ nor a GPU is needed to read the book. Reading the code takes some C++, and the GPU exercises take an NVIDIA GPU with CUDA.",
  "If transformers are new to you, start with Andrej Karpathy's video Let's build GPT or Jay Alammar's The Illustrated Transformer, then come back.",
];

const tokenLife = [
  "A prompt arrives as text. The tokenizer cuts it into tokens, numbers from a vocabulary of 32,000, and each token becomes a vector of 2,048 values.",
  "Prefill runs the whole prompt through the model at once. Decode then writes the answer one token at a time: the newest token goes through all 22 layers, and its last vector becomes 32,000 scores, one for every possible next token. Sampling picks one and the loop starts again.",
  "Every layer does the same ten things, shown in the map of one decode step. Most of the time goes to reading weights and cached keys and values from memory, not to arithmetic. Chapter 1 shows why, and the rest of the book is about moving fewer bytes and wasting less time between kernels.",
  "Read it in order the first time, because the parts follow the token. Each chapter says what its piece is, walks through how it works, and ends with something to try and an exercise to build.",
];

function textPage(side: Side, kicker: string, title: string, paragraphs: string[], page: () => number | undefined, size = 34) {
  return (canvas: HTMLCanvasElement, palette: BookPalette, fonts: Fonts) => {
    const { ctx, x0, x1 } = begin(canvas, side);
    ctx.fillStyle = palette.faint;
    ctx.font = `26px ${fonts.sans}`;
    ctx.fillText(kicker, x0, 150);
    ctx.fillStyle = palette.ink;
    ctx.font = `92px ${fonts.serif}`;
    ctx.fillText(title, x0, 260);
    let px = size;
    const fit = (draw: boolean) => {
      ctx.font = `${px}px ${fonts.sans}`;
      let y = 360;
      for (const text of paragraphs) y = paragraph(ctx, text, x0, y, x1 - x0, px * 1.5, draw) + px * 0.8;
      return y;
    };
    while (px > 26 && fit(false) > H - 170) px -= 1;
    ctx.fillStyle = palette.ink;
    fit(true);
    if (side === "left") folio(ctx, String(page() ?? ""), side, x0, x1, palette, fonts);
  };
}

const chapterNumber = (slug: string) => bookOrder.indexOf(slug) + 1;

type MapRow = { y: number; label: string; slugs: string[]; inLayer: boolean };

function mapRows(): { rows: MapRow[]; layerTop: number; layerBottom: number } {
  const rows: MapRow[] = [];
  let y = 360;
  const add = (label: string, slugs: string[], inLayer = false, gap = 64) => {
    rows.push({ y, label, slugs, inLayer });
    y += gap;
  };
  add("token in, its embedding row read", [], false, 92);
  const layerTop = y - 50;
  for (const op of layerOps) add(op.label, op.slug === "paged-attention" ? ["paged-attention", "attention-v2", "split-attention"] : [op.slug], true);
  const layerBottom = y - 26;
  y += 46;
  add("final RMSNorm", ["add-rmsnorm"]);
  add("LM head: 32,000 scores", ["gemm"]);
  add("sampling: the next token", ["sampling"]);
  return { rows, layerTop, layerBottom };
}

function tokenMapPage(canvas: HTMLCanvasElement, palette: BookPalette, fonts: Fonts) {
  const { ctx, x0, x1 } = begin(canvas, "right");
  ctx.fillStyle = palette.faint;
  ctx.font = `26px ${fonts.sans}`;
  ctx.fillText("Every Byte · start here", x0, 150);
  ctx.fillStyle = palette.ink;
  ctx.font = `92px ${fonts.serif}`;
  ctx.fillText("one decode step", x0, 260);
  const { rows, layerTop, layerBottom } = mapRows();
  ctx.strokeStyle = palette.faint;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(x0, layerTop, x1 - x0, layerBottom - layerTop, 14);
  ctx.stroke();
  ctx.fillStyle = palette.faint;
  ctx.font = `26px ${fonts.sans}`;
  ctx.textAlign = "right";
  ctx.fillText("× 22 layers", x1 - 20, layerTop - 14);
  ctx.textAlign = "left";
  for (const row of rows) {
    const x = row.inLayer ? x0 + 30 : x0;
    ctx.fillStyle = palette.ink;
    ctx.font = `36px ${fonts.serif}`;
    ctx.fillText(row.label, x, row.y);
    if (row.slugs.length) {
      ctx.fillStyle = palette.blue;
      ctx.font = `28px ${fonts.sans}`;
      const text = `ch ${row.slugs.map(chapterNumber).join(", ")}`;
      ctx.textAlign = "right";
      ctx.fillText(text, x1 - (row.inLayer ? 30 : 0), row.y);
      ctx.textAlign = "left";
    }
  }
  ctx.fillStyle = palette.faint;
  ctx.font = `28px ${fonts.sans}`;
  paragraph(ctx, `Around the whole step: CUDA Graphs (ch ${chapterNumber("cuda-graphs")}) replay it with one launch, and the scheduler with chunked prefill (ch ${chapterNumber("chunked-prefill")}) decides which requests share it. Click a row to open its chapter.`, x0, H - 250, x1 - x0, 40);
}

function mapLinks(): PageLink[] {
  const { rows } = mapRows();
  return rows
    .filter((r) => r.slugs.length)
    .map((r) => ({ side: "right" as const, box: [170 / W, (r.y - 46) / H, (W - 130) / W, (r.y + 16) / H] as [number, number, number, number], href: bookHref(r.slugs[0]), label: `${r.label}, chapter ${chapterNumber(r.slugs[0])}` }));
}

export const startText = { forWho, needFirst, tokenLife, map: () => mapRows().rows };

export function startStops(): Stop[] {
  const page = () => chapterPage("start");
  const next = () => (chapterPage("start") ?? 0) + 1;
  return [
    {
      ...flatStop(0, "Start here: who this book is for", "start here", [...forWho, ...needFirst]),
      paint: { left: textPage("left", "Every Byte", "start here", forWho, page, 36), right: textPage("right", "Every Byte · start here", "what you need first", needFirst, page, 34) },
    },
    {
      ...flatStop(1, "Start here: the life of a token", "the life of a token", tokenLife),
      paint: { left: textPage("left", "Every Byte · start here", "the life of a token", tokenLife, next, 32), right: tokenMapPage },
      links: mapLinks(),
    },
  ];
}

function fitLayout(layout: (scale: number, draw: boolean) => number, bottom: number) {
  let scale = 1;
  while (scale > 0.6 && layout(scale, false) > bottom) scale -= 0.02;
  layout(scale, true);
}

function reviewLeft(def: ChapterDef, r: ChapterReview, page: () => number) {
  const story = kernelStory(def.slug);
  return (canvas: HTMLCanvasElement, palette: BookPalette, fonts: Fonts) => {
    const { ctx, x0, x1 } = begin(canvas, "left");
    const width = x1 - x0;
    ctx.fillStyle = palette.faint;
    ctx.font = `26px ${fonts.sans}`;
    ctx.fillText(`chapter ${def.number} · ${chapterTitle(def.slug)} · review`, x0, 150);
    ctx.fillStyle = palette.ink;
    ctx.font = `92px ${fonts.serif}`;
    ctx.fillText("what to remember", x0, 260);
    fitLayout((scale, draw) => {
      let y = 300;
      ctx.fillStyle = palette.blue;
      ctx.font = `${44 * scale}px ${fonts.serif}`;
      y = paragraph(ctx, r.lesson, x0, y + 30 * scale, width, 54 * scale, draw) + 40 * scale;
      ctx.fillStyle = palette.faint;
      ctx.font = `${24 * scale}px ${fonts.sans}`;
      if (draw) ctx.fillText("the number that matters", x0, y + 10);
      y += 40 * scale;
      if (story) {
        ctx.fillStyle = palette.blue;
        ctx.font = `${76 * scale}px ${fonts.serif}`;
        if (draw) ctx.fillText(story.headline.value, x0, y + 60 * scale);
        y += 100 * scale;
        ctx.fillStyle = palette.ink;
        ctx.font = `${28 * scale}px ${fonts.sans}`;
        y = paragraph(ctx, story.headline.label, x0, y, width, 40 * scale, draw) + 6 * scale;
      }
      ctx.fillStyle = palette.faint;
      ctx.font = `${28 * scale}px ${fonts.sans}`;
      y = paragraph(ctx, r.why, x0, y + 10 * scale, width, 40 * scale, draw) + 50 * scale;
      r.takeaways.forEach((t, i) => {
        ctx.fillStyle = palette.blue;
        ctx.font = `${40 * scale}px ${fonts.serif}`;
        if (draw) ctx.fillText(String(i + 1), x0, y + 30 * scale);
        ctx.fillStyle = palette.ink;
        ctx.font = `${31 * scale}px ${fonts.sans}`;
        y = paragraph(ctx, t, x0 + 50, y + 30 * scale, width - 50, 45 * scale, draw) + 24 * scale;
      });
      return y;
    }, H - 170);
    folio(ctx, String(page()), "left", x0, x1, palette, fonts);
  };
}

function reviewRight(def: ChapterDef, r: ChapterReview) {
  return (canvas: HTMLCanvasElement, palette: BookPalette, fonts: Fonts) => {
    const { ctx, x0, x1 } = begin(canvas, "right");
    const width = x1 - x0;
    ctx.fillStyle = palette.faint;
    ctx.font = `26px ${fonts.sans}`;
    ctx.fillText(`chapter ${def.number} · review`, x0, 150);
    ctx.fillStyle = palette.ink;
    ctx.font = `92px ${fonts.serif}`;
    ctx.fillText("check yourself", x0, 260);
    fitLayout((scale, draw) => {
      let y = 360;
      r.questions.forEach((q, i) => {
        ctx.fillStyle = palette.blue;
        ctx.font = `${44 * scale}px ${fonts.serif}`;
        if (draw) ctx.fillText(String(i + 1), x0, y);
        ctx.fillStyle = palette.ink;
        ctx.font = `${34 * scale}px ${fonts.sans}`;
        y = paragraph(ctx, q.question, x0 + 56, y, width - 56, 50 * scale, draw) + 60 * scale;
      });
      return y;
    }, H - 320);
    ctx.fillStyle = palette.blue;
    ctx.font = `40px ${fonts.serif}`;
    paragraph(ctx, "Decide your answers, then turn the page.", x0, H - 250, width, 48);
  };
}

function answerPage(def: ChapterDef, r: ChapterReview, k: number, side: Side, page: () => number) {
  return (canvas: HTMLCanvasElement, palette: BookPalette, fonts: Fonts) => {
    const { ctx, x0, x1 } = begin(canvas, side);
    const width = x1 - x0;
    const q = r.questions[k];
    ctx.fillStyle = palette.faint;
    ctx.font = `26px ${fonts.sans}`;
    ctx.fillText(`chapter ${def.number} · review · answers`, x0, 150);
    ctx.fillStyle = palette.ink;
    ctx.font = `92px ${fonts.serif}`;
    ctx.fillText(`answer ${k + 1}`, x0, 260);
    if (q) {
      fitLayout((scale, draw) => {
        ctx.fillStyle = palette.faint;
        ctx.font = `${30 * scale}px ${fonts.sans}`;
        let y = paragraph(ctx, q.question, x0, 350, width, 44 * scale, draw) + 50 * scale;
        ctx.fillStyle = palette.blue;
        ctx.font = `${50 * scale}px ${fonts.serif}`;
        y = paragraph(ctx, q.answer, x0, y + 20 * scale, width, 62 * scale, draw) + 30 * scale;
        ctx.fillStyle = palette.ink;
        ctx.font = `${32 * scale}px ${fonts.sans}`;
        return paragraph(ctx, q.why, x0, y, width, 48 * scale, draw);
      }, H - 170);
    }
    if (side === "left") folio(ctx, String(page()), side, x0, x1, palette, fonts);
  };
}

export function reviewStops(def: ChapterDef): Stop[] {
  const r = chapterReview(def.slug);
  if (!r) return [];
  const first = def.steps.length + def.workbench.length;
  const page = (local: number) => () => (chapterPage(def.slug) ?? 0) + local + 1;
  const base = (local: number, title: string, caption: string, paragraphs: string[]): Stop => ({
    exhibit: null,
    spread: local,
    page: { kicker: `chapter ${def.number} · review`, title, paragraphs: paragraphs.map((text) => [{ text, accent: false }]), folio: "" },
    figure: "",
    caption: "",
    folio: "",
    overlay: null,
    beats: [flat(caption)],
  });
  return [
    { ...base(first, reviewTitles[0], "Review: what to remember, then check yourself", [r.lesson, r.why, ...r.takeaways]), paint: { left: reviewLeft(def, r, page(first)), right: reviewRight(def, r) } },
    { ...base(first + 1, reviewTitles[1], "Review: the answers", r.questions.map((q) => `${q.answer} ${q.why}`)), paint: { left: answerPage(def, r, 0, "left", page(first + 1)), right: answerPage(def, r, 1, "right", page(first + 1)) } },
  ];
}
