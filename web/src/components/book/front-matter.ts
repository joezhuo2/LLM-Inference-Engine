import { type DeadEnd, deadEnds, kindLabel } from "@/content/dead-ends";
import { kernelStory } from "@/content/kernels";
import { PAGE_D, PAGE_W, type PageLink, type Stop } from "./book-scene";
import { type ChapterDef, chapterTitles, frames, stepNumber } from "./chapter";
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
  for (const key of ["contents", ...bookOrder.filter((slug) => spreads[slug] !== undefined), "errata"]) {
    firstPage[key] = page;
    page += key === "contents" ? 1 : key === "errata" ? errataSpreads.length : spreads[key];
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
const toBox = (x0: number, x1: number, y: number): [number, number, number, number] => [x0 / W, (y - 62) / H, x1 / W, (y + 30) / H];

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
    ctx.fillStyle = palette.faint;
    ctx.fillText(` · ${d.branch} · ${d.prs.map((n) => `#${n}`).join(" ")}`, kx + ctx.measureText(d.kernel).width, y);
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

function introRows(def: ChapterDef) {
  const titles = chapterTitles(def);
  const steps = def.steps.length;
  const step = Math.min(ROW, (H - 700) / (titles.length + (titles.length > steps ? 1.5 : 0)));
  const rows: { y: number; text: string; number?: string; bench?: boolean; k?: number }[] = [];
  let y = 380;
  titles.forEach((text, k) => {
    if (k === steps) {
      rows.push({ y: y + step * 0.2, text: "try it yourself" });
      y += step * 1.5;
    }
    rows.push({ y, text, number: stepNumber(def, k) || undefined, bench: k >= steps, k });
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
  const links: PageLink[] = rows
    .filter((r) => r.k !== undefined)
    .map((r) => ({ side: "right", box: [170 / W, (r.y - step * 0.65) / H, (W - 130) / W, (r.y + step * 0.32) / H], href: `${bookHref(def.slug)}#${r.k}`, label: r.number ? `step ${r.number}, ${r.text}` : r.bench ? `try it yourself, ${r.text}` : r.text }));
  return {
    ...flatStop(0, `Chapter ${def.number}: ${title}`, title, [summary]),
    paint: { left, right },
    links,
  };
}
