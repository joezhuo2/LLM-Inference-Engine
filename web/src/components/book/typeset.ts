import { splitTerms } from "@/lib/glossary";
import type { BookPalette } from "./ink";

export type Run = { text: string; accent: boolean; term?: string };
export type TermBox = { box: [number, number, number, number]; key: string };
type Marks = { boxes: TermBox[]; w: number; h: number; color: string };
export type PageText = { kicker: string; title: string; note?: string; paragraphs: Run[][]; folio: string };
export type Fonts = { serif: string; sans: string };

export function readFonts(): Fonts {
  const s = getComputedStyle(document.documentElement);
  const serif = s.getPropertyValue("--font-instrument-serif").trim() || "Georgia, serif";
  const inter = s.getPropertyValue("--font-inter").trim();
  return { serif, sans: `"Segoe UI", ${inter ? `${inter}, ` : ""}system-ui, sans-serif` };
}

export async function loadFonts(fonts: Fonts): Promise<void> {
  await Promise.allSettled([document.fonts.load(`88px ${fonts.serif}`), document.fonts.load(`34px ${fonts.sans}`)]);
}

export function readRuns(element: Element): Run[][] {
  return [...element.querySelectorAll("p")].map((p) => {
    const runs: Run[] = [];
    p.childNodes.forEach((node) => {
      const text = (node.textContent ?? "").replace(/\s+/g, " ");
      if (!text) return;
      const accent = node instanceof Element && node.classList.contains("text-accent");
      runs.push({ text, accent });
    });
    return runs;
  });
}

export function glossRuns(paragraphs: Run[][], seen: Set<string>): Run[][] {
  return paragraphs.map((runs) => runs.flatMap((run) => splitTerms(run.text, seen).map((p) => ({ text: p.text, accent: run.accent, term: p.term }))));
}

type Word = { text: string; accent: boolean; width: number; glue: boolean; term?: string };

function layoutWords(ctx: CanvasRenderingContext2D, runs: Run[]): Word[] {
  const words: Word[] = [];
  let glue = false;
  for (const run of runs) {
    const parts = run.text.split(/(\s+)/);
    for (const part of parts) {
      if (!part) continue;
      if (/^\s+$/.test(part)) {
        glue = false;
        continue;
      }
      words.push({ text: part, accent: run.accent, width: ctx.measureText(part).width, glue, term: run.term });
      glue = true;
    }
  }
  return words;
}

function drawParagraph(ctx: CanvasRenderingContext2D, runs: Run[], x: number, y: number, width: number, lineHeight: number, ink: string, accent: string, dry: boolean, marks?: Marks): number {
  const words = layoutWords(ctx, runs);
  const space = ctx.measureText(" ").width;
  let cx = x;
  let cy = y;
  for (const word of words) {
    const gap = word.glue || cx === x ? 0 : space;
    if (cx + gap + word.width > x + width && cx > x) {
      cx = x;
      cy += lineHeight;
    } else {
      cx += gap;
    }
    if (!dry) {
      ctx.fillStyle = word.accent ? accent : ink;
      ctx.fillText(word.text, cx, cy);
      if (word.term && marks) markWord(ctx, marks, word, cx, cy, lineHeight);
    }
    cx += word.width;
  }
  return cy + lineHeight;
}

function markWord(ctx: CanvasRenderingContext2D, marks: Marks, word: Word, x: number, y: number, lineHeight: number): void {
  const dot = Math.max(1.5, lineHeight * 0.05);
  const under = y + lineHeight * 0.16;
  ctx.save();
  ctx.fillStyle = marks.color;
  for (let dx = 0; dx < word.width; dx += dot * 2.4) ctx.fillRect(x + dx, under, dot, dot);
  ctx.restore();
  const top = y - lineHeight * 0.62;
  marks.boxes.push({ key: word.term!, box: [x / marks.w, top / marks.h, (x + word.width) / marks.w, (under + dot * 3) / marks.h] });
}

export function drawTextPage(canvas: HTMLCanvasElement, page: PageText | null, palette: BookPalette, fonts: Fonts, seen?: Set<string>): TermBox[] {
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!page) return [];
  const marks: Marks = { boxes: [], w, h, color: palette.blue };
  const paragraphs = seen ? glossRuns(page.paragraphs, seen) : page.paragraphs;
  const unit = w / 1200;
  const marginX = 130 * unit;
  const marginR = 170 * unit;
  const textW = w - marginX - marginR;
  ctx.textBaseline = "alphabetic";
  ctx.fillStyle = palette.faint;
  ctx.font = `${26 * unit}px ${fonts.sans}`;
  ctx.fillText(page.kicker, marginX, 150 * unit);
  ctx.fillStyle = palette.ink;
  ctx.font = `${92 * unit}px ${fonts.serif}`;
  ctx.font = `${Math.min(92, (92 * (w - marginX - 60 * unit)) / ctx.measureText(page.title).width) * unit}px ${fonts.serif}`;
  ctx.fillText(page.title, marginX, 260 * unit);
  let y = 325 * unit;
  if (page.note) {
    ctx.fillStyle = palette.blue;
    ctx.font = `${30 * unit}px ${fonts.sans}`;
    y = drawParagraph(ctx, [{ text: page.note, accent: false }], marginX, y, textW, 42 * unit, palette.blue, palette.blue, false) + 20 * unit;
  }
  let size = 26;
  const bottom = h - 170 * unit;
  for (; size > 24; size -= 1) {
    ctx.font = `${size * unit}px ${fonts.sans}`;
    let probe = y + size * unit;
    for (const runs of page.paragraphs) probe = drawParagraph(ctx, runs, marginX, probe, textW, size * 1.46 * unit, palette.ink, palette.blue, true) + size * 0.7 * unit;
    if (probe < bottom) break;
  }
  ctx.font = `${size * unit}px ${fonts.sans}`;
  let cy = y + size * unit;
  for (const runs of paragraphs) cy = drawParagraph(ctx, runs, marginX, cy, textW, size * 1.46 * unit, palette.ink, palette.blue, false, marks) + size * 0.7 * unit;
  ctx.fillStyle = palette.faint;
  ctx.font = `${26 * unit}px ${fonts.sans}`;
  ctx.fillText(page.folio, marginX, h - 90 * unit);
  return marks.boxes;
}

export function drawCaptionPage(canvas: HTMLCanvasElement, figure: string, caption: string, instruction: string | undefined, folio: string, palette: BookPalette, fonts: Fonts, seen?: Set<string>): TermBox[] {
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const unit = w / 1200;
  const x = 170 * unit;
  const width = w - 260 * unit;
  let y = h - 300 * unit;
  ctx.textBaseline = "alphabetic";
  ctx.font = `italic ${28 * unit}px ${fonts.sans}`;
  const marks: Marks = { boxes: [], w, h, color: palette.blue };
  const runs: Run[] = [{ text: `${figure}. `, accent: false }, ...(seen ? glossRuns([[{ text: caption, accent: false }]], seen)[0] : [{ text: caption, accent: false }])];
  y = drawParagraph(ctx, runs, x, y, width, 38 * unit, palette.faint, palette.faint, false, marks) + 8 * unit;
  if (instruction) {
    ctx.font = `${40 * unit}px ${fonts.serif}`;
    drawParagraph(ctx, [{ text: instruction, accent: false }], x, y + 12 * unit, width, 48 * unit, palette.blue, palette.blue, false);
  }
  ctx.fillStyle = palette.faint;
  ctx.font = `${26 * unit}px ${fonts.sans}`;
  ctx.textAlign = "right";
  ctx.fillText(folio, w - x, h - 90 * unit);
  ctx.textAlign = "left";
  return marks.boxes;
}
