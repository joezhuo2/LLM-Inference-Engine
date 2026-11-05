import type { BookPalette } from "./ink";

export type Run = { text: string; accent: boolean };
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

type Word = { text: string; accent: boolean; width: number; glue: boolean };

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
      words.push({ text: part, accent: run.accent, width: ctx.measureText(part).width, glue });
      glue = true;
    }
  }
  return words;
}

function drawParagraph(ctx: CanvasRenderingContext2D, runs: Run[], x: number, y: number, width: number, lineHeight: number, ink: string, accent: string, dry: boolean): number {
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
    }
    cx += word.width;
  }
  return cy + lineHeight;
}

export function drawTextPage(canvas: HTMLCanvasElement, page: PageText | null, palette: BookPalette, fonts: Fonts): void {
  const ctx = canvas.getContext("2d")!;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!page) return;
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
  for (const runs of page.paragraphs) cy = drawParagraph(ctx, runs, marginX, cy, textW, size * 1.46 * unit, palette.ink, palette.blue, false) + size * 0.7 * unit;
  ctx.fillStyle = palette.faint;
  ctx.font = `${26 * unit}px ${fonts.sans}`;
  ctx.fillText(page.folio, marginX, h - 90 * unit);
}

export function drawCaptionPage(canvas: HTMLCanvasElement, figure: string, caption: string, instruction: string | undefined, folio: string, palette: BookPalette, fonts: Fonts): void {
  const ctx = canvas.getContext("2d")!;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const unit = w / 1200;
  const x = 170 * unit;
  const width = w - 260 * unit;
  let y = h - 300 * unit;
  ctx.textBaseline = "alphabetic";
  ctx.font = `italic ${28 * unit}px ${fonts.sans}`;
  y = drawParagraph(ctx, [{ text: `${figure}. ${caption}`, accent: false }], x, y, width, 38 * unit, palette.faint, palette.faint, false) + 8 * unit;
  if (instruction) {
    ctx.font = `${40 * unit}px ${fonts.serif}`;
    drawParagraph(ctx, [{ text: instruction, accent: false }], x, y + 12 * unit, width, 48 * unit, palette.blue, palette.blue, false);
  }
  ctx.fillStyle = palette.faint;
  ctx.font = `${26 * unit}px ${fonts.sans}`;
  ctx.textAlign = "right";
  ctx.fillText(folio, w - x, h - 90 * unit);
  ctx.textAlign = "left";
}
