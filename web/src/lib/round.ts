export const sig = (v: number, figures = 2) => (v === 0 || !Number.isFinite(v) ? v : Number(v.toPrecision(figures)));

export const num = (v: number, figures = 2) => sig(v, figures).toLocaleString("en-US", { maximumFractionDigits: 20 });

export const about = (v: number, figures = 2, prefix = "about ") => `${sig(v, figures) === v ? "" : prefix}${num(v, figures)}`;

type Ladder = [number, string][];

export const byteUnits: Ladder = [
  [1, "bytes"],
  [1e3, "KB"],
  [1e6, "MB"],
  [1e9, "GB"],
  [1e12, "TB"],
];

export const timeUnits: Ladder = [
  [1e-6, "ns"],
  [1e-3, "µs"],
  [1, "ms"],
  [1e3, "s"],
];

export function scaled(v: number, ladder: Ladder, figures = 2, prefix = "about ") {
  const r = sig(v, figures);
  const [factor, unit] = [...ladder].reverse().find(([f]) => Math.abs(r) >= f) ?? ladder[0];
  return `${r === v ? "" : prefix}${num(v / factor, figures)} ${unit}`;
}

export const bytesText = (bytes: number, figures = 2, prefix = "about ") => scaled(bytes, byteUnits, figures, prefix);

export const msText = (ms: number, figures = 2, prefix = "about ") => scaled(ms, timeUnits, figures, prefix);
