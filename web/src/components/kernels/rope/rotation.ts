export const HEAD_DIM = 64;
export const PAIRS = HEAD_DIM / 2;
export const THETA = 10000;
export const MAX_POS = 2048;

export const invFreq = (pair: number) => 1 / Math.pow(THETA, (2 * pair) / HEAD_DIM);
export const wavelength = (pair: number) => (2 * Math.PI) / invFreq(pair);
export const angle = (pair: number, pos: number) => pos * invFreq(pair);

export function rotatePair(lo: number, hi: number, a: number): [number, number] {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [lo * c - hi * s, hi * c + lo * s];
}

export function rotate(vec: number[], pos: number): number[] {
  const out = new Array<number>(HEAD_DIM);
  for (let i = 0; i < PAIRS; i++) {
    const [lo, hi] = rotatePair(vec[i], vec[i + PAIRS], angle(i, pos));
    out[i] = lo;
    out[i + PAIRS] = hi;
  }
  return out;
}

export function score(q: number[], m: number, k: number[], n: number): number {
  const rq = rotate(q, m);
  const rk = rotate(k, n);
  return rq.reduce((sum, v, d) => sum + v * rk[d], 0);
}

export function seededVector(seed: number): number[] {
  let s = seed >>> 0;
  return Array.from({ length: HEAD_DIM }, () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (((t ^ (t >>> 14)) >>> 0) / 4294967296) * 2 - 1;
  });
}
