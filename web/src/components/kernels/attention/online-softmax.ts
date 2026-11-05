const f = Math.fround;

export const START_MAX = -1e20;

export type Problem = {
  contextLen: number;
  blockSize: number;
  headDim: number;
  poolBlocks: number;
  table: number[];
  q: number[];
  k: Map<number, number[][]>;
  v: Map<number, number[][]>;
};

export type Step = {
  logical: number;
  physical: number;
  tokens: number;
  scores: number[];
  weights: number[];
  blockMax: number;
  maxBefore: number;
  maxAfter: number;
  rescale: number;
  sumBefore: number;
  sumAfter: number;
  accBefore: number[];
  accAfter: number[];
  grew: boolean;
};

export type Run = { steps: Step[]; out: number[] };

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeProblem(contextLen: number, blockSize: number, seed: number, headDim = 4, poolBlocks = 64): Problem {
  const rand = mulberry32(seed);
  const blocks = Math.ceil(contextLen / blockSize);
  const ids = Array.from({ length: poolBlocks - 1 }, (_, i) => i + 1);
  for (let i = ids.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [ids[i], ids[j]] = [ids[j], ids[i]];
  }
  const table = ids.slice(0, blocks);
  const vec = (span: number) => Array.from({ length: headDim }, () => f((rand() * 2 - 1) * span));
  const fill = (span: number) => new Map(table.map((id) => [id, Array.from({ length: blockSize }, () => vec(span))]));
  return { contextLen, blockSize, headDim, poolBlocks, table, q: vec(1), k: fill(2), v: fill(1) };
}

export function tokensIn(p: Problem, logical: number) {
  return Math.min(p.blockSize, p.contextLen - logical * p.blockSize);
}

function score(p: Problem, physical: number, slot: number) {
  const key = p.k.get(physical)![slot];
  let dot = 0;
  for (let d = 0; d < p.headDim; d++) dot = f(dot + f(p.q[d] * key[d]));
  return f(dot * f(1 / Math.sqrt(p.headDim)));
}

export function pagedAttention(p: Problem): Run {
  const steps: Step[] = [];
  let m = START_MAX;
  let l = 0;
  let acc: number[] = new Array(p.headDim).fill(0);
  p.table.forEach((physical, logical) => {
    const tokens = tokensIn(p, logical);
    const scores = Array.from({ length: tokens }, (_, slot) => score(p, physical, slot));
    const blockMax = Math.max(...scores);
    const maxAfter = Math.max(m, blockMax);
    const rescale = f(Math.exp(f(m - maxAfter)));
    const weights = scores.map((s) => f(Math.exp(f(s - maxAfter))));
    const sumBefore = l;
    const accBefore = acc;
    l = f(f(l * rescale) + weights.reduce((a, w) => f(a + w), 0));
    acc = accBefore.map((a, d) => {
      let sum = 0;
      weights.forEach((w, slot) => (sum = f(sum + f(w * p.v.get(physical)![slot][d]))));
      return f(f(a * rescale) + sum);
    });
    steps.push({ logical, physical, tokens, scores, weights, blockMax, maxBefore: m, maxAfter, rescale, sumBefore, sumAfter: l, accBefore, accAfter: acc, grew: maxAfter > m });
    m = maxAfter;
  });
  return { steps, out: acc.map((a) => (l > 0 ? f(a / l) : 0)) };
}

export function directAttention(p: Problem): number[] {
  const scores: number[] = [];
  const values: number[][] = [];
  p.table.forEach((physical, logical) => {
    for (let slot = 0; slot < tokensIn(p, logical); slot++) {
      scores.push(score(p, physical, slot));
      values.push(p.v.get(physical)![slot]);
    }
  });
  const max = Math.max(...scores);
  const weights = scores.map((s) => f(Math.exp(f(s - max))));
  const total = weights.reduce((a, w) => f(a + w), 0);
  return Array.from({ length: p.headDim }, (_, d) => f(weights.reduce((a, w, t) => f(a + f(w * values[t][d])), 0) / total));
}

export function maxDiff(a: number[], b: number[]) {
  return Math.max(...a.map((x, i) => Math.abs(x - b[i])));
}
