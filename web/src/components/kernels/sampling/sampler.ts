export type Dump = { tokens: string[]; logits: number[]; tail: [number, number][] };

export type Group = { logit: number; count: number; label: string };

export type Knobs = { temperature: number; topK: number; topP: number };

export type Stages = {
  scaled: number[];
  base: number[];
  afterK: number[];
  keptK: boolean[];
  afterP: number[];
  final: number[];
  kept: boolean[];
  survivors: number;
  entropy: number;
  greedy: number;
};

export function softmaxMass(scaled: number[], counts: number[], keep: boolean[] | null = null): number[] {
  const top = Math.max(...scaled.filter((_, i) => !keep || keep[i]));
  const mass = scaled.map((s, i) => (!keep || keep[i] ? counts[i] * Math.exp(s - top) : 0));
  const total = mass.reduce((a, b) => a + b, 0);
  return mass.map((m) => m / total);
}

export function topKKeep(counts: number[], k: number): boolean[] {
  if (k <= 0) return counts.map(() => true);
  let before = 0;
  return counts.map((c) => {
    const keep = before < k;
    before += c;
    return keep;
  });
}

export function topPKeep(mass: number[], keep: boolean[], p: number): boolean[] {
  if (p >= 1) return keep;
  const total = mass.reduce((a, m, i) => a + (keep[i] ? m : 0), 0);
  const target = p * total;
  let before = 0;
  return mass.map((m, i) => {
    if (!keep[i]) return false;
    const kept = before < target;
    before += m;
    return kept;
  });
}

export function entropyBits(mass: number[], counts: number[]): number {
  return mass.reduce((a, m, i) => (m > 0 ? a - m * Math.log2(m / counts[i]) : a), 0);
}

export function stages(groups: Group[], { temperature, topK, topP }: Knobs): Stages {
  const counts = groups.map((g) => g.count);
  if (temperature <= 0) {
    const one = groups.map((_, i) => (i === 0 ? 1 : 0));
    const raw = softmaxMass(groups.map((g) => g.logit), counts);
    return { scaled: groups.map((g) => g.logit), base: raw, afterK: raw, keptK: raw.map(() => true), afterP: raw, final: one, kept: one.map((v) => v === 1), survivors: 1, entropy: 0, greedy: 0 };
  }
  const scaled = groups.map((g) => g.logit / temperature);
  const base = softmaxMass(scaled, counts);
  const keptK = topKKeep(counts, topK);
  const afterK = base.map((m, i) => (keptK[i] ? m : 0));
  const kept = topPKeep(afterK, keptK, topP);
  const afterP = base.map((m, i) => (kept[i] ? m : 0));
  const final = softmaxMass(scaled, counts, kept);
  const survivors = groups.reduce((a, g, i) => a + (kept[i] ? g.count : 0), 0);
  return { scaled, base, afterK, keptK, afterP, final, kept, survivors, entropy: entropyBits(final, counts), greedy: 0 };
}

export function uniform(seed: number, step: number, i: number): number {
  let h = (Math.imul(seed ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(step + 1, 0xc2b2ae35) ^ Math.imul(i + 1, 0x27d4eb2f)) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d) >>> 0;
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39) >>> 0;
  h = (h ^ (h >>> 15)) >>> 0;
  return (h + 0.5) / 4294967296;
}

export function gumbel(u: number): number {
  return -Math.log(-Math.log(u));
}

export function draw(groups: Group[], knobs: Knobs, seed: number, step: number): number {
  const s = stages(groups, knobs);
  if (knobs.temperature <= 0) return s.greedy;
  let best = -1;
  let bestKey = -Infinity;
  groups.forEach((g, i) => {
    if (!s.kept[i]) return;
    const key = s.scaled[i] + Math.log(g.count) + gumbel(uniform(seed, step, i));
    if (key > bestKey) {
      bestKey = key;
      best = i;
    }
  });
  return best;
}

export function buildGroups({ tokens, logits, tail }: Dump): Group[] {
  const floor = logits[logits.length - 1];
  return [
    ...tokens.map((label, i) => ({ logit: logits[i], count: 1, label })),
    ...tail.map(([logit, count]) => ({ logit: Math.min(logit, floor), count, label: "other" })),
  ];
}
