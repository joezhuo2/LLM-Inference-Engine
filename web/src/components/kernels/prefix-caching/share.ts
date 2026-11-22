export type ShareKnobs = { shared: number; own: number; requests: number; atOnce: boolean; budget: number; maxBatch: number; block: number };

export type ShareRun = { perHit: number; firstStep: number; cached: number[]; total: number; cachedTotal: number; prefill: number; hitRate: number };

export function cachedOf(shared: number, own: number, block: number): number {
  const prompt = shared + own;
  return Math.min(Math.floor(shared / block), Math.floor((prompt - 1) / block)) * block;
}

export function shareRun(k: ShareKnobs): ShareRun {
  const prompt = k.shared + k.own;
  const firstStep = k.atOnce ? Math.max(1, Math.min(k.requests, k.maxBatch, Math.floor(k.budget / prompt))) : 1;
  const perHit = cachedOf(k.shared, k.own, k.block);
  const cached = Array.from({ length: k.requests }, (_, i) => (i < firstStep ? 0 : perHit));
  const total = k.requests * prompt;
  const cachedTotal = cached.reduce((a, b) => a + b, 0);
  return { perHit, firstStep, cached, total, cachedTotal, prefill: total - cachedTotal, hitRate: total ? cachedTotal / total : 0 };
}
