import { graphRule, modelShape } from "@/lib/data";

export function decodeGraphBuckets(maxBatch: number, stepsPerOctave: number): number[] {
  if (maxBatch < 1) throw new Error("max batch must be at least 1");
  if (stepsPerOctave < 1 || (stepsPerOctave & (stepsPerOctave - 1)) !== 0) throw new Error("steps per octave must be a power of two");
  const buckets: number[] = [];
  for (let octave = 1; octave < maxBatch; octave *= 2) {
    const step = Math.max(1, Math.floor(octave / stepsPerOctave));
    for (let b = octave; b < 2 * octave && b < maxBatch; b += step) buckets.push(b);
  }
  if (buckets[buckets.length - 1] !== maxBatch) buckets.push(maxBatch);
  return buckets;
}

export function bucketFor(rows: number, buckets: number[]): number {
  const bucket = buckets.find((b) => rows <= b);
  if (bucket === undefined) throw new Error(`no bucket holds ${rows} rows`);
  return bucket;
}

export function attentionSplits(rows: number, context: number): number {
  const rule = graphRule();
  const heads = modelShape().heads;
  let splits = 1;
  while (splits < rule.max_splits && 2 * splits * rows * heads <= rule.split_blocks && context >= 2 * splits * rule.min_split_tokens) splits *= 2;
  return splits;
}

export function capturedSplits(bucket: number, maxContext: number): number[] {
  const seen = new Set<number>();
  for (let context = 0; context <= maxContext; context++) seen.add(attentionSplits(bucket, context));
  return [...seen].sort((a, b) => a - b);
}

export function splitThresholds(bucket: number, maxContext: number): { context: number; splits: number }[] {
  const found: { context: number; splits: number }[] = [];
  for (let context = 0; context <= maxContext; context++) {
    const splits = attentionSplits(bucket, context);
    if (found[found.length - 1]?.splits !== splits) found.push({ context, splits });
  }
  return found;
}
