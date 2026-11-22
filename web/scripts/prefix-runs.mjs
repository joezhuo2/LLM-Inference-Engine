const name = /^(on|off)_p(\d+)_q([\d.]+)_r(\d+)_summary\.csv$/;

export const median = (values) => {
  const v = [...values].sort((a, b) => a - b);
  if (!v.length) throw new Error("median of nothing");
  const mid = Math.floor(v.length / 2);
  return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
};

export function prefixRun(file, summary, requests) {
  const match = name.exec(file);
  if (!match) throw new Error(`results/prefix/${file}: the name must read on|off_p<prefix>_q<rate>_r<run>_summary.csv`);
  const first = requests.find((r) => r.id === 0);
  const later = requests.filter((r) => r.id !== 0).map((r) => r.ttft_s);
  if (!first || !later.length) throw new Error(`results/prefix/${file}: needs request 0 and at least one later request`);
  if ((summary.prefix_cache === 1) !== (match[1] === "on") || summary.shared_prefix !== Number(match[2])) throw new Error(`results/prefix/${file}: the summary's prefix_cache or shared_prefix does not match the file name`);
  return {
    cache: match[1],
    shared: Number(match[2]),
    rate: Number(match[3]),
    run: Number(match[4]),
    own: summary.prompt_len,
    requests: requests.length,
    first_ttft_ms: first.ttft_s * 1e3,
    later_ttft_ms: median(later) * 1e3,
    hit_rate: summary.prefix_hit_rate,
    output_tps: summary.output_throughput,
    prefill_tokens: summary.prefill_tokens,
    cached_tokens: summary.cached_tokens,
    date: summary.date,
    gpu: summary.gpu,
  };
}

export function prefixRows(runs) {
  const groups = new Map();
  for (const r of runs) {
    const key = `${r.shared}|${r.own}|${r.rate}|${r.cache}`;
    groups.set(key, [...(groups.get(key) ?? []), r]);
  }
  const of = (group, key) => median(group.map((r) => r[key]));
  return [...groups.values()]
    .map((g) => ({ shared: g[0].shared, own: g[0].own, rate: g[0].rate, arrivals: g[0].rate === 0 ? "at once" : `${g[0].rate}/s`, cache: g[0].cache, requests: g[0].requests, runs: g.length, first_ttft_ms: of(g, "first_ttft_ms"), later_ttft_ms: of(g, "later_ttft_ms"), hit_rate: of(g, "hit_rate"), output_tps: of(g, "output_tps"), prefill_tokens: of(g, "prefill_tokens"), cached_tokens: of(g, "cached_tokens") }))
    .sort((a, b) => a.shared - b.shared || b.rate - a.rate || (a.cache === "on" ? -1 : 1));
}
