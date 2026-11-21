export function listRows(count: number, starts: number[], top: number, bottom: number, max: number) {
  const step = Math.min(max, (bottom - top) / Math.max(1, count - 1 + starts.length * 1.5));
  const rows: { y: number; k?: number; head?: number }[] = [];
  let y = top;
  for (let k = 0; k < count; k++) {
    if (starts.includes(k)) {
      rows.push({ y: y + step * 0.2, head: k });
      y += step * 1.5;
    }
    rows.push({ y, k });
    y += step;
  }
  return { rows, step, size: Math.min(46, Math.round(step * 0.8)) };
}
