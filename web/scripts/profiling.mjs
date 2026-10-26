export function slug(text) {
  return text
    .toLowerCase()
    .replace(/`/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

function cells(line) {
  return line
    .trim()
    .replace(/^\||\|$/g, "")
    .split("|")
    .map((c) => c.trim());
}

function isSeparator(line) {
  return /^\|?(\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$/.test(line.trim()) || /^\|(\s*-{3,}\s*\|)+$/.test(line.trim());
}

export function parseProfiling(text) {
  const sections = [];
  let current = null;
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const heading = /^(#{2,3})\s+(.+?)\s*$/.exec(line);
    if (heading) {
      const title = heading[2];
      const branch = /\(Branch (\d+)\)/.exec(title);
      current = { id: slug(title), title, level: heading[1].length, branch: branch ? Number(branch[1]) : null, tables: [] };
      sections.push(current);
      continue;
    }
    if (!current || !line.trim().startsWith("|") || !lines[i + 1] || !isSeparator(lines[i + 1])) continue;
    const header = cells(line);
    const rows = [];
    i += 2;
    while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(cells(lines[i++]));
    i--;
    current.tables.push({ header, rows });
  }
  return sections;
}

export function number(cell) {
  const match = /^-?[\d,]*\.?\d+/.exec(cell.replace(/\s/g, ""));
  return match ? Number(match[0].replace(/,/g, "")) : null;
}

export function parseLaunchGaps(text) {
  const steps = {};
  const pattern =
    /^- (\w+): (\d+) steps, (\d+) kernels, median span (\d+) us, median busy (\d+) us, median idle ([\d.]+)% of the span, (\d+) gaps over ([\d.]+) us per step$/;
  for (const line of text.split("\n")) {
    const m = pattern.exec(line.trim());
    if (!m) continue;
    steps[m[1]] = {
      steps: Number(m[2]),
      kernels: Number(m[3]),
      span_us: Number(m[4]),
      busy_us: Number(m[5]),
      idle_pct: Number(m[6]),
      gaps: Number(m[7]),
      gap_us: Number(m[8]),
    };
  }
  return steps;
}
