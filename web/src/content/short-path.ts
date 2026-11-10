import { bookOrder } from "./book-order";
import counted from "./short-path-steps.json";

export type ShortChapter = { slug: string; cap?: number; beforeGroups?: string[] };

export const shortPathName = "one-hour path";
export const shortPathQuery = "path=hour";

export const shortPath: ShortChapter[] = [
  { slug: "foundations", cap: 6, beforeGroups: ["idea-memory", "idea-cache", "idea-batch", "bytes", "roofline", "measured"] },
  { slug: "gemm", beforeGroups: ["job", "gemv", "roof", "batch"] },
  { slug: "paged-kv", beforeGroups: ["why", "pool", "append", "reuse"] },
  { slug: "paged-attention", beforeGroups: ["decode", "cache", "job", "blocks"] },
  { slug: "add-rmsnorm", cap: 2, beforeGroups: ["fusion", "round-trip"] },
  { slug: "cuda-graphs", beforeGroups: ["launches", "gaps", "capture", "replay"] },
  { slug: "serving" },
  { slug: "where-the-time-went" },
];

export const defaultCap = 4;

export const reading = { wordsPerMinute: 200, secondsPerExhibit: 30, secondsPerTitle: 10, wordsPerUncountedStep: 110, wordsPerUncountedLead: 250 };

export type PathStep = { id: string; group?: string };
type Counted = { about: number; steps: { id: string; title: string; group?: string; words: number }[] };
const countedSteps: Record<string, Counted | undefined> = counted;

export const shortChapters = (present: (slug: string) => boolean) => shortPath.filter((c) => present(c.slug)).sort((a, b) => bookOrder.indexOf(a.slug) - bookOrder.indexOf(b.slug));

export function pickSteps(chapter: ShortChapter, steps: PathStep[]): number[] {
  const all = steps.map((_, i) => i);
  const reordered = steps.some((s) => s.group);
  const named = !reordered && chapter.beforeGroups ? all.filter((i) => chapter.beforeGroups?.includes(steps[i].id)) : [];
  return (named.length ? named : all.filter((i) => !steps[i].group)).slice(0, chapter.cap ?? defaultCap);
}

export const leadWords = (slug: string) => countedSteps[slug]?.about || reading.wordsPerUncountedLead;
export const stepWords = (slug: string, id: string) => countedSteps[slug]?.steps.find((s) => s.id === id)?.words ?? reading.wordsPerUncountedStep;
export const countedChapter = (slug: string) => countedSteps[slug];

export const pathMinutes = (words: number, exhibits: number, titles: number) => words / reading.wordsPerMinute + (exhibits * reading.secondsPerExhibit + titles * reading.secondsPerTitle) / 60;

export const roundMinutes = (minutes: number) => Math.round(minutes / 5) * 5;

export type RouteChapter = { slug: string; steps: PathStep[] };
export type Route = { chapters: { slug: string; picks: number[] }[]; words: number; spreads: number; minutes: number };

export function shortRoute(chapters: RouteChapter[], startWords: number, startSpreads: number): Route {
  const picked = shortChapters((slug) => chapters.some((c) => c.slug === slug)).map((c) => {
    const steps = chapters.find((x) => x.slug === c.slug)?.steps ?? [];
    return { slug: c.slug, picks: pickSteps(c, steps), steps };
  });
  const words = picked.reduce((n, c) => n + leadWords(c.slug) + c.picks.reduce((w, i) => w + stepWords(c.slug, c.steps[i].id), 0), startWords);
  const exhibits = picked.reduce((n, c) => n + 1 + c.picks.length, startSpreads);
  return { chapters: picked.map(({ slug, picks }) => ({ slug, picks })), words, spreads: exhibits + picked.length, minutes: pathMinutes(words, exhibits, picked.length) };
}
