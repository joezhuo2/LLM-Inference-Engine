import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { codeToHtml } from "shiki";
import { repoUrl } from "@/content/build-log";

export type ExcerptSpec = { file: string; from: string; to: string; caption: string };

export type Excerpt = { file: string; caption: string; startLine: number; endLine: number; html: string; url: string };

function dedent(lines: string[]) {
  const indent = Math.min(...lines.filter((l) => l.trim()).map((l) => l.length - l.trimStart().length));
  return lines.map((l) => l.slice(indent)).join("\n");
}

export async function loadExcerpt(spec: ExcerptSpec): Promise<Excerpt> {
  const lines = readFileSync(resolve(process.cwd(), "..", spec.file), "utf8").split("\n");
  const start = lines.findIndex((l) => l.includes(spec.from));
  if (start < 0) throw new Error(`excerpt start "${spec.from}" not found in ${spec.file}`);
  const offset = lines.slice(start).findIndex((l) => l.includes(spec.to));
  if (offset < 0) throw new Error(`excerpt end "${spec.to}" not found after line ${start + 1} of ${spec.file}`);
  const end = start + offset;
  const html = await codeToHtml(dedent(lines.slice(start, end + 1)), {
    lang: "cpp",
    themes: { light: "github-light-default", dark: "github-dark" },
    defaultColor: false,
  });
  return {
    file: spec.file,
    caption: spec.caption,
    startLine: start + 1,
    endLine: end + 1,
    html,
    url: `${repoUrl}/blob/main/${spec.file}#L${start + 1}-L${end + 1}`,
  };
}
