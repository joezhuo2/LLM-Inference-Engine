// Regenerate after a build with: npm run build && node scripts/search-index.mjs (reads web/out/kernels, writes src/content/search-index.json)
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const web = resolve(fileURLToPath(import.meta.url), "../..");
const pages = resolve(web, "out/kernels");
const out = resolve(web, "src/content/search-index.json");

const entities = { "&#x27;": "'", "&#39;": "'", "&amp;": "&", "&quot;": '"', "&lt;": "<", "&gt;": ">", "&nbsp;": " " };
export const text = (html) => html.replace(/<(script|style|svg|button)\b[\s\S]*?<\/\1>/g, " ").replace(/<[^>]+>/g, "").replace(/&#x27;|&#39;|&amp;|&quot;|&lt;|&gt;|&nbsp;/g, (e) => entities[e]).replace(/\s+/g, " ").trim();

export function indexPage(html) {
  const title = text(html.match(/<title>([\s\S]*?)<\/title>/)?.[1] ?? "").replace(/ · llm-inference-engine$/, "");
  const heads = [...html.matchAll(/<h2 id="([^"]+)"[^>]*>([\s\S]*?)<\/h2>/g)].map((m) => ({ id: m[1], title: text(m[2]), at: m.index }));
  const entries = [];
  heads.forEach((head, k) => {
    const part = html.slice(head.at, heads[k + 1]?.at ?? html.length);
    if (head.id === "review") {
      const list = part.match(/<h3\b[^>]*>what to remember<\/h3>\s*<ol\b[^>]*>([\s\S]*?)<\/ol>/);
      if (list) entries.push({ id: "review", title: "what to remember", text: [...list[1].matchAll(/<li\b[^>]*><span\b[^>]*>\d+<\/span>([\s\S]*?)<\/li>/g)].map((m) => text(m[1])).join(" ") });
      return;
    }
    const opens = [...part.matchAll(/<section\b[^>]*\bid="(step-[^"]*)"[^>]*>/g)];
    opens.forEach((open, i) => {
      const body = part.slice(open.index + open[0].length, opens[i + 1]?.index ?? part.length);
      const h = body.match(/<h2\b[^>]*>([\s\S]*?)<\/h2>/);
      const paragraphs = [...body.slice(h ? h.index + h[0].length : 0).matchAll(/<p\b[^>]*>([\s\S]*?)<\/p>/g)].map((m) => text(m[1])).filter(Boolean);
      entries.push({ id: open[1], title: h ? text(h[1]) : head.title, text: paragraphs.join(" ") });
    });
  });
  return { title, entries };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  if (!existsSync(pages)) throw new Error("out/kernels is missing: run npm run build first, this script reads the text of the built pages");
  const index = {};
  for (const file of readdirSync(pages).filter((f) => f.endsWith(".html")).sort()) {
    const page = indexPage(readFileSync(resolve(pages, file), "utf8"));
    if (page.entries.length) index[file.slice(0, -5)] = page;
  }
  writeFileSync(out, `${JSON.stringify(index)}\n`);
  console.log(`search-index: ${Object.keys(index).length} chapters, ${Object.values(index).reduce((n, p) => n + p.entries.length, 0)} entries`);
}
