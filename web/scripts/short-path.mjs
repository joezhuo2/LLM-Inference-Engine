import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const web = resolve(fileURLToPath(import.meta.url), "../..");
const pages = resolve(web, "out/kernels");
const out = resolve(web, "src/content/short-path-steps.json");

const entities = { "&#x27;": "'", "&#39;": "'", "&amp;": "&", "&quot;": '"', "&lt;": "<", "&gt;": ">" };
const text = (html) => html.replace(/<(script|style|svg)\b[\s\S]*?<\/\1>/g, " ").replace(/<[^>]+>/g, " ").replace(/&#x27;|&#39;|&amp;|&quot;|&lt;|&gt;/g, (e) => entities[e]).replace(/\s+/g, " ").trim();
const words = (html) => text(html).split(" ").filter((w) => /[\p{L}\p{N}]/u.test(w)).length;
const attr = (tag, name) => tag.match(new RegExp(`\\b${name}="([^"]*)"`))?.[1];

export function countPage(html) {
  const heads = [...html.matchAll(/<h2 id="([^"]+)"/g)].map((m) => ({ id: m[1], at: m.index }));
  const page = { about: 0, steps: [] };
  heads.forEach((head, k) => {
    const part = html.slice(head.at, heads[k + 1]?.at ?? html.length);
    const opens = [...part.matchAll(/<section\b[^>]*\bid="step-[^"]*"[^>]*>/g)];
    opens.forEach((open, i) => {
      const body = part.slice(open.index + open[0].length, opens[i + 1]?.index ?? part.length);
      const title = body.match(/<h2\b[^>]*>([\s\S]*?)<\/h2>/);
      const count = words(body.slice(title ? title.index + title[0].length : 0));
      if (head.id === "about") page.about += count;
      else page.steps.push({ id: attr(open[0], "id").slice(5), title: title ? text(title[1]) : "", ...(attr(open[0], "data-group") ? { group: attr(open[0], "data-group") } : {}), words: count });
    });
  });
  return page;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  if (!existsSync(pages)) throw new Error("out/kernels is missing: run npm run build first, this script counts the words of the built pages");
  const counted = {};
  for (const file of readdirSync(pages).filter((f) => f.endsWith(".html")).sort()) {
    const page = countPage(readFileSync(resolve(pages, file), "utf8"));
    if (page.steps.length) counted[file.slice(0, -5)] = page;
  }
  writeFileSync(out, `${JSON.stringify(counted, null, 1)}\n`);
  console.log(`short-path: counted ${Object.keys(counted).length} chapters, ${Object.values(counted).reduce((n, p) => n + p.steps.length, 0)} steps`);
}
