import assert from "node:assert/strict";
import { test } from "node:test";
import { bookOrder } from "../src/content/book-order.ts";
import { engineHref, engineKeys, engineRows, engines, enginesChecked } from "../src/content/engines-map.ts";

test("every chapter of the book has exactly one row, in book order", () => {
  assert.deepEqual(
    engineRows.map((r) => r.slug),
    bookOrder,
  );
});

test("every cell names a file, a line and a symbol, or says plainly that there is none", () => {
  for (const r of engineRows) {
    assert.ok(r.idea.trim(), r.slug);
    for (const k of engineKeys) {
      const c = r[k];
      if ("none" in c) assert.ok(c.none.trim(), `${r.slug} ${k}`);
      else {
        assert.ok(c.path.trim() && !c.path.startsWith("/"), `${r.slug} ${k} path`);
        assert.ok(Number.isInteger(c.line) && c.line > 0, `${r.slug} ${k} line`);
        assert.ok(c.symbol.trim() && c.note.trim(), `${r.slug} ${k} symbol and note`);
        assert.ok(engineHref(k, c).includes(`/blob/${engines[k].tag}/${c.path}#L${c.line}`), `${r.slug} ${k} link`);
      }
    }
  }
});

test("both engines are pinned to a tag and the date checked is set", () => {
  for (const k of engineKeys) {
    assert.match(engines[k].repo, /^https:\/\/github\.com\/[\w.-]+\/[\w.-]+$/);
    assert.ok(engines[k].tag && engines[k].tag !== "main" && engines[k].tag !== "master", k);
  }
  assert.match(enginesChecked, /^\d{4}-\d{2}-\d{2}$/);
});
