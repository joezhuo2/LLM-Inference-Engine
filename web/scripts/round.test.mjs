import assert from "node:assert/strict";
import { test } from "node:test";
import { about, bytesText, msText, num, sig } from "../src/lib/round.ts";

test("significant figures, with thousands separators", () => {
  assert.equal(sig(9.53), 9.5);
  assert.equal(sig(12.86, 3), 12.9);
  assert.equal(num(12915), "13,000");
  assert.equal(num(206624, 3), "207,000");
  assert.equal(num(0.04123), "0.041");
  assert.equal(num(4.54, 3), "4.54");
});

test("about only where rounding changed the value", () => {
  assert.equal(about(12915), "about 13,000");
  assert.equal(about(100), "100");
  assert.equal(about(2048, 4), "2,048");
  assert.equal(about(15.9, 2, ""), "16");
});

test("units scale with the rounded value", () => {
  assert.equal(bytesText(22528), "about 23 KB");
  assert.equal(bytesText(22528, 5), "22.528 KB");
  assert.equal(bytesText(360448), "about 360 KB");
  assert.equal(bytesText(999600), "about 1 MB");
  assert.equal(bytesText(4.655e9), "about 4.7 GB");
  assert.equal(bytesText(512), "about 510 bytes");
  assert.equal(msText(32.9), "about 33 ms");
  assert.equal(msText(0.0004), "400 ns");
  assert.equal(msText(4592), "about 4.6 s");
});

test("a rounded value is never shown as 0", () => {
  for (const v of [1e-9, 0.0004, 0.049, 3e-7]) {
    assert.notEqual(sig(v), 0);
    assert.notEqual(num(v), "0");
    assert.doesNotMatch(bytesText(v), /^(about )?0 /);
  }
  assert.equal(num(0), "0");
});
