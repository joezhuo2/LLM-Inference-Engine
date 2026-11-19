import assert from "node:assert/strict";
import { test } from "node:test";
import { indexPage } from "./search-index.mjs";

const html = `<title>RoPE · llm-inference-engine</title><h2 id="about">what it is</h2><section id="step-about"><h2>what <b>RoPE</b> does</h2><p>short for <span>rotary</span> position&#x27;s embedding</p><svg><text>not kept</text></svg></section><h2 id="how">how it works</h2><section class="x" id="step-job"><h2>the job</h2><p>one</p><figure><figcaption>no</figcaption></figure><p>two &amp; three</p></section><h2 id="review">review</h2><div><p>the lesson</p><h3>what to remember</h3><ol><li><span>1</span><span>first <span>line</span></span></li><li><span>2</span><span>second</span></li></ol><h3>check yourself</h3><ol><li><p>a question</p></li></ol></div><h2 id="code">code</h2><p>skipped</p>`;

test("step titles, step paragraphs and the review's what-to-remember lines are read from a built page", () => {
  assert.deepEqual(indexPage(html), {
    title: "RoPE",
    entries: [
      { id: "step-about", title: "what RoPE does", text: "short for rotary position's embedding" },
      { id: "step-job", title: "the job", text: "one two & three" },
      { id: "review", title: "what to remember", text: "first line second" },
    ],
  });
});
