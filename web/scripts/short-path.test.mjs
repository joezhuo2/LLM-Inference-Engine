import assert from "node:assert/strict";
import { test } from "node:test";
import { countPage } from "./short-path.mjs";

const html = `<h2 id="about">what it is</h2><section id="step-lead"><h2>the lead</h2><p>one two three</p></section><h2 id="story">the story</h2><section class="x" id="step-job"><h2>the job</h2><p>four five <b>six</b></p><svg><text>not counted</text></svg></section><section id="step-past" data-group="how it got built"><h2>the past</h2><p>seven &amp; eight</p></section>`;

test("the lead-in, each step's words and its group are counted from a built page", () => {
  assert.deepEqual(countPage(html), {
    about: 3,
    steps: [
      { id: "job", title: "the job", words: 3 },
      { id: "past", title: "the past", group: "how it got built", words: 2 },
    ],
  });
});
