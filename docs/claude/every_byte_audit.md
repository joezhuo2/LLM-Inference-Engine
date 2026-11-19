# Every Byte as a beginner's tool: audit, 2026-11-18

Audited at main `52e3bcf` (static build of `web/`, served locally).

**Verdict.** Every Byte is a very good second resource and a hard first one. A reader who already has the transformer diagram in their head and one evening gets the central idea of inference (a token costs the bytes read for it) better here than anywhere I know. A true beginner stalls in Part II, where most pages are about tuning CUDA kernels, not about inference. The fixes are mostly about the path through the book, not the content.

## How it was checked

| Check | Done |
|---|---|
| Chapters 1, 2, 3, start here, one-hour path | read in full |
| Chapters 4 to 19 | headings and the opening of every section, plus every review |
| 3D book | headless Chrome at 1440×860 and 390×844: cover, start here, foundations lead-in, steps, sandbox, review, exercise |
| CPU exercise | BlockManager run as the page says, from a fresh clone: builds, 11 tests fail as promised |
| NumPy exercises | stub fails, all 7 solutions print `ok` |
| Glossary | every acronym in the 19 chapters looked up against the 136 terms |

Not checked: frame rate and load time on a real GPU (headless used software rendering), dark theme, the GPU exercises, dashboard and results pages.

## What works

1. **One thesis, stated early, repeated with numbers.** "Decode waits on memory" arrives in chapter 2 and every later chapter is judged against it. Chapter 19 closes the loop. Most inference material never gives a beginner a single idea to hold.
2. **The front matter is honest.** "Start here" says who the book is for, what to know first, what it leaves out, and where to go if transformers are new (Karpathy, Alammar).
3. **Chapter 1 is the right opening.** One real prompt, one real token, seven stations, each with a measured time. It ends by correcting the usual wrong guess (tokenizer and sampler are under 1%).
4. **The chapter shape is sound.** "What it is" → idea steps → "how it got built" → try it → review → build it. The idea steps and the build history are already separate groups.
5. **"Predict first" and "check yourself".** Readers commit to an answer before the figure or the answer opens. This is the strongest teaching device in the book.
6. **Measured, calculated and illustrative are labelled everywhere.** A beginner learns what a trustworthy number looks like.
7. **Glossary coverage is complete.** No unexplained acronym turned up in any chapter. Terms are underlined on first use per spread.
8. **A real no-GPU track exists.** 14 of 22 exercises start without an NVIDIA GPU (7 CPU, 7 NumPy), plus 8 paper estimates. The CPU flow works exactly as written.
9. **Dead ends and "beyond this engine".** Showing 11 changes that failed, and what the engine does not do, is rare and useful.
10. **Fallbacks.** Reading view for every chapter, a no-WebGL message, arrow keys, reduced motion, a text article under the book on phones.

## What does not work for a beginner

1. **The book is 515 pages and about 43,000 words, and nothing says which pages a beginner can skip.** That is 3.5 hours of reading before any figure. The one-hour path solves this, but it is a separate mode; the full book has no "idea" versus "deep dive" marks in its contents.
2. **Part II changes subject.** Chapters 4 and 7 to 11 are mostly about warps, wide loads, shuffles, `mma.sync` and bit-exact rounding. The book itself says each of these kernels is under 2% of a step. This is kernel engineering, and a reader who came to learn inference meets six chapters of it in a row.
3. **The attention primer comes late.** "Attention, step by step" (query, key, value, softmax, mask, heads) is in chapter 6. RoPE (chapter 4) and the paged KV cache (chapter 5) both need it and each re-explains a short version. The order follows the token, not the learner.
4. **Too many exact numbers in the prose.** "9.53 ms at 16, 12.86 ms at 32, 16.93 ms at 64, 32.9 ms at 128", "22,528 bytes", "12,915 blocks". The captions already hold the exact values. In body text a beginner cannot tell which three numbers to remember.
5. **The first exercise needs a GPU and has no tests.** Chapter 2's "measure the bandwidth roof" is GPU only and open ended. Chapters 1 and 3 have none. The first exercise a laptop reader can run is in chapter 5.
6. **Sandboxes are split across spreads.** In foundations the control is on one spread and the numbers on the next ("go back a page, change the op, and come back"). Cause and effect are never on screen together.
7. **Reading comfort in the 3D book.** Body text renders at about 13 px on a 1440-wide screen and exhibit labels at about 9 px. Text on the canvas cannot be selected, searched or translated. On a phone the article below the book reads well, but exhibit labels are unreadable.
8. **You cannot keep your place.** The URL holds the chapter, not the spread, so a reload returns to the chapter opening. Nothing remembers where the reader stopped in a 515-page book.
9. **Beginners cannot find it.** The top bar has no entry for Every Byte or the exercises (hidden on purpose in `f0cb732`). The only way in from the home page is a muted footer link.

## Needs fixing

Small, concrete, cheap. Ordered by how much confusion each one causes.

| # | Problem | Where |
|---|---|---|
| 1 | Three "time per token" values with no bridge: 8.65 ms in chapter 1, then 6.9 ms and 6.4 ms in chapter 2. One sentence in chapter 2 should say chapter 1's step was launched kernel by kernel and this one is replayed from a graph. | `components/kernels/foundations/ideas.tsx` |
| 2 | The lead-in caption calls 6.4 ms "measured"; the body calls 6.9 ms measured and 6.4 ms calculated. | `components/kernels/foundations/about.tsx:55` |
| 3 | Stale cross-reference: "the scheduler in the paged KV cache chapter". The scheduler is chapter 17, how many requests fit. | `components/kernels/foundations/ideas.tsx:89` |
| 4 | "109% of the roof" in the sandbox and "99 to 110%" in GEMM. A roof you can exceed needs one line: the roof is what a plain copy kernel reached, not a physical limit. | `components/kernels/gemm/story.tsx`, foundations sandbox |
| 5 | The KV pool is 206,640 token slots in chapter 2 and 206,624 tokens in chapter 16 (the null block). Use one, or say why. | `foundations/ideas.tsx:99`, memory-budget |
| 6 | The one-hour path skips chapter 1 and sampling. A reader finishes it without the map of a token or how a token is picked. Add chapter 1's lead-in spread. | `content/short-path.ts` |
| 7 | The browser tab title never changes while turning into the next chapter (URL said `/gemm`, title said foundations). | `components/book/every-byte.tsx` |
| 8 | One heading is in title case: "Prefill vs. Decode (the token lifecycle)". | `content/kernels/paged-attention.ts:23` |
| 9 | Each NumPy stub has the looped `reference` function directly above the function to write, and no line saying what the task is. Say in the file that the job is the vectorised version. | `docs/exercises/numpy/*.py` |

## Good to have

Ordered by what I think helps a beginner most.

1. **An "ideas only" switch for the whole book.** The one-hour path already knows which steps are idea steps. Offer the same filter on all 19 chapters, and mark "how it got built" steps as deep dives in each chapter's contents. This answers problems 1 and 2 without cutting a page.
2. **Move the attention primer forward.** Put "attention, step by step" before RoPE, or in chapter 1 after the layer spread. Then chapters 4 and 5 can drop their own short versions.
3. **A no-GPU exercise in each of the first three chapters.**
   - Chapter 1: a NumPy decode step for one token (lookup, one layer, LM head, argmax) against saved logits.
   - Chapter 2: measure memory bandwidth on the CPU with a large NumPy copy, and place your own machine on a roofline.
   - Chapter 12: Gumbel-max sampling in NumPy. It is ten lines and proves "sampling is an argmax".
4. **Remember the reader's place.** Put the spread in the URL and keep the last spread in `localStorage`, with "continue at page N" on the cover.
5. **Round numbers in body text.** Keep exact values in captions and tables. A rule such as "two significant figures in prose" would do it.
6. **"On a bigger model" box in each idea chapter.** Chapter 16 does this once with Llama 3 8B. Readers want to carry the roofline, the KV bytes per token and the batch limit to a model they will actually serve. `content/published.ts` already has the specs.
7. **A quantization chapter.** It is the direct answer to the book's thesis (move fewer bytes) and the one technique every beginner has heard of. Today it is half a page in "beyond". It is already on the roadmap.
8. **One screen for each sandbox.** Put the control and its readout on the same spread, or mirror the control on the readout page.
9. **A glossary spread and search.** The 136 terms exist only as hover cards. A glossary page in the back matter, and text search in the reading view, make the book usable as a reference.
10. **A map to other engines.** One table at the end: each chapter's idea → where it lives in vLLM and llama.cpp. It tells the reader what to open next.
11. **An entry on the top bar when the book is ready for readers.** Until then the footer link is the only door.

## Left as it is on purpose

- Hiding Every Byte and exercises from the top bar is Joe's decision (`f0cb732`); listed above only because it decides who finds the book.
- One sandbox part per spread is a rule Joe set for the book. Item 8 would change it.
- Query, key and value are explained in four chapters. That repetition is right for chapters read alone and only costs time in book order.
