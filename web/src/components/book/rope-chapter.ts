import type { Beat, Frame, Stop } from "./book-scene";
import type { PageText, Run } from "./typeset";

const spread: Frame = { center: [0, 0, 0.02], width: 2.3, depth: 1.62, az: 0, el: 62 };
const flatRight: Frame = { center: [0, 0, 0], width: 2.08, depth: 1.42, az: 0, el: 89.6 };
const flatNarrow: Frame = { center: [0.5, 0, 0], width: 1.02, depth: 1.42, az: 0, el: 89.6 };
const page: Frame = { center: [0.5, 0.03, -0.02], width: 1.06, depth: 1.42, az: 0, el: 62 };

const overview = (caption: string): Beat => ({ caption, frame: spread, narrow: page });
const close = (caption: string, center: [number, number, number], width: number, depth: number, az: number, el: number): Beat => {
  const frame = { center, width, depth, az, el };
  return { caption, frame, narrow: { ...frame, width: width * 0.9 } };
};

export const ropeBeats: Beat[][] = [
  [
    overview("Eight of the 32 pairs of one head"),
    close("Close up: pair 0 turns once every 6.3 positions", [0.26, 0.05, -0.42], 0.56, 0.42, -24, 34),
    close("From the side: pair 31 has barely moved", [0.74, 0.05, -0.08], 0.56, 0.42, 26, 30),
    close("The rail: position is the only input", [0.5, 0.03, 0.22], 0.9, 0.36, 0, 42),
  ],
  [
    overview("One token's QKV row: 40 heads of 64 values"),
    close("Close up: head 0, the tallest block", [0.2, 0.05, -0.42], 0.5, 0.34, -30, 30),
    close("Value i pairs with value i + 32, not its neighbour", [0.5, 0.08, 0.02], 0.98, 0.4, 0, 28),
  ],
  [
    overview("One last bit, 2,047 times over"),
    close("From the side: the two arms spread with the position", [0.32, 0.18, -0.27], 0.72, 0.5, -8, 16),
    close("Close up: the rounding boundary between two BF16 values", [0.61, 0.13, -0.27], 0.42, 0.34, 18, 24),
  ],
  [
    overview("36 heads × 32 pairs, one chain of 32 per head"),
    close("Close up: every thread walks its 32 pairs in order", [0.36, 0.02, 0.02], 0.72, 0.5, -28, 40),
    close("The jar: 1,152 sines and cosines per token", [0.83, 0.1, -0.03], 0.44, 0.4, 22, 26),
  ],
  [
    overview("Every pair at once: 32 pairs × 8 head groups"),
    close("Close up: whole rows rise, not one cell at a time", [0.36, 0.02, 0.02], 0.72, 0.5, 24, 38),
    close("The second jar: 256 instead of 1,152", [0.83, 0.1, -0.03], 0.44, 0.4, -20, 26),
  ],
  [{ caption: "Pick a token count", frame: flatRight, narrow: flatNarrow }],
  [{ caption: "Rotate it yourself: one position, 32 pairs", frame: flatRight, narrow: flatNarrow }],
  [{ caption: "Rotate it yourself: the score depends on the gap", frame: flatRight, narrow: flatNarrow }],
];

export const ropeInstructions = ["Drag the pink ball along the rail to change the position.", "Watch each value find its partner, 32 places along.", "Watch where the two values land on the steps.", "Watch the chain cross the plate and the jar fill.", "Compare the rows, the clock and the second jar."];

export type ChapterText = { title: string; paragraphs: Run[][] };

const plain = (...paragraphs: string[]): Run[][] => paragraphs.map((text) => [{ text, accent: false }]);

export const ropeWorkbench: (ChapterText & { overlay: string })[] = [
  {
    title: "one position, 32 pairs",
    overlay: "sandbox-one",
    paragraphs: plain(
      "Drag the position and pick a dimension pair on the right. The dial is pair i of a query vector, the two values xᵢ and xᵢ₊₃₂, turned by the angle m · θᵢ, where m is the token's position.",
      "Pair 0 has θ = 1, so it turns a full circle every 6.3 positions. The grid under the dial shows all 32 pairs at the same position: the low pairs spin quickly and tell neighbouring tokens apart, the high pairs barely move.",
      "Pair 31 needs about 47,000 positions for one turn, far longer than TinyLlama's 2,048-token context, so its angle never repeats inside a prompt. The fast pairs repeat often, but the slow ones keep every position's set of angles different.",
      "The vectors are made up for illustration, from seeded random numbers rather than the model. The frequencies, the pairing of dimension i with i + 32 and the rotation are the engine's.",
    ),
  },
  {
    title: "the score depends on the gap",
    overlay: "sandbox-two",
    paragraphs: plain(
      "Attention scores a query against a key with a dot product. After RoPE the query at position m and the key at position n are each turned by their own angles, so their score q·k depends only on the gap m − n, not on where the pair sits in the prompt.",
      "Predict first, then try it: with the gap locked, move both positions together by 100 and watch the score. Then unlock the gap and move one position at a time.",
      "The curve is the score for every gap from −2,048 to 2,048 with the same two vectors. With the gap locked the dot stays put; with it free, the dot slides along the curve.",
      "This is why the kernel only has to turn each vector by its own position: the relative distance falls out of the dot product, with no table of pairwise positions anywhere.",
    ),
  },
];

export function ropeStops(chapter: { index: number; title: string; formula: string }, steps: ChapterText[], captions: string[], workbench: (ChapterText & { overlay: string })[]): Stop[] {
  const kicker = `chapter ${chapter.index} · ${chapter.title}`;
  const pageFor = (i: number, note?: string): PageText => ({ kicker, title: steps[i].title, note, paragraphs: steps[i].paragraphs, folio: String(i * 2 + 41) });
  const exhibits = [0, 1, 2, 3, 3];
  return steps.map((_, i): Stop => {
    const flat = i >= 5;
    return {
      exhibit: flat ? null : exhibits[i],
      spread: i,
      page: pageFor(i, i === 0 ? chapter.formula : undefined),
      figure: `Fig. ${chapter.index}.${i + 1}`,
      caption: captions[i],
      instruction: ropeInstructions[i],
      folio: String(i * 2 + 42),
      overlay: flat ? "tokens" : null,
      beats: ropeBeats[i],
    };
  }).concat(
    workbench.map((w, k): Stop => ({
      exhibit: null,
      spread: steps.length + k,
      page: { kicker: `${kicker} · rotate it yourself`, title: w.title, paragraphs: w.paragraphs, folio: String((steps.length + k) * 2 + 41) },
      figure: "",
      caption: "",
      folio: "",
      overlay: w.overlay,
      beats: ropeBeats[steps.length + k],
    })),
  );
}
