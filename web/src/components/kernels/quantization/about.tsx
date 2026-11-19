"use client";

import Link from "next/link";
import { bf16Bits, e4m3 } from "@/content/kernels/quantization";
import { StickyStory, type StoryStep } from "../sticky-story";
import { BitsFigure } from "./figures";

export const quantAbout: StoryStep = {
  id: "about",
  title: "what quantization is",
  body: (
    <>
      <p>
        <Link href="/kernels/where-the-time-went" className="link-underline text-accent">
          Where the time went
        </Link>{" "}
        ended on a floor: a decoded token reads every weight once, and nothing can make that read faster than the memory delivers it. The kernels shrank the time above the floor. This chapter is about the floor itself, which is set by one thing: how many bytes the weights take.
      </p>
      <p>
        A weight is a number stored in some bits. The engine stores every weight in BF16: {bf16Bits.sign} sign bit, {bf16Bits.exponent} exponent bits that pick a power of two, and {bf16Bits.mantissa} mantissa bits that say where inside it the number sits. That is 2 bytes. FP8 E4M3 keeps the same three parts in one byte: {e4m3.sign} sign bit, {e4m3.exponent} exponent bits and {e4m3.mantissa} mantissa bits. The 4-bit files llama.cpp users download, GGUF Q4, spend about half a byte per weight.
      </p>
      <p>
        Quantization is storing the weights in such a narrower format. <span className="text-accent">Fewer bytes per weight means fewer bytes per token</span>, and the floor moves with them. The price is precision: {e4m3.mantissa} mantissa bits hold {e4m3.perPower} steps in each power of two, where BF16 holds {bf16Bits.perPower}.
      </p>
      <p>The engine does not run quantized yet. FP8 for its weights and its KV cache is in the tree as stubs with specs and tests, for Joe to write. So this chapter is the plan and the arithmetic: every number in it is calculated, and the measurements arrive when the kernels do.</p>
    </>
  ),
};

export const quantAboutCaption = `to scale in bits; BF16 and E4M3 fields as the spec in src/kernels/fp8_kv.cu describes E4M3; a GGUF Q4 weight is 4 bits before its block's scale; not measured: the engine does not run quantized`;

export function QuantAboutStory() {
  return <StickyStory kicker="the idea" steps={[quantAbout]} caption={() => quantAboutCaption} figure={() => <BitsFigure />} />;
}
