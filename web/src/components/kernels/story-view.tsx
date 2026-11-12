"use client";

import { AddRmsnormAboutStory } from "./add-rmsnorm/about";
import { AddRmsnormStory } from "./add-rmsnorm/story";
import { PagedAttentionAboutStory } from "./attention/about";
import { AttentionBasicsStory } from "./attention/basics";
import { PrologueStory } from "./attention/prologue-story";
import { PagedAttentionSandbox } from "./attention/sandbox";
import { PagedAttentionStory } from "./attention/story";
import { AttentionV2AboutStory } from "./attention-v2/about";
import { AttentionV2Story } from "./attention-v2/story";
import { ChunkedPrefillAboutStory } from "./chunked-prefill/about";
import { ChunkedPrefillSandbox } from "./chunked-prefill/sandbox";
import { ChunkedPrefillStory } from "./chunked-prefill/story";
import { CudaGraphsAboutStory } from "./cuda-graphs/about";
import { CudaGraphsStory } from "./cuda-graphs/story";
import { FlashPrefillAboutStory } from "./flash-prefill/about";
import { FlashPrefillStory } from "./flash-prefill/story";
import { GemmAboutStory } from "./gemm/about";
import { TilingSandbox as GemmSandbox } from "./gemm/sandbox";
import { GemmStory } from "./gemm/story";
import { GemmStream } from "./gemm/stream";
import { MemoryAboutStory } from "./memory-budget/about";
import { MemoryStory } from "./memory-budget/story";
import { SpeculativeAboutStory } from "./speculative-decoding/about";
import { SpeculativeStory } from "./speculative-decoding/story";
import { OneTokenAboutStory } from "./one-token/about";
import { OneTokenStory } from "./one-token/story";
import { FoundationsAboutStory } from "./foundations/about";
import { FoundationsIdeasStory } from "./foundations/ideas";
import { FoundationsSandbox } from "./foundations/sandbox";
import { FoundationsStory } from "./foundations/story";
import { PagedKvAboutStory } from "./paged-kv/about";
import { PagedKvStory } from "./paged-kv/story";
import { RopeAboutStory } from "./rope/about";
import { RopeSandbox } from "./rope/sandbox";
import { RopeStory } from "./rope/story";
import { SamplingAboutStory } from "./sampling/about";
import { SamplingSandbox } from "./sampling/sandbox";
import { SamplingStory } from "./sampling/story";
import { ServingAboutStory } from "./serving/about";
import { ServingSandbox } from "./serving/sandbox";
import { ServingStory } from "./serving/story";
import { SplitAboutStory } from "./split-attention/about";
import { SplitAttentionStory } from "./split-attention/story";
import { SwigluAboutStory } from "./swiglu/about";
import { SwigluStory } from "./swiglu/story";
import { TimeAboutStory } from "./where-the-time-went/about";
import { TimeStory } from "./where-the-time-went/story";

export function KernelStoryView({ slug }: { slug: string }) {
  switch (slug) {
    case "foundations":
      return <FoundationsStory />;
    case "paged-attention":
      return <PagedAttentionStory />;
    case "add-rmsnorm":
      return <AddRmsnormStory />;
    case "rope":
      return <RopeStory />;
    case "swiglu":
      return <SwigluStory />;
    case "gemm":
      return <GemmStory />;
    case "cuda-graphs":
      return <CudaGraphsStory />;
    case "paged-kv":
      return <PagedKvStory />;
    case "split-attention":
      return <SplitAttentionStory />;
    case "attention-v2":
      return <AttentionV2Story />;
    case "sampling":
      return <SamplingStory />;
    case "chunked-prefill":
      return <ChunkedPrefillStory />;
    case "flash-prefill":
      return <FlashPrefillStory />;
    case "serving":
      return <ServingStory />;
    case "where-the-time-went":
      return <TimeStory />;
    case "one-token":
      return <OneTokenStory />;
    case "memory-budget":
      return <MemoryStory />;
    case "speculative-decoding":
      return <SpeculativeStory />;
    default:
      return null;
  }
}

export function KernelIdeasView({ slug }: { slug: string }) {
  switch (slug) {
    case "foundations":
      return <FoundationsIdeasStory />;
    case "paged-attention":
      return <AttentionBasicsStory />;
    default:
      return null;
  }
}

export function KernelAboutView({ slug }: { slug: string }) {
  switch (slug) {
    case "foundations":
      return <FoundationsAboutStory />;
    case "gemm":
      return <GemmAboutStory />;
    case "chunked-prefill":
      return <ChunkedPrefillAboutStory />;
    case "flash-prefill":
      return <FlashPrefillAboutStory />;
    case "swiglu":
      return <SwigluAboutStory />;
    case "attention-v2":
      return <AttentionV2AboutStory />;
    case "add-rmsnorm":
      return <AddRmsnormAboutStory />;
    case "cuda-graphs":
      return <CudaGraphsAboutStory />;
    case "rope":
      return <RopeAboutStory />;
    case "paged-kv":
      return <PagedKvAboutStory />;
    case "paged-attention":
      return <PagedAttentionAboutStory />;
    case "split-attention":
      return <SplitAboutStory />;
    case "sampling":
      return <SamplingAboutStory />;
    case "serving":
      return <ServingAboutStory />;
    case "where-the-time-went":
      return <TimeAboutStory />;
    case "one-token":
      return <OneTokenAboutStory />;
    case "memory-budget":
      return <MemoryAboutStory />;
    case "speculative-decoding":
      return <SpeculativeAboutStory />;
    default:
      return null;
  }
}

export function KernelLeadView({ slug }: { slug: string }) {
  switch (slug) {
    case "paged-attention":
      return <PrologueStory />;
    case "gemm":
      return <GemmStream />;
    default:
      return null;
  }
}

export function KernelPrologueView({ slug }: { slug: string }) {
  switch (slug) {
    default:
      return null;
  }
}

export function KernelSandboxView({ slug }: { slug: string }) {
  switch (slug) {
    case "chunked-prefill":
      return <ChunkedPrefillSandbox />;
    case "rope":
      return <RopeSandbox />;
    case "foundations":
      return <FoundationsSandbox />;
    case "gemm":
      return <GemmSandbox />;
    case "paged-attention":
      return <PagedAttentionSandbox />;
    case "sampling":
      return <SamplingSandbox />;
    case "serving":
      return <ServingSandbox />;
    default:
      return null;
  }
}
