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
import { GemmAboutStory } from "./gemm/about";
import { TilingSandbox as GemmSandbox } from "./gemm/sandbox";
import { GemmStory } from "./gemm/story";
import { FoundationsAboutStory } from "./foundations/about";
import { FoundationsIdeasStory } from "./foundations/ideas";
import { FoundationsSandbox } from "./foundations/sandbox";
import { FoundationsStory } from "./foundations/story";
import { PagedKvAboutStory } from "./paged-kv/about";
import { BatchingStory, PagedKvStory } from "./paged-kv/story";
import { KvVisualizer } from "./paged-kv/visualizer";
import { RopeAboutStory } from "./rope/about";
import { RopeSandbox } from "./rope/sandbox";
import { RopeStory } from "./rope/story";
import { SamplingAboutStory } from "./sampling/about";
import { SamplingSandbox } from "./sampling/sandbox";
import { SamplingStory } from "./sampling/story";
import { SplitAboutStory } from "./split-attention/about";
import { SplitAttentionStory } from "./split-attention/story";
import { SwigluAboutStory } from "./swiglu/about";
import { SwigluStory } from "./swiglu/story";

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
    default:
      return null;
  }
}

export function KernelLeadView({ slug }: { slug: string }) {
  switch (slug) {
    case "paged-attention":
      return <PrologueStory />;
    case "paged-kv":
      return <KvVisualizer />;
    default:
      return null;
  }
}

export function KernelPrologueView({ slug }: { slug: string }) {
  switch (slug) {
    case "paged-kv":
      return <BatchingStory />;
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
    default:
      return null;
  }
}
