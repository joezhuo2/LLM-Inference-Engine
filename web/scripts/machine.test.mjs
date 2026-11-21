import assert from "node:assert/strict";
import { test } from "node:test";
import { machineAge, machineStages, setMachineInk, tone } from "../src/components/kernels/machine.ts";

const slugs = ["one-token", "foundations", "gemm", "rope", "paged-kv", "paged-attention", "attention-v2", "split-attention", "flash-prefill", "add-rmsnorm", "swiglu", "sampling", "cuda-graphs", "speculative-decoding", "tensor-parallelism", "memory-budget", "serving", "chunked-prefill", "where-the-time-went", "quantization", "beyond", "dead-ends"];

test("the new quantization chapter holds the where-the-time-went stage and later stories keep theirs", () => {
  const stages = machineStages(slugs);
  assert.deepEqual(stages, [...Array.from({ length: 19 }, (_, i) => i), 18, 19, 20]);
  assert.equal(machineAge(stages, -1), -1);
  assert.equal(machineAge(stages, 2.5), 2.5);
  assert.equal(machineAge(stages, 18.5), 18);
  assert.equal(machineAge(stages, 19.5), 18.5);
  assert.equal(machineAge(stages, 21), 20);
  assert.equal(machineAge(stages, 30), 20);
});

test("the dark site theme draws the prototype's colours unchanged and the light theme maps the anchors to its tokens", () => {
  setMachineInk({ bg: "#191813", text: "#e7e2d6", accent: "#a9c3d9", warn: "#e8b7c2" });
  assert.equal(tone("#bd8055"), "#bd8055");
  setMachineInk({ bg: "#e7e2d6", text: "#2b2a24", accent: "#37617f", warn: "#964260" });
  assert.equal(tone("#191813"), "rgb(231,226,214)");
  assert.equal(tone("#e7e2d6"), "rgb(43,42,36)");
  assert.equal(tone("#a9c3d9"), "rgb(55,97,127)");
  assert.equal(tone("#e8b7c2"), "rgb(150,66,96)");
  assert.equal(tone("rgba(17,23,18,0.2)"), "rgba(43,42,36,0.2)");
});
