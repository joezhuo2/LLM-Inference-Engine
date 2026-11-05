export type Shape = { hidden: number; intermediate: number; layers: number; heads: number; kv_heads: number };

export const ops = ["qkv", "o_proj", "gate_up", "down", "attention", "rmsnorm", "swiglu", "rope"] as const;
export type Op = (typeof ops)[number];

export type Cost = { flops: number; bytes: number };

const BF16 = 2;
export const rowSteps = 90;

export const rowsAt = (step: number) => Math.round(2 ** (step / 10));

function gemm(rows: number, nIn: number, nOut: number, count: number): Cost {
  const params = nIn * nOut;
  return { flops: 2 * rows * params * count, bytes: BF16 * (params + rows * (nIn + nOut)) * count };
}

export function opCost(shape: Shape, op: Op, rows: number, context: number): Cost {
  const headDim = shape.hidden / shape.heads;
  const q = shape.heads * headDim;
  const kv = shape.kv_heads * headDim;
  const { hidden, intermediate, layers } = shape;
  switch (op) {
    case "qkv":
      return gemm(rows, hidden, q + 2 * kv, layers);
    case "o_proj":
      return gemm(rows, q, hidden, layers);
    case "gate_up":
      return gemm(rows, hidden, 2 * intermediate, layers);
    case "down":
      return gemm(rows, intermediate, hidden, layers);
    case "attention":
      return { flops: 4 * rows * q * context * layers, bytes: BF16 * rows * (2 * kv * context + 2 * q) * layers };
    case "rmsnorm":
      return { flops: 5 * rows * hidden * 2 * layers, bytes: BF16 * (2 * rows * hidden + hidden) * 2 * layers };
    case "swiglu":
      return { flops: 5 * rows * intermediate * layers, bytes: BF16 * 3 * rows * intermediate * layers };
    case "rope":
      return { flops: 3 * rows * (q + kv) * layers, bytes: (BF16 * 2 * rows * (q + kv) + 4 * rows) * layers };
  }
}

export type Ceilings = { gbs: number; tflops: number };

export function analyse(cost: Cost, ceilings: Ceilings) {
  const memUs = (cost.bytes / (ceilings.gbs * 1e9)) * 1e6;
  const computeUs = (cost.flops / (ceilings.tflops * 1e12)) * 1e6;
  const timeUs = Math.max(memUs, computeUs);
  return {
    intensity: cost.flops / cost.bytes,
    ridge: (ceilings.tflops * 1e12) / (ceilings.gbs * 1e9),
    memUs,
    computeUs,
    timeUs,
    bound: computeUs > memUs ? ("compute" as const) : ("memory" as const),
    tflops: cost.flops / (timeUs * 1e-6) / 1e12,
  };
}

export function crossoverRows(shape: Shape, op: Op, context: number, ceilings: Ceilings): number | null {
  const last = rowsAt(rowSteps);
  for (let rows = 1; rows <= last; rows++) {
    if (analyse(opCost(shape, op, rows, context), ceilings).bound === "compute") return rows;
  }
  return null;
}
