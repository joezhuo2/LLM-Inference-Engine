export type Shape = { out: number; in: number };

export type Ceilings = { gbs: number; tflops: number };

export const tileSizes = [16, 32, 64, 128];

export const kSteps = [16, 32, 64, 128];

export const rowChoices = [1, 16, 64, 512];

const stages = 2;
const bytes = 2;

export type Tiling = {
  rowTiles: number;
  colTiles: number;
  blocks: number;
  kIters: number;
  tileLoads: number;
  loadBytes: number;
  naiveBytes: number;
  writeBytes: number;
  floorBytes: number;
  reuse: number;
  flops: number;
  paddedFlops: number;
  padWaste: number;
  intensity: number;
  naiveIntensity: number;
  sharedBytes: number;
  memUs: number;
  computeUs: number;
  bound: "memory" | "compute";
  ridge: number;
};

export function tiling(shape: Shape, m: number, tile: number, bk: number, ceil: Ceilings): Tiling {
  const n = shape.out;
  const k = shape.in;
  const rowTiles = Math.ceil(m / tile);
  const colTiles = Math.ceil(n / tile);
  const kIters = Math.ceil(k / bk);
  const blocks = rowTiles * colTiles;
  const loadBytes = (colTiles * m * k + rowTiles * n * k) * bytes;
  const naiveBytes = m * n * 2 * k * bytes;
  const writeBytes = m * n * bytes;
  const flops = 2 * m * n * k;
  const paddedFlops = 2 * rowTiles * tile * colTiles * tile * k;
  const total = loadBytes + writeBytes;
  const memUs = (total / (ceil.gbs * 1e9)) * 1e6;
  const computeUs = (paddedFlops / (ceil.tflops * 1e12)) * 1e6;
  return {
    rowTiles,
    colTiles,
    blocks,
    kIters,
    tileLoads: blocks * kIters * 2,
    loadBytes,
    naiveBytes,
    writeBytes,
    floorBytes: (m * k + n * k) * bytes,
    reuse: naiveBytes / loadBytes,
    flops,
    paddedFlops,
    padWaste: 1 - flops / paddedFlops,
    intensity: flops / total,
    naiveIntensity: flops / (naiveBytes + writeBytes),
    sharedBytes: stages * 2 * tile * bk * bytes,
    memUs,
    computeUs,
    bound: memUs >= computeUs ? "memory" : "compute",
    ridge: (ceil.tflops * 1e12) / (ceil.gbs * 1e9),
  };
}

export function blockLoadBytes(m: number, tile: number, bk: number, done: number, k: number): number {
  const cols = Math.min(done * bk, k);
  return (Math.min(tile, m) + tile) * cols * bytes;
}
