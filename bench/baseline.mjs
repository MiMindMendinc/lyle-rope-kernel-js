// Fair scalar baseline for the throughput bench: what a straightforward hand-written
// in-place RoPE loop does without this package's plans. It writes into the input
// tensor (no output allocation), computes the inverse frequencies once per call into
// a caller-provided scratch array, and calls Math.cos/Math.sin for every pair.
// Adjacent layout only. It is a comparison point, not part of the package.
export function plainScalarRoPEInPlace(x, headDim, base, invFreqScratch) {
  const half = headDim / 2;
  for (let i = 0; i < half; i++) invFreqScratch[i] = 1 / Math.pow(base, (2 * i) / headDim);
  const tokens = x.length / headDim;
  for (let pos = 0; pos < tokens; pos++) {
    const offset = pos * headDim;
    for (let i = 0; i < half; i++) {
      const theta = pos * invFreqScratch[i];
      const c = Math.cos(theta), s = Math.sin(theta);
      const a = offset + 2 * i;
      const x0 = x[a], x1 = x[a + 1];
      x[a] = x0 * c - x1 * s;
      x[a + 1] = x0 * s + x1 * c;
    }
  }
  return x;
}
