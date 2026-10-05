// Independent scalar oracle: no imports from the implementation or its plans.
export function referenceRoPE(input, headDim, {
  startPos = 0, base = 10000, heads = 1, layout = 'adjacent',
} = {}) {
  const out = new Float32Array(input);
  const tokens = input.length / (heads * headDim);
  for (let token = 0; token < tokens; token++) {
    for (let head = 0; head < heads; head++) {
      for (let pair = 0; pair < headDim / 2; pair++) {
        const angle = (startPos + token) / Math.pow(base, 2 * pair / headDim);
        const c = Math.cos(angle), s = Math.sin(angle);
        const a = (token * heads + head) * headDim +
          (layout === 'adjacent' ? 2 * pair : pair);
        const b = a + (layout === 'adjacent' ? 1 : headDim / 2);
        out[a] = input[a] * c - input[b] * s;
        out[b] = input[a] * s + input[b] * c;
      }
    }
  }
  return out;
}

export function maxAbsError(actual, expected) {
  if (actual.length !== expected.length) return Infinity;
  let error = 0;
  for (let i = 0; i < actual.length; i++) {
    if (!Number.isFinite(actual[i]) || !Number.isFinite(expected[i])) return Infinity;
    error = Math.max(error, Math.abs(actual[i] - expected[i]));
  }
  return error;
}

export function fixture(length, seed = 1) {
  let state = seed >>> 0;
  return Float32Array.from({ length }, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2147483648 - 1;
  });
}
