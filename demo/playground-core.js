// Pure demo logic: no DOM, no network, no imports. The kernel and the reference
// oracle are passed in, so the same code runs in the browser page and in Node tests.

export const TOLERANCE = 1e-6; // Same absolute tolerance as the repository tests.
export const LIMITS = Object.freeze({
  headDims: [8, 16, 32, 64, 128, 256],
  maxSeqLen: 4096,
  maxHeads: 64,
  // Highest absolute position sampled by the repository's numerical tests
  // (docs/NUMERICS.md). Every rotated position, startPos .. startPos + seqLen - 1,
  // must stay at or below it, so the demo never runs outside the tested range.
  maxPosition: 131071,
  maxStartPos: 131071,
  maxElementsPerTensor: 4 * 1024 * 1024,
  maxCacheBytes: 64 * 1024 * 1024,
});

// Small seeded PRNG (mulberry32) so a run can be repeated exactly from its seed.
export function seededInput(length, seed) {
  let state = seed >>> 0;
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    out[i] = (((t ^ (t >>> 14)) >>> 0) / 4294967296) * 2 - 1;
  }
  return out;
}

// Parse a raw form value. Empty or non-numeric text becomes NaN so validation rejects
// it; Number('') would otherwise silently turn an empty field into 0.
export function parseNumberField(text) {
  const trimmed = String(text ?? '').trim();
  return trimmed === '' ? NaN : Number(trimmed);
}

export const NUMBER_FIELDS = Object.freeze(['seqLen', 'startPos', 'base', 'qHeads', 'kvHeads', 'seed']);

// Form labels: plain English with the code name in parentheses. Validation messages use the same text.
export const FIELD_LABELS = Object.freeze({
  api: 'API',
  layout: 'Layout',
  headDim: 'Head dim (headDim)',
  seqLen: 'Sequence length (seqLen)',
  startPos: 'Start position (startPos)',
  base: 'RoPE base (base)',
  qHeads: 'Q heads (qHeads)',
  kvHeads: 'KV heads (kvHeads)',
  seed: 'Seed (synthetic data)',
  cached: 'Cached trig table',
});

export const LAYOUT_LABELS = Object.freeze({ adjacent: 'Adjacent', 'split-half': 'Split-half' });

// Build a config from raw form values (strings for the number fields).
export function configFromFields(fields) {
  const cfg = {
    api: fields.api,
    layout: fields.layout,
    headDim: Number(fields.headDim),
    cached: Boolean(fields.cached),
  };
  for (const name of NUMBER_FIELDS) cfg[name] = parseNumberField(fields[name]);
  return cfg;
}

// Highest startPos allowed for a sequence length so the last position stays in the tested range.
export function maxStartPosFor(seqLen) {
  return LIMITS.maxPosition - seqLen + 1;
}

// Each problem as { field, message }; `field` names the form control the page marks next to it.
// Messages are complete sentences without a trailing period, each starting with a capital letter.
export function validateConfigFields(cfg) {
  const errors = [];
  const add = (field, message) => errors.push({ field, message });
  const int = (value, min, max, field) => {
    if (typeof value !== 'number' || Number.isNaN(value)) {
      add(field, `${FIELD_LABELS[field]} is empty or not a number; enter an integer from ${min} to ${max}`);
    } else if (!Number.isSafeInteger(value) || value < min || value > max) {
      add(field, `${FIELD_LABELS[field]} must be an integer from ${min} to ${max}`);
    }
  };
  if (cfg.api !== 'single' && cfg.api !== 'packed') add('api', 'API must be single or packed');
  if (cfg.layout !== 'adjacent' && cfg.layout !== 'split-half') add('layout', 'Layout must be adjacent or split-half');
  if (!LIMITS.headDims.includes(cfg.headDim)) add('headDim', `${FIELD_LABELS.headDim} must be one of ${LIMITS.headDims.join(', ')}`);
  int(cfg.seqLen, 1, LIMITS.maxSeqLen, 'seqLen');
  int(cfg.startPos, 0, LIMITS.maxStartPos, 'startPos');
  int(cfg.seed, 0, 0xffffffff, 'seed');
  if (typeof cfg.base !== 'number' || Number.isNaN(cfg.base)) {
    add('base', `${FIELD_LABELS.base} is empty or not a number; enter a value greater than 1 and at most 10000000`);
  } else if (!Number.isFinite(cfg.base) || cfg.base <= 1 || cfg.base > 1e7) {
    add('base', `${FIELD_LABELS.base} must be greater than 1 and at most 10000000`);
  }
  if (cfg.api === 'packed') {
    int(cfg.qHeads, 1, LIMITS.maxHeads, 'qHeads');
    int(cfg.kvHeads, 1, LIMITS.maxHeads, 'kvHeads');
  }
  if (!errors.length) {
    const last = cfg.startPos + cfg.seqLen - 1;
    if (last > LIMITS.maxPosition) {
      add('startPos', `Start position + sequence length - 1 (the last position, ${last}) must be at most ` +
        `${LIMITS.maxPosition}, the highest position the repository's numerical tests cover; ` +
        `with seqLen ${cfg.seqLen}, use startPos ${Math.max(0, maxStartPosFor(cfg.seqLen))} or lower`);
    }
    const heads = cfg.api === 'packed' ? Math.max(cfg.qHeads, cfg.kvHeads) : 1;
    if (cfg.seqLen * heads * cfg.headDim > LIMITS.maxElementsPerTensor) {
      add('seqLen', `Sequence length x heads x head dim must be at most ${LIMITS.maxElementsPerTensor} per tensor`);
    }
    if (cfg.cached && (cfg.startPos + cfg.seqLen) * cfg.headDim * 8 > LIMITS.maxCacheBytes) {
      add('cached', 'Cached plan would exceed 64 MiB; lower the start position or sequence length, ' +
        'or turn the cache off');
    }
  }
  return errors;
}

export function validateConfig(cfg) {
  return validateConfigFields(cfg).map(error => error.message);
}

export function makePlan(kernel, cfg) {
  return kernel.createRoPEPlan(cfg.headDim, cfg.base,
    cfg.cached ? { maxSeqLen: cfg.startPos + cfg.seqLen } : {});
}

export function makeInputs(cfg) {
  if (cfg.api === 'single') return { x: seededInput(cfg.seqLen * cfg.headDim, cfg.seed) };
  return {
    q: seededInput(cfg.seqLen * cfg.qHeads * cfg.headDim, cfg.seed),
    k: seededInput(cfg.seqLen * cfg.kvHeads * cfg.headDim, (cfg.seed + 1) >>> 0),
  };
}

function rotate(kernel, cfg, plan, tensors) {
  if (cfg.api === 'single') {
    const fn = cfg.layout === 'adjacent' ? kernel.applyRoPEWithPlan : kernel.applyRoPESplitHalfWithPlan;
    fn(tensors.x, plan, { startPos: cfg.startPos });
  } else {
    kernel.applyRoPEQK(tensors.q, tensors.k, plan, {
      qHeads: cfg.qHeads, kvHeads: cfg.kvHeads, startPos: cfg.startPos, layout: cfg.layout,
    });
  }
  return tensors;
}

export function pairsPerCall(cfg) {
  const heads = cfg.api === 'packed' ? cfg.qHeads + cfg.kvHeads : 1;
  return cfg.seqLen * heads * cfg.headDim / 2;
}

function l2(values) {
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += values[i] * values[i];
  return Math.sqrt(sum);
}

// Rotate copies of the inputs with the kernel and compare with the scalar reference.
export function runCheck(kernel, reference, cfg) {
  const errors = validateConfig(cfg);
  if (errors.length) throw new RangeError(errors.join('; '));
  const plan = makePlan(kernel, cfg);
  const inputs = makeInputs(cfg);
  const outputs = rotate(kernel, cfg, plan, Object.fromEntries(
    Object.entries(inputs).map(([name, value]) => [name, new Float32Array(value)])));
  let maxError = 0, normIn = 0, normOut = 0;
  for (const name of Object.keys(inputs)) {
    const heads = name === 'x' ? 1 : name === 'q' ? cfg.qHeads : cfg.kvHeads;
    const expected = reference.referenceRoPE(inputs[name], cfg.headDim, {
      startPos: cfg.startPos, base: cfg.base, heads, layout: cfg.layout,
    });
    maxError = Math.max(maxError, reference.maxAbsError(outputs[name], expected));
    normIn = Math.hypot(normIn, l2(inputs[name]));
    normOut = Math.hypot(normOut, l2(outputs[name]));
  }
  return {
    pass: maxError <= TOLERANCE,
    maxAbsError: maxError,
    tolerance: TOLERANCE,
    relativeNormChange: Math.abs(normOut - normIn) / normIn,
    plan, inputs, outputs,
  };
}

// Time in-place rotation of one reused buffer. `now` is performance.now in either runtime.
export function timeKernel(kernel, cfg, { now, targetMs = 50, samples = 5, warmupMs = 50 } = {}) {
  const plan = makePlan(kernel, cfg);
  const tensors = makeInputs(cfg);
  const run = () => rotate(kernel, cfg, plan, tensors);
  const warmupEnd = now() + warmupMs;
  let warm = 0;
  while (now() < warmupEnd || warm < 3) { run(); warm++; }
  let iterations = 1;
  for (;;) {
    const start = now();
    for (let i = 0; i < iterations; i++) run();
    const elapsed = now() - start;
    if (elapsed >= targetMs || iterations >= 1 << 16) break;
    iterations = Math.min(1 << 16, Math.max(iterations * 2,
      Math.ceil(iterations * targetMs / Math.max(elapsed, 0.1))));
  }
  const pairs = pairsPerCall(cfg);
  const nsPerPair = [];
  for (let s = 0; s < samples; s++) {
    const start = now();
    for (let i = 0; i < iterations; i++) run();
    nsPerPair.push(((now() - start) * 1e6) / iterations / pairs);
  }
  const sorted = [...nsPerPair].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  return { iterations, samples: nsPerPair, medianNsPerPair: median, pairsPerCall: pairs };
}

// cos(theta) for each displayed position (rows) and frequency pair (columns).
export function cosineGrid(plan, startPos, rows) {
  const { halfDim, invFreq } = plan;
  const grid = new Float64Array(rows * halfDim);
  for (let r = 0; r < rows; r++) {
    for (let i = 0; i < halfDim; i++) grid[r * halfDim + i] = Math.cos((startPos + r) * invFreq[i]);
  }
  return grid;
}

// The (x0, x1) components of pair `pair` in token `token`, head 0, for a given layout.
export function pairAt(tensor, cfg, token, pair) {
  const heads = cfg.api === 'packed' ? cfg.qHeads : 1;
  const base = token * heads * cfg.headDim;
  const a = base + (cfg.layout === 'adjacent' ? 2 * pair : pair);
  const b = a + (cfg.layout === 'adjacent' ? 1 : cfg.headDim / 2);
  return [tensor[a], tensor[b]];
}
