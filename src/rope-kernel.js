const DEFAULT_BASE = 10000.0;
const createdPlans = new WeakSet();

function checkOptions(options) {
  if (!options || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError('options must be an object');
  }
}

function safeProduct(a, b, name) {
  const result = a * b;
  if (!Number.isSafeInteger(result)) {
    throw new RangeError(name + ' exceeds safe integer range');
  }
  return result;
}

function checkAngleRange(invFreq, startPos, seqLen) {
  if (seqLen > 0 && !Number.isFinite(
    (startPos + seqLen - 1) * Math.max(1, invFreq[invFreq.length - 1]),
  )) throw new RangeError('position and base produce a non-finite angle');
}

function isF32(x) {
  return x instanceof Float32Array;
}

function checkTensor(tensor, name) {
  if (!isF32(tensor)) throw new TypeError(name + ' must be a Float32Array');
}

function checkHeadDim(headDim) {
  if (!Number.isSafeInteger(headDim) || headDim <= 0 || headDim % 2 !== 0) {
    throw new RangeError('headDim must be a positive even integer in the safe integer range');
  }
}

function checkNonNegativeInteger(value, name) {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(name + ' must be a non-negative safe integer');
  }
}

function checkBase(base) {
  if (!Number.isFinite(base) || base <= 0) {
    throw new RangeError('base must be a positive finite number');
  }
}

export function createRoPEPlan(headDim, base = DEFAULT_BASE, options = {}) {
  checkOptions(options);
  checkHeadDim(headDim);
  checkBase(base);
  const maxSeqLen = options.maxSeqLen ?? 0;
  checkNonNegativeInteger(maxSeqLen, 'maxSeqLen');

  // Division does not truncate headDim to 32 bits, unlike >>> 1.
  const halfDim = headDim / 2;
  safeProduct(halfDim, Float64Array.BYTES_PER_ELEMENT, 'frequency byte length');
  const tableLength = safeProduct(maxSeqLen, halfDim, 'cache length');
  safeProduct(tableLength, 2 * Float64Array.BYTES_PER_ELEMENT, 'cache byte length');
  const invFreq = new Float64Array(halfDim);
  for (let i = 0; i < halfDim; i++) {
    // Keep frequency precision until the final Float32 tensor write.
    invFreq[i] = 1.0 / Math.pow(base, (2 * i) / headDim);
    if (!Number.isFinite(invFreq[i])) throw new RangeError('base produces non-finite frequencies');
  }
  checkAngleRange(invFreq, 0, maxSeqLen);

  let cosTable = null;
  let sinTable = null;
  if (maxSeqLen > 0) {
    cosTable = new Float64Array(tableLength);
    sinTable = new Float64Array(tableLength);
    for (let pos = 0; pos < maxSeqLen; pos++) {
      const tableOffset = pos * halfDim;
      for (let i = 0; i < halfDim; i++) {
        const theta = pos * invFreq[i];
        cosTable[tableOffset + i] = Math.cos(theta);
        sinTable[tableOffset + i] = Math.sin(theta);
      }
    }
  }
  const plan = Object.freeze({ headDim, halfDim, base, invFreq, maxSeqLen, cosTable, sinTable });
  createdPlans.add(plan);
  return plan;
}

function checkPlan(plan) {
  // Plans are module-local objects, not a serialization format. Length checks
  // also catch storage detached through structuredClone/worker transfer.
  if (!plan || !createdPlans.has(plan) ||
      plan.invFreq.length !== plan.halfDim ||
      (plan.maxSeqLen > 0 && (
        plan.cosTable.length !== plan.maxSeqLen * plan.halfDim ||
        plan.sinTable.length !== plan.maxSeqLen * plan.halfDim
      ))) {
    throw new TypeError('plan must be created by createRoPEPlan with intact storage');
  }
}

function getApplyConfig(tensor, plan, options) {
  checkOptions(options);
  checkTensor(tensor, 'tensor');
  checkPlan(plan);

  const { headDim, halfDim, invFreq } = plan;
  const startPos = options.startPos ?? 0;
  const seqLen = options.seqLen ?? tensor.length / headDim;

  checkNonNegativeInteger(startPos, 'startPos');
  checkNonNegativeInteger(seqLen, 'seqLen');

  if (tensor.length % headDim !== 0) {
    throw new RangeError('tensor length must be a multiple of headDim');
  }
  if (seqLen > Number.MAX_SAFE_INTEGER - startPos) {
    throw new RangeError('startPos + seqLen exceeds safe integer range');
  }
  checkAngleRange(invFreq, startPos, seqLen);
  if (safeProduct(seqLen, headDim, 'tensor length') > tensor.length) {
    throw new RangeError('seqLen * headDim exceeds tensor length');
  }

  return {
    headDim,
    halfDim,
    invFreq,
    startPos,
    seqLen,
    maxSeqLen: plan.maxSeqLen ?? 0,
    cosTable: plan.cosTable,
    sinTable: plan.sinTable,
  };
}

function applyAdjacentPairs(tensor, config) {
  const { headDim, halfDim, invFreq, startPos, seqLen, maxSeqLen, cosTable, sinTable } = config;
  let offset = 0;
  for (let pos = 0; pos < seqLen; pos++) {
    const absPos = startPos + pos;
    if (absPos !== 0) {
      if (absPos < maxSeqLen) {
        const tableOffset = absPos * halfDim;
        for (let i = 0, idx = offset; i < halfDim; i++, idx += 2) {
          const cos = cosTable[tableOffset + i];
          const sin = sinTable[tableOffset + i];
          const x0 = tensor[idx];
          const x1 = tensor[idx + 1];
          tensor[idx] = x0 * cos - x1 * sin;
          tensor[idx + 1] = x0 * sin + x1 * cos;
        }
      } else {
        for (let i = 0, idx = offset; i < halfDim; i++, idx += 2) {
          const theta = absPos * invFreq[i];
          const cos = Math.cos(theta);
          const sin = Math.sin(theta);
          const x0 = tensor[idx];
          const x1 = tensor[idx + 1];
          tensor[idx] = x0 * cos - x1 * sin;
          tensor[idx + 1] = x0 * sin + x1 * cos;
        }
      }
    }
    offset += headDim;
  }
  return tensor;
}

function applySplitHalves(tensor, config) {
  const { headDim, halfDim, invFreq, startPos, seqLen, maxSeqLen, cosTable, sinTable } = config;
  let offset = 0;
  for (let pos = 0; pos < seqLen; pos++) {
    const absPos = startPos + pos;
    if (absPos !== 0) {
      if (absPos < maxSeqLen) {
        const tableOffset = absPos * halfDim;
        for (let i = 0; i < halfDim; i++) {
          const cos = cosTable[tableOffset + i];
          const sin = sinTable[tableOffset + i];
          const idx0 = offset + i;
          const idx1 = idx0 + halfDim;
          const x0 = tensor[idx0];
          const x1 = tensor[idx1];
          tensor[idx0] = x0 * cos - x1 * sin;
          tensor[idx1] = x0 * sin + x1 * cos;
        }
      } else {
        for (let i = 0; i < halfDim; i++) {
          const theta = absPos * invFreq[i];
          const cos = Math.cos(theta);
          const sin = Math.sin(theta);
          const idx0 = offset + i;
          const idx1 = idx0 + halfDim;
          const x0 = tensor[idx0];
          const x1 = tensor[idx1];
          tensor[idx0] = x0 * cos - x1 * sin;
          tensor[idx1] = x0 * sin + x1 * cos;
        }
      }
    }
    offset += headDim;
  }
  return tensor;
}

export function applyRoPEWithPlan(tensor, plan, options = {}) {
  return applyAdjacentPairs(tensor, getApplyConfig(tensor, plan, options));
}

export function applyRoPE(tensor, headDim, options = {}) {
  checkOptions(options);
  return applyRoPEWithPlan(tensor, createRoPEPlan(headDim, options.base ?? DEFAULT_BASE), options);
}

export function applyRoPESplitHalfWithPlan(tensor, plan, options = {}) {
  return applySplitHalves(tensor, getApplyConfig(tensor, plan, options));
}

export function applyRoPESplitHalf(tensor, headDim, options = {}) {
  checkOptions(options);
  return applyRoPESplitHalfWithPlan(
    tensor,
    createRoPEPlan(headDim, options.base ?? DEFAULT_BASE),
    options,
  );
}

/**
 * Rotate packed Q and K tensors in place. Shapes are [seqLen, qHeads, headDim]
 * and [seqLen, kvHeads, headDim]; both tensors use the same absolute position
 * for every head belonging to a token. V is intentionally left untouched.
 */
export function applyRoPEQK(q, k, plan, options = {}) {
  checkOptions(options);
  checkTensor(q, 'q');
  checkTensor(k, 'k');
  checkPlan(plan);

  const { qHeads, kvHeads, startPos = 0, layout = 'adjacent' } = options;
  if (!Number.isSafeInteger(qHeads) || qHeads <= 0 ||
      !Number.isSafeInteger(kvHeads) || kvHeads <= 0) {
    throw new RangeError('qHeads and kvHeads must be positive safe integers');
  }
  checkNonNegativeInteger(startPos, 'startPos');
  if (layout !== 'adjacent' && layout !== 'split-half') {
    throw new RangeError('layout must be adjacent or split-half');
  }
  const qStride = qHeads * plan.headDim;
  const kStride = kvHeads * plan.headDim;
  if (!Number.isSafeInteger(qStride) || !Number.isSafeInteger(kStride)) {
    throw new RangeError('head count and dimension exceed safe integer range');
  }
  const seqLen = options.seqLen ?? q.length / qStride;
  checkNonNegativeInteger(seqLen, 'seqLen');
  if (!Number.isSafeInteger(seqLen) ||
      q.length !== seqLen * qStride || k.length !== seqLen * kStride ||
      !Number.isSafeInteger(seqLen * qStride) || !Number.isSafeInteger(seqLen * kStride) ||
      seqLen > Number.MAX_SAFE_INTEGER - startPos) {
    throw new RangeError('Q/K lengths must match packed shapes and the position range must be safe');
  }
  checkAngleRange(plan.invFreq, startPos, seqLen);
  // Shared or partially overlapping views would rotate some elements twice.
  if (q.buffer === k.buffer &&
      q.byteOffset < k.byteOffset + k.byteLength &&
      k.byteOffset < q.byteOffset + q.byteLength) {
    throw new RangeError('q and k must not overlap');
  }

  const { halfDim, invFreq, maxSeqLen, cosTable, sinTable } = plan;
  for (let pos = 0; pos < seqLen; pos++) {
    const absPos = startPos + pos;
    if (absPos === 0) continue;
    const cached = absPos < maxSeqLen;
    const tableOffset = absPos * halfDim;
    for (let i = 0; i < halfDim; i++) {
      const theta = cached ? 0 : absPos * invFreq[i];
      const cos = cached ? cosTable[tableOffset + i] : Math.cos(theta);
      const sin = cached ? sinTable[tableOffset + i] : Math.sin(theta);
      for (let kind = 0; kind < 2; kind++) {
        const tensor = kind === 0 ? q : k;
        const count = kind === 0 ? qHeads : kvHeads;
        const stride = kind === 0 ? qStride : kStride;
        for (let head = 0; head < count; head++) {
          const idx0 = pos * stride + head * plan.headDim +
            (layout === 'adjacent' ? i * 2 : i);
          const idx1 = idx0 + (layout === 'adjacent' ? 1 : halfDim);
          const x0 = tensor[idx0];
          const x1 = tensor[idx1];
          tensor[idx0] = x0 * cos - x1 * sin;
          tensor[idx1] = x0 * sin + x1 * cos;
        }
      }
    }
  }
  return { q, k };
}

export function applyToHead(head, pos, headDim, base = DEFAULT_BASE) {
  checkTensor(head, 'head');
  checkHeadDim(headDim);
  checkNonNegativeInteger(pos, 'pos');
  if (head.length !== headDim) {
    throw new RangeError('head length must equal headDim');
  }
  return applyRoPEWithPlan(head, createRoPEPlan(headDim, base), { startPos: pos, seqLen: 1 });
}

export function verifyNormPreservation(original, afterRoPE, tolerance = 1e-5) {
  checkTensor(original, 'original');
  checkTensor(afterRoPE, 'afterRoPE');
  if (original.length !== afterRoPE.length) return false;

  let normOrig = 0;
  let normAfter = 0;
  for (let i = 0; i < original.length; i++) {
    normOrig += original[i] * original[i];
    normAfter += afterRoPE[i] * afterRoPE[i];
  }
  return Math.abs(Math.sqrt(normOrig) - Math.sqrt(normAfter)) < tolerance;
}

export { DEFAULT_BASE };
