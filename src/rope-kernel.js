const DEFAULT_BASE = 10000.0;

function isF32(x) {
  return x instanceof Float32Array;
}

function checkTensor(tensor, name) {
  if (!isF32(tensor)) throw new TypeError(name + ' must be a Float32Array');
}

function checkHeadDim(headDim) {
  if (!Number.isInteger(headDim) || headDim <= 0 || headDim % 2 !== 0) {
    throw new RangeError('headDim must be a positive even integer');
  }
}

function checkNonNegativeInteger(value, name) {
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError(name + ' must be a non-negative integer');
  }
}

function checkBase(base) {
  if (!Number.isFinite(base) || base <= 0) {
    throw new RangeError('base must be a positive finite number');
  }
}

export function createRoPEPlan(headDim, base = DEFAULT_BASE) {
  checkHeadDim(headDim);
  checkBase(base);

  const halfDim = headDim >>> 1;
  const invFreq = new Float32Array(halfDim);
  const invHeadDim = 1.0 / headDim;

  for (let i = 0; i < halfDim; i++) {
    invFreq[i] = 1.0 / Math.pow(base, (2 * i) * invHeadDim);
  }

  return Object.freeze({ headDim, halfDim, base, invFreq });
}

function checkPlan(plan) {
  if (!plan || typeof plan !== 'object' ||
      !Number.isInteger(plan.headDim) || !Number.isInteger(plan.halfDim) ||
      !(plan.invFreq instanceof Float32Array) ||
      plan.headDim !== plan.halfDim * 2 ||
      plan.invFreq.length !== plan.halfDim) {
    throw new TypeError('plan must be created by createRoPEPlan');
  }
}

function applyRoPEWithPlanInternal(tensor, plan, options = {}, splitHalf = false) {
  checkTensor(tensor, 'tensor');
  checkPlan(plan);

  const headDim = plan.headDim;
  const halfDim = plan.halfDim;
  const invFreq = plan.invFreq;
  const startPos = options.startPos ?? 0;
  const seqLen = options.seqLen ?? Math.floor(tensor.length / headDim);

  checkNonNegativeInteger(startPos, 'startPos');
  checkNonNegativeInteger(seqLen, 'seqLen');

  if (seqLen * headDim > tensor.length) {
    throw new RangeError('seqLen * headDim exceeds tensor length');
  }

  let offset = 0;
  for (let pos = 0; pos < seqLen; pos++) {
    const absPos = startPos + pos;
    for (let i = 0; i < halfDim; i++) {
      const theta = absPos * invFreq[i];
      const cos = Math.cos(theta);
      const sin = Math.sin(theta);
      const idx0 = offset + (splitHalf ? i : (i << 1));
      const idx1 = offset + (splitHalf ? i + halfDim : ((i << 1) + 1));
      const x0 = tensor[idx0];
      const x1 = tensor[idx1];
      tensor[idx0] = x0 * cos - x1 * sin;
      tensor[idx1] = x0 * sin + x1 * cos;
    }
    offset += headDim;
  }
  return tensor;
}

export function applyRoPEWithPlan(tensor, plan, options = {}) {
  return applyRoPEWithPlanInternal(tensor, plan, options);
}

export function applyRoPE(tensor, headDim, options = {}) {
  return applyRoPEWithPlan(tensor, createRoPEPlan(headDim, options.base ?? DEFAULT_BASE), options);
}

export function applyRoPESplitHalf(tensor, headDim, options = {}) {
  return applyRoPEWithPlanInternal(tensor, createRoPEPlan(headDim, options.base ?? DEFAULT_BASE), options, true);
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
