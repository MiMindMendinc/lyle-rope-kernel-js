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

export function createRoPEPlan(headDim, base = DEFAULT_BASE, options = {}) {
  checkHeadDim(headDim);
  checkBase(base);

  const halfDim = headDim >>> 1;
  const invFreq = new Float32Array(halfDim);
  const invHeadDim = 1.0 / headDim;

  for (let i = 0; i < halfDim; i++) {
    invFreq[i] = 1.0 / Math.pow(base, (2 * i) * invHeadDim);
  }

  const maxSeqLen = options.maxSeqLen ?? 0;
  checkNonNegativeInteger(maxSeqLen, 'maxSeqLen');

  let cosTable = null;
  let sinTable = null;
  if (maxSeqLen > 0) {
    const tableLength = maxSeqLen * halfDim;
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

  return Object.freeze({ headDim, halfDim, base, invFreq, maxSeqLen, cosTable, sinTable });
}

function checkPlan(plan) {
  if (!plan || typeof plan !== 'object' ||
      !Number.isInteger(plan.headDim) || !Number.isInteger(plan.halfDim) ||
      !(plan.invFreq instanceof Float32Array) ||
      plan.headDim !== plan.halfDim * 2 ||
      plan.invFreq.length !== plan.halfDim) {
    throw new TypeError('plan must be created by createRoPEPlan');
  }

  const hasCache = plan.maxSeqLen !== undefined ||
    plan.cosTable !== undefined || plan.sinTable !== undefined;
  if (hasCache && (
    !Number.isInteger(plan.maxSeqLen) || plan.maxSeqLen < 0 ||
    (plan.maxSeqLen === 0 && (plan.cosTable !== null || plan.sinTable !== null)) ||
    (plan.maxSeqLen > 0 && (
      !(plan.cosTable instanceof Float64Array) ||
      !(plan.sinTable instanceof Float64Array) ||
      plan.cosTable.length !== plan.maxSeqLen * plan.halfDim ||
      plan.sinTable.length !== plan.maxSeqLen * plan.halfDim
    ))
  )) {
    throw new TypeError('plan cache must be created by createRoPEPlan');
  }
}

function getApplyConfig(tensor, plan, options) {
  checkTensor(tensor, 'tensor');
  checkPlan(plan);

  const { headDim, halfDim, invFreq } = plan;
  const startPos = options.startPos ?? 0;
  const seqLen = options.seqLen ?? Math.floor(tensor.length / headDim);

  checkNonNegativeInteger(startPos, 'startPos');
  checkNonNegativeInteger(seqLen, 'seqLen');

  if (seqLen * headDim > tensor.length) {
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
  return applyRoPEWithPlan(tensor, createRoPEPlan(headDim, options.base ?? DEFAULT_BASE), options);
}

export function applyRoPESplitHalfWithPlan(tensor, plan, options = {}) {
  return applySplitHalves(tensor, getApplyConfig(tensor, plan, options));
}

export function applyRoPESplitHalf(tensor, headDim, options = {}) {
  return applyRoPESplitHalfWithPlan(
    tensor,
    createRoPEPlan(headDim, options.base ?? DEFAULT_BASE),
    options,
  );
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
