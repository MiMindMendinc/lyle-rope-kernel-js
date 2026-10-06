/** Educational single-head, non-causal attention. Not a transformer or KV cache. */
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { applyRoPEWithPlan, createRoPEPlan } from '../src/rope-kernel.js';

export class SingleHeadAttention {
  constructor(headDim = 128) {
    if (arguments.length > 1) throw new TypeError('This example supports one head; no nHeads argument');
    this.ropePlan = createRoPEPlan(headDim);
    this.headDim = headDim;
    this.scale = 1 / Math.sqrt(headDim);
  }

  forward(q, k, v, startPos = 0) {
    for (const tensor of [q, k, v]) {
      if (!(tensor instanceof Float32Array)) throw new TypeError('Q/K/V must be Float32Array');
      if (tensor.length !== q.length || tensor.length % this.headDim !== 0) {
        throw new RangeError('Q/K/V must have equal [seqLen, headDim] shapes');
      }
      if (!tensor.every(Number.isFinite)) throw new RangeError('Q/K/V must contain finite values');
    }
    const seqLen = q.length / this.headDim;
    const scoreCount = seqLen * seqLen;
    if (!Number.isSafeInteger(scoreCount * Float64Array.BYTES_PER_ELEMENT)) {
      throw new RangeError('attention score allocation exceeds safe integer range');
    }
    // The kernel is in-place; this educational wrapper preserves caller inputs.
    const rq = new Float32Array(q), rk = new Float32Array(k);
    applyRoPEWithPlan(rq, this.ropePlan, { startPos });
    applyRoPEWithPlan(rk, this.ropePlan, { startPos });
    const scores = new Float64Array(scoreCount);
    const out = new Float32Array(q.length);
    for (let i = 0; i < seqLen; i++) {
      let max = -Infinity;
      for (let j = 0; j < seqLen; j++) {
        let dot = 0;
        for (let d = 0; d < this.headDim; d++) {
          dot += rq[i * this.headDim + d] * rk[j * this.headDim + d];
        }
        scores[i * seqLen + j] = dot * this.scale;
        max = Math.max(max, scores[i * seqLen + j]);
      }
      let sum = 0;
      for (let j = 0; j < seqLen; j++) {
        scores[i * seqLen + j] = Math.exp(scores[i * seqLen + j] - max);
        sum += scores[i * seqLen + j];
      }
      for (let d = 0; d < this.headDim; d++) {
        let value = 0;
        for (let j = 0; j < seqLen; j++) {
          value += scores[i * seqLen + j] / sum * v[j * this.headDim + d];
        }
        out[i * this.headDim + d] = value;
      }
    }
    return out;
  }
}

// Compatibility name for imports of the old example, not a multi-head API.
export { SingleHeadAttention as SimpleAttention };

export function runCheckedExample() {
  const q = Float32Array.from([1, 0, 1, 0]);
  const k = new Float32Array(q);
  const v = Float32Array.from([1, 0, 0, 1]);
  const result = new SingleHeadAttention(2).forward(q, k, v);
  // Analytic oracle: diagonal Q.K=1; off-diagonal Q.K=cos(1).
  const a = 1 / (1 + Math.exp((Math.cos(1) - 1) / Math.sqrt(2)));
  const expected = [a, 1 - a, 1 - a, a];
  for (let i = 0; i < result.length; i++) assert.ok(Math.abs(result[i] - expected[i]) <= 1e-6);
  return { example: 'single-head non-causal attention', checked: true, output: Array.from(result) };
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  console.log(JSON.stringify(runCheckedExample(), null, 2));
}
