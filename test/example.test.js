import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { SingleHeadAttention, SimpleAttention, runCheckedExample } from '../examples/minimal-transformer.js';
import { referenceRoPE, fixture, maxAbsError } from '../support/reference.mjs';

function referenceAttention(q, k, v, dim, startPos) {
  const rq = referenceRoPE(q, dim, { startPos });
  const rk = referenceRoPE(k, dim, { startPos });
  const n = q.length / dim;
  const output = new Float32Array(q.length);
  for (let token = 0; token < n; token++) {
    const weights = Array.from({ length: n }, (_, other) => {
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += rq[token * dim + d] * rk[other * dim + d];
      return Math.exp(dot / Math.sqrt(dim)); // Bounded fixtures, independent small oracle.
    });
    const total = weights.reduce((a, b) => a + b, 0);
    for (let d = 0; d < dim; d++) {
      output[token * dim + d] = weights.reduce((sum, w, other) => sum + w / total * v[other * dim + d], 0);
    }
  }
  return output;
}

describe('single-head educational example', () => {
  it('checks its output against an analytic two-token answer', () => {
    assert.equal(runCheckedExample().checked, true);
  });
  it('matches an independent attention reference and preserves all inputs', () => {
    for (const dim of [2, 8, 64]) {
      for (const startPos of [0, 8192, 131069]) {
        const q = fixture(3 * dim, 1), k = fixture(3 * dim, 2), v = fixture(3 * dim, 3);
        const before = [new Float32Array(q), new Float32Array(k), new Float32Array(v)];
        const expected = referenceAttention(q, k, v, dim, startPos);
        const actual = new SingleHeadAttention(dim).forward(q, k, v, startPos);
        assert.ok(maxAbsError(actual, expected) <= 1e-6);
        assert.deepEqual([q, k, v], before);
      }
    }
  });
  it('rejects the formerly ignored nHeads parameter', () => {
    assert.equal(SimpleAttention, SingleHeadAttention);
    assert.throws(() => new SingleHeadAttention(8, 8), /one head/);
  });
  it('validates shapes, values and position before touching caller data', () => {
    const model = new SingleHeadAttention(4);
    const q = fixture(8), k = fixture(8), v = fixture(8);
    const before = [new Float32Array(q), new Float32Array(k), new Float32Array(v)];
    assert.throws(() => model.forward(q, k.subarray(0, 4), v), RangeError);
    assert.throws(() => model.forward(q.subarray(0, 5), k.subarray(0, 5), v.subarray(0, 5)), RangeError);
    assert.throws(() => model.forward(q, k, []), TypeError);
    assert.throws(() => model.forward(q, k, v, 2 ** 53), RangeError);
    assert.throws(() => model.forward(q, k, new Float32Array(8).fill(NaN)), RangeError);
    assert.deepEqual([q, k, v], before);
  });
  it('handles an empty sequence explicitly', () => {
    const empty = new Float32Array(0);
    assert.deepEqual(new SingleHeadAttention(4).forward(empty, empty, empty), empty);
  });
});
