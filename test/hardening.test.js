import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  applyRoPE, applyRoPEWithPlan, applyRoPESplitHalf,
  applyRoPESplitHalfWithPlan, applyRoPEQK, applyToHead, createRoPEPlan,
} from '../src/rope-kernel.js';
import { referenceRoPE, maxAbsError, fixture } from '../support/reference.mjs';

const EPS = 1e-6; // Same absolute tolerance as the original short-position tests.
const singlePaths = [applyRoPEWithPlan, applyRoPESplitHalfWithPlan];

describe('numerical contract and input boundaries', () => {
  for (const layout of ['adjacent', 'split-half']) {
    it(`matches independent long-position reference: ${layout}`, () => {
      const apply = layout === 'adjacent' ? singlePaths[0] : singlePaths[1];
      for (const headDim of [2, 4, 8, 64, 96, 128, 256]) {
        for (const base of [1, 10000, 500000]) {
          const plans = [createRoPEPlan(headDim, base),
            createRoPEPlan(headDim, base, { maxSeqLen: 8193 })];
          for (const startPos of [0, 1, 17, 1024, 8191, 8192, 32768, 65536, 131069]) {
            const input = fixture(3 * headDim, startPos + headDim);
            const expected = referenceRoPE(input, headDim, { startPos, base, layout });
            for (const plan of plans) {
              const actual = new Float32Array(input);
              apply(actual, plan, { startPos });
              const error = maxAbsError(actual, expected);
              assert.ok(error <= EPS, JSON.stringify({ headDim, base, startPos,
                cached: plan.maxSeqLen > 0, error }));
            }
          }
        }
      }
    });

    it(`matches packed Q/K reference at high positions: ${layout}`, () => {
      for (const headDim of [64, 96, 128, 256]) {
        for (const base of [10000, 500000]) {
          const plan = createRoPEPlan(headDim, base, { maxSeqLen: 8193 });
          for (const startPos of [8191, 32768, 131069]) {
            const q = fixture(3 * 4 * headDim, 3);
            const k = fixture(3 * headDim, 5);
            const eq = referenceRoPE(q, headDim, { startPos, base, layout, heads: 4 });
            const ek = referenceRoPE(k, headDim, { startPos, base, layout });
            applyRoPEQK(q, k, plan, { qHeads: 4, kvHeads: 1, startPos, layout });
            assert.ok(maxAbsError(q, eq) <= EPS);
            assert.ok(maxAbsError(k, ek) <= EPS);
          }
        }
      }
    });
  }

  it('keeps consecutive large positions distinct', () => {
    const input = Float32Array.from([1, 0, 1, 0]);
    const expected = referenceRoPE(input, 2, { startPos: 131070 });
    applyRoPE(input, 2, { startPos: 131070 });
    assert.deepEqual(input, expected);
    assert.notDeepEqual(input.subarray(0, 2), input.subarray(2));
  });

  it('rejects unsafe positions and exclusive-end overflow without mutation', () => {
    const plan = createRoPEPlan(4);
    for (const apply of singlePaths) {
      for (const startPos of [2 ** 53, Number.MAX_SAFE_INTEGER, Infinity, NaN, -1, 1.5]) {
        const x = fixture(8), before = new Float32Array(x);
        assert.throws(() => apply(x, plan, { startPos }), RangeError);
        assert.deepEqual(x, before);
      }
    }
    for (const apply of [applyRoPE, applyRoPESplitHalf]) {
      const x = fixture(4), before = new Float32Array(x);
      assert.throws(() => apply(x, 4, { startPos: 2 ** 53 }), RangeError);
      assert.deepEqual(x, before);
    }
    assert.throws(() => applyToHead(fixture(4), 2 ** 53, 4), RangeError);
  });

  it('uses the same exclusive-end rule for single and packed paths', () => {
    const plan = createRoPEPlan(4);
    for (const apply of singlePaths) {
      assert.throws(() => apply(fixture(4), plan, {
        startPos: Number.MAX_SAFE_INTEGER, seqLen: 1,
      }), RangeError);
      assert.doesNotThrow(() => apply(new Float32Array(0), plan, {
        startPos: Number.MAX_SAFE_INTEGER, seqLen: 0,
      }));
    }
    assert.throws(() => applyRoPEQK(fixture(4), fixture(4), plan, {
      qHeads: 1, kvHeads: 1, startPos: Number.MAX_SAFE_INTEGER,
    }), RangeError);
    assert.doesNotThrow(() => applyRoPEQK(new Float32Array(0), new Float32Array(0), plan, {
      qHeads: 1, kvHeads: 1, startPos: Number.MAX_SAFE_INTEGER,
    }));
  });

  it('rejects incomplete rows instead of silently leaving an unrotated tail', () => {
    for (const apply of singlePaths) {
      for (const options of [{ startPos: 1 }, { startPos: 1, seqLen: 1 }]) {
        const x = fixture(5), before = new Float32Array(x);
        assert.throws(() => apply(x, createRoPEPlan(4), options), RangeError);
        assert.deepEqual(x, before);
      }
    }
  });

  it('rejects unsafe dimensions, sequence lengths and cache sizes', () => {
    assert.throws(() => createRoPEPlan(2 ** 53), RangeError);
    assert.throws(() => createRoPEPlan(4, 10000, { maxSeqLen: 2 ** 53 }), RangeError);
    for (const apply of singlePaths) {
      assert.throws(() => apply(fixture(4), createRoPEPlan(4), { seqLen: 2 ** 53 }), RangeError);
    }
    const q = fixture(4), k = fixture(4);
    assert.throws(() => applyRoPEQK(q, k, createRoPEPlan(4), {
      qHeads: Number.MAX_SAFE_INTEGER, kvHeads: 1,
    }), RangeError);
  });

  it('rejects zero-dimension fabricated plans', () => {
    const fake = { headDim: 0, halfDim: 0, invFreq: new Float32Array(0) };
    for (const apply of singlePaths) {
      const x = fixture(4), before = new Float32Array(x);
      assert.throws(() => apply(x, fake, { seqLen: 0 }), TypeError);
      assert.deepEqual(x, before);
    }
  });

  it('rejects null, array and primitive options consistently', () => {
    for (const options of [null, [], 7, 'x']) {
      assert.throws(() => createRoPEPlan(4, 10000, options), TypeError);
      assert.throws(() => applyRoPE(fixture(4), 4, options), TypeError);
      assert.throws(() => applyRoPESplitHalf(fixture(4), 4, options), TypeError);
      for (const apply of singlePaths) {
        assert.throws(() => apply(fixture(4), createRoPEPlan(4), options), TypeError);
      }
      assert.throws(() => applyRoPEQK(fixture(4), fixture(4), createRoPEPlan(4), options), TypeError);
    }
  });

  it('rejects detached plan storage before mutation', () => {
    const plan = createRoPEPlan(4);
    structuredClone(plan.invFreq.buffer, { transfer: [plan.invFreq.buffer] });
    const x = fixture(4), before = new Float32Array(x);
    assert.throws(() => applyRoPEWithPlan(x, plan, { startPos: 1 }), TypeError);
    assert.deepEqual(x, before);
  });

  it('retains partial-row-count operation and chunked continuation', () => {
    for (const apply of singlePaths) {
      const plan = createRoPEPlan(64, 10000, { maxSeqLen: 8193 });
      const full = fixture(4 * 64), chunked = new Float32Array(full);
      apply(full, plan, { startPos: 8191 });
      const tail = new Float32Array(chunked.subarray(128));
      apply(chunked, plan, { startPos: 8191, seqLen: 2 });
      assert.deepEqual(chunked.subarray(128), tail);
      apply(chunked.subarray(128), plan, { startPos: 8193 });
      assert.deepEqual(chunked, full);
    }
  });
});
