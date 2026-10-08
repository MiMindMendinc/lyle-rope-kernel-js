import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { applyRoPEQK, applyRoPESplitHalfWithPlan, applyRoPEWithPlan, createRoPEPlan } from '../src/rope-kernel.js';
import { fixture, referenceRoPE } from '../support/reference.mjs';

// The absolute 1e-6 tolerance used elsewhere only suits unit-scale fixtures: one Float32
// ulp at |x| ~ 1e12 is ~1.2e5. For scaled inputs use a combined absolute + relative bound:
//   |actual - reference| <= ATOL + RTOL * |reference|, RTOL = 4 Float32 ulps (relative).
const ATOL = 1e-6;
const RTOL = 4 * 2 ** -23;
const SCALES = [1e3, 1e6, 1e12, 1e30]; // 1e30 stays far below Float32 max (~3.4e38) after rotation.
const POSITIONS = [0, 1, 17, 1024, 8191, 8192, 32768, 65536, 131069];

function worstRatio(actual, expected) {
  assert.equal(actual.length, expected.length);
  let worst = 0;
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Number.isFinite(actual[i]) && Number.isFinite(expected[i]), `non-finite at ${i}`);
    worst = Math.max(worst, Math.abs(actual[i] - expected[i]) / (ATOL + RTOL * Math.abs(expected[i])));
  }
  return worst; // <= 1 means within tolerance
}

const scaled = (length, seed, scale) => fixture(length, seed).map(v => v * scale);

describe('scaled inputs (combined absolute + relative tolerance)', () => {
  for (const layout of ['adjacent', 'split-half']) {
    it(`single-tensor ${layout} stays within ${ATOL} + 4 ulp * |ref| up to |x| = 1e30`, () => {
      const apply = layout === 'adjacent' ? applyRoPEWithPlan : applyRoPESplitHalfWithPlan;
      for (const headDim of [8, 64, 128, 256]) {
        for (const base of [10000, 500000]) {
          const plans = [createRoPEPlan(headDim, base), createRoPEPlan(headDim, base, { maxSeqLen: 8193 })];
          for (const scale of SCALES) {
            for (const startPos of POSITIONS) {
              const input = scaled(3 * headDim, startPos + headDim, scale);
              const expected = referenceRoPE(input, headDim, { startPos, base, layout });
              for (const plan of plans) {
                const actual = apply(new Float32Array(input), plan, { startPos });
                const ratio = worstRatio(actual, expected);
                assert.ok(ratio <= 1, JSON.stringify({ layout, headDim, base, scale, startPos,
                  cached: plan.maxSeqLen > 0, ratio }));
              }
            }
          }
        }
      }
    });
  }

  it('packed Q/K stays within the same combined tolerance at large magnitudes', () => {
    const headDim = 64, qHeads = 4, kvHeads = 1, seqLen = 3;
    const plans = [createRoPEPlan(headDim, 500000), createRoPEPlan(headDim, 500000, { maxSeqLen: 8193 })];
    for (const layout of ['adjacent', 'split-half']) {
      for (const scale of SCALES) {
        for (const startPos of [1, 8191, 131069]) {
          const q = scaled(seqLen * qHeads * headDim, startPos + 1, scale);
          const k = scaled(seqLen * kvHeads * headDim, startPos + 2, scale);
          const qRef = referenceRoPE(q, headDim, { startPos, base: 500000, heads: qHeads, layout });
          const kRef = referenceRoPE(k, headDim, { startPos, base: 500000, heads: kvHeads, layout });
          for (const plan of plans) {
            const out = applyRoPEQK(new Float32Array(q), new Float32Array(k), plan, { qHeads, kvHeads, startPos, layout });
            const ratio = Math.max(worstRatio(out.q, qRef), worstRatio(out.k, kRef));
            assert.ok(ratio <= 1, JSON.stringify({ layout, scale, startPos, cached: plan.maxSeqLen > 0, ratio }));
          }
        }
      }
    }
  });
});
