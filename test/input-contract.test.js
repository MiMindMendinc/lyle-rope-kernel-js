import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { runInNewContext } from 'node:vm';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
  applyRoPE, applyRoPEQK, applyRoPESplitHalf, applyRoPESplitHalfWithPlan, applyRoPEWithPlan,
  applyToHead, createRoPEPlan, verifyNormPreservation,
} from '../src/rope-kernel.js';
import { fixture, maxAbsError, referenceRoPE } from '../support/reference.mjs';
import { plainScalarRoPEInPlace } from '../bench/baseline.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const plan = createRoPEPlan(8);
const cachedPlan = createRoPEPlan(8, 10000, { maxSeqLen: 16 });
const QK = { qHeads: 1, kvHeads: 1 };

describe('tensor type, empty-tensor and non-finite-value behavior (current semantics)', () => {
  it('rejects every non-Float32Array tensor type with a TypeError and leaves it unchanged', () => {
    const wrong = [Float64Array, Int32Array, Uint8Array, Uint8ClampedArray, Int16Array, BigInt64Array];
    if (typeof globalThis.Float16Array === 'function') wrong.push(globalThis.Float16Array);
    for (const Type of wrong) {
      const make = () => (Type === BigInt64Array ? new Type(8).fill(1n) : new Type(8).fill(1));
      const t = make();
      const before = Array.from(t);
      assert.throws(() => applyRoPEWithPlan(t, plan, { startPos: 3 }), { name: 'TypeError', message: /tensor must be a Float32Array/ }, Type.name);
      assert.throws(() => applyRoPESplitHalfWithPlan(t, plan, { startPos: 3 }), TypeError, Type.name);
      assert.throws(() => applyRoPE(t, 8, { startPos: 3 }), TypeError, Type.name);
      assert.throws(() => applyRoPESplitHalf(t, 8, { startPos: 3 }), TypeError, Type.name);
      assert.throws(() => applyToHead(t, 3, 8), { name: 'TypeError', message: /head must be a Float32Array/ }, Type.name);
      assert.throws(() => applyRoPEQK(t, new Float32Array(8), plan, { ...QK, startPos: 3 }), { name: 'TypeError', message: /q must be a Float32Array/ });
      assert.throws(() => applyRoPEQK(new Float32Array(8), t, plan, { ...QK, startPos: 3 }), { name: 'TypeError', message: /k must be a Float32Array/ });
      assert.deepEqual(Array.from(t), before, `${Type.name} was modified`);
    }
    assert.throws(() => applyRoPEWithPlan(new DataView(new ArrayBuffer(32)), plan), TypeError);
    assert.throws(() => applyRoPEWithPlan(new ArrayBuffer(32), plan), TypeError);
  });

  it('documents that a Float32Array from another realm (vm context) is rejected', () => {
    // The check is `instanceof Float32Array`, so cross-realm arrays (e.g. from node:vm or an
    // iframe) are rejected. Pinned so any change to this behavior is deliberate.
    const foreign = runInNewContext('new Float32Array(8)');
    assert.equal(Object.prototype.toString.call(foreign), '[object Float32Array]');
    assert.throws(() => applyRoPEWithPlan(foreign, plan), TypeError);
  });

  it('treats empty tensors as no-ops that return the same object', () => {
    for (const p of [plan, cachedPlan]) {
      for (const apply of [applyRoPEWithPlan, applyRoPESplitHalfWithPlan]) {
        const empty = new Float32Array(0);
        assert.equal(apply(empty, p), empty);
        assert.equal(apply(empty, p, { startPos: 1000 }), empty);
        assert.equal(apply(empty, p, { seqLen: 0 }), empty);
        assert.throws(() => apply(empty, p, { seqLen: 1 }), RangeError);
      }
      const q = new Float32Array(0), k = new Float32Array(0);
      const out = applyRoPEQK(q, k, p, { qHeads: 4, kvHeads: 2, startPos: 7 });
      assert.equal(out.q, q);
      assert.equal(out.k, k);
      // Empty Q with non-empty K is a shape mismatch, not a no-op.
      assert.throws(() => applyRoPEQK(new Float32Array(0), new Float32Array(8), p, QK), RangeError);
    }
    assert.equal(applyRoPE(new Float32Array(0), 64).length, 0);
    assert.equal(applyRoPESplitHalf(new Float32Array(0), 64).length, 0);
    // applyToHead needs exactly one head, so an empty tensor is rejected.
    assert.throws(() => applyToHead(new Float32Array(0), 1, 8), RangeError);
    assert.equal(verifyNormPreservation(new Float32Array(0), new Float32Array(0)), true);
  });

  for (const [name, apply, pairOf] of [
    ['adjacent', applyRoPEWithPlan, i => [i - (i % 2), i - (i % 2) + 1]],
    ['split-half', applyRoPESplitHalfWithPlan, i => [i % 4, (i % 4) + 4]],
  ]) {
    it(`does not scan values: NaN/Infinity stay within their rotated pair (${name})`, () => {
      // Values are not validated (docs/NUMERICS.md). A non-finite input element makes both
      // elements of its pair non-finite at positions > 0; every other element is unaffected.
      for (const p of [plan, cachedPlan]) {
        for (const bad of [NaN, Infinity, -Infinity]) {
          const input = fixture(16, 5);
          const corrupted = new Float32Array(input);
          corrupted[2] = bad; // token 0, so the pair is unrotated at startPos 0
          corrupted[8 + 2] = bad; // token 1
          const clean = referenceRoPE(input, 8, { startPos: 3, layout: name });
          const out = apply(new Float32Array(corrupted), p, { startPos: 3 });
          const badSet = new Set([...pairOf(2), ...pairOf(2).map(i => i + 8)]);
          for (let i = 0; i < 16; i++) {
            if (badSet.has(i)) {
              assert.ok(!Number.isFinite(out[i]), `${name} ${bad} index ${i} should be non-finite`);
              if (Number.isNaN(bad)) assert.ok(Number.isNaN(out[i]));
            } else {
              assert.equal(out[i], clean[i], `${name} ${bad} index ${i} changed`);
            }
          }
          // Position 0 is the identity and is skipped: only the bad element itself is non-finite.
          const atZero = apply(new Float32Array(corrupted), p, { startPos: 0, seqLen: 1 });
          for (let i = 0; i < 8; i++) {
            if (i === 2) assert.ok(Object.is(atZero[i], bad) || (Number.isNaN(bad) && Number.isNaN(atZero[i])));
            else assert.equal(atZero[i], corrupted[i]);
          }
        }
      }
    });
  }

  it('lets finite inputs near Float32 max overflow to Infinity (outside the contract, not checked)', () => {
    const t = Float32Array.from([3e38, 3e38, 0, 0, 0, 0, 0, 0]);
    applyRoPEWithPlan(t, plan, { startPos: 1 });
    assert.ok(t.some(v => v === Infinity || v === -Infinity));
    assert.ok(t.every(v => !Number.isNaN(v)));
  });
});

describe('benchmark hygiene', () => {
  it('the bench baseline is an in-place scalar loop that matches the reference', () => {
    for (const headDim of [8, 64, 128]) {
      const input = fixture(33 * headDim, headDim);
      const x = new Float32Array(input);
      const out = plainScalarRoPEInPlace(x, headDim, 10000, new Float64Array(headDim / 2));
      assert.equal(out, x);
      assert.ok(maxAbsError(out, referenceRoPE(input, headDim)) <= 1e-6);
    }
  });

  it('keeps the wall-clock timing guard out of npm test and every CI workflow', () => {
    const pkg = JSON.parse(readFileSync(root + 'package.json', 'utf8'));
    for (const name of ['test', 'verify', 'test:package']) {
      assert.doesNotMatch(pkg.scripts[name], /bench\/|test:performance|cached-regression/, name);
    }
    // node --test's default patterns match test/**, *.test.*, *-test.*, *_test.*, test-*.*,
    // test.*; bench/cached-regression.mjs matches none of them.
    assert.doesNotMatch('bench/cached-regression.mjs', /(^|\/)test\/|[.\-_]test\.m?js$|(^|\/)test-[^/]*$|(^|\/)test\.m?js$/);
    for (const file of readdirSync(root + '.github/workflows')) {
      const text = readFileSync(root + '.github/workflows/' + file, 'utf8');
      assert.doesNotMatch(text, /test:performance|cached-regression|npm run bench/, file);
    }
  });
});
