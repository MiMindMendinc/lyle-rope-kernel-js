import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { applyRoPE, applyRoPEWithPlan, applyRoPESplitHalf, applyRoPESplitHalfWithPlan, applyRoPEQK, applyToHead, createRoPEPlan, verifyNormPreservation, DEFAULT_BASE } from '../src/rope-kernel.js';

const EPS = 1e-6;
function close(a, b, msg) { assert.ok(Math.abs(a - b) <= EPS, msg + ': ' + a + ' != ' + b); }
function ref(input, headDim, opts = {}) {
  const startPos = opts.startPos ?? 0;
  const seqLen = opts.seqLen ?? Math.floor(input.length / headDim);
  const base = opts.base ?? DEFAULT_BASE;
  const out = new Float32Array(input);
  const half = headDim >>> 1;
  for (let p = 0; p < seqLen; p++) {
    for (let i = 0; i < half; i++) {
      const theta = (startPos + p) / Math.pow(base, (2 * i) / headDim);
      const c = Math.cos(theta), s = Math.sin(theta);
      const j = p * headDim + i * 2;
      const x0 = input[j], x1 = input[j + 1];
      out[j] = x0 * c - x1 * s;
      out[j + 1] = x0 * s + x1 * c;
    }
  }
  return out;
}

function refSplitHalf(input, headDim, opts = {}) {
  const startPos = opts.startPos ?? 0;
  const seqLen = opts.seqLen ?? Math.floor(input.length / headDim);
  const base = opts.base ?? DEFAULT_BASE;
  const out = new Float32Array(input);
  const half = headDim >>> 1;
  for (let p = 0; p < seqLen; p++) {
    for (let i = 0; i < half; i++) {
      const theta = (startPos + p) / Math.pow(base, (2 * i) / headDim);
      const c = Math.cos(theta), s = Math.sin(theta);
      const j0 = p * headDim + i;
      const j1 = j0 + half;
      const x0 = input[j0], x1 = input[j1];
      out[j0] = x0 * c - x1 * s;
      out[j1] = x0 * s + x1 * c;
    }
  }
  return out;
}

describe('lyle-rope-kernel', () => {
  it('exports the default base', () => assert.equal(DEFAULT_BASE, 10000));

  it('matches known values for headDim=4 pos=1', () => {
    const x = new Float32Array([1, 2, 3, 4]);
    const y = new Float32Array([-1.1426396637476532, 1.922075596544176, 2.9598506689133295, 4.029799500837498]);
    applyRoPE(x, 4, { startPos: 1, seqLen: 1 });
    for (let i = 0; i < x.length; i++) close(x[i], y[i], 'known ' + i);
  });

  it('matches a scalar reference', () => {
    const headDim = 8, seqLen = 5, startPos = 17;
    const x = new Float32Array(seqLen * headDim);
    for (let i = 0; i < x.length; i++) x[i] = ((i % 11) - 5) / 7;
    const y = ref(x, headDim, { startPos, seqLen });
    const z = new Float32Array(x);
    applyRoPE(z, headDim, { startPos, seqLen });
    for (let i = 0; i < z.length; i++) close(z[i], y[i], 'ref ' + i);
  });

  it('preserves L2 norm', () => {
    const headDim = 128, seqLen = 64;
    const x = new Float32Array(seqLen * headDim);
    for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.017) * 0.75;
    const before = new Float32Array(x);
    applyRoPE(x, headDim);
    assert.ok(verifyNormPreservation(before, x));
  });

  it('keeps position zero as identity', () => {
    const x = new Float32Array([1, 0, 0, 1]);
    applyRoPE(x, 4, { startPos: 0, seqLen: 1 });
    assert.deepEqual(Array.from(x), [1, 0, 0, 1]);
  });

  it('supports KV-cache continuation startPos', () => {
    const x = new Float32Array(8).fill(0.5);
    applyRoPE(x, 8, { startPos: 1024, seqLen: 1 });
    assert.ok(x.some(v => Math.abs(v - 0.5) > EPS));
  });

  it('rotates only requested partial seqLen', () => {
    const x = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const tail = Array.from(x.slice(4));
    applyRoPE(x, 4, { startPos: 1, seqLen: 1 });
    assert.deepEqual(Array.from(x.slice(4)), tail);
  });

  it('applyToHead matches applyRoPE', () => {
    const a = new Float32Array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]);
    const b = new Float32Array(a);
    applyToHead(a, 33, 8);
    applyRoPE(b, 8, { startPos: 33, seqLen: 1 });
    for (let i = 0; i < a.length; i++) close(a[i], b[i], 'head ' + i);
  });

  it('matches split-half scalar reference', () => {
    const headDim = 8, seqLen = 3, startPos = 11;
    const x = new Float32Array(seqLen * headDim);
    for (let i = 0; i < x.length; i++) x[i] = ((i % 9) - 4) / 5;
    const y = refSplitHalf(x, headDim, { startPos, seqLen });
    const z = new Float32Array(x);
    applyRoPESplitHalf(z, headDim, { startPos, seqLen });
    for (let i = 0; i < z.length; i++) close(z[i], y[i], 'split-half ' + i);
  });

  it('split-half layout differs from adjacent-pair layout away from zero', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const b = new Float32Array(a);
    applyRoPESplitHalf(a, 8, { startPos: 7, seqLen: 1 });
    applyRoPE(b, 8, { startPos: 7, seqLen: 1 });
    assert.ok(a.some((v, i) => Math.abs(v - b[i]) > 1e-5));
  });

  it('planned API matches direct API', () => {
    const x = new Float32Array(9 * 16);
    for (let i = 0; i < x.length; i++) x[i] = Math.sin(i / 13) * 0.5;
    const a = new Float32Array(x), b = new Float32Array(x);
    applyRoPEWithPlan(a, createRoPEPlan(16, 500000), { startPos: 2048, seqLen: 9 });
    applyRoPE(b, 16, { startPos: 2048, seqLen: 9, base: 500000 });
    for (let i = 0; i < a.length; i++) close(a[i], b[i], 'plan ' + i);
  });

  it('uses a cached plan without changing results and falls back beyond its range', () => {
    const headDim = 16, seqLen = 6, startPos = 3;
    const x = new Float32Array(seqLen * headDim);
    for (let i = 0; i < x.length; i++) x[i] = Math.sin(i / 7) * 0.5;

    const cached = new Float32Array(x);
    const direct = new Float32Array(x);
    const plan = createRoPEPlan(headDim, DEFAULT_BASE, { maxSeqLen: 5 });
    applyRoPEWithPlan(cached, plan, { startPos, seqLen });
    applyRoPE(direct, headDim, { startPos, seqLen });

    for (let i = 0; i < cached.length; i++) close(cached[i], direct[i], 'cached ' + i);

    const splitCached = new Float32Array(x);
    const splitDirect = new Float32Array(x);
    applyRoPESplitHalfWithPlan(splitCached, plan, { startPos, seqLen });
    applyRoPESplitHalf(splitDirect, headDim, { startPos, seqLen });
    for (let i = 0; i < splitCached.length; i++) close(splitCached[i], splitDirect[i], 'cached split ' + i);
  });

  it('matches direct RoPE across cache boundaries and supported dimensions', () => {
    for (const headDim of [2, 4, 8, 64]) {
      for (const base of [10000, 500000]) {
        const seqLen = 4;
        const input = new Float32Array(seqLen * headDim);
        for (let i = 0; i < input.length; i++) input[i] = Math.sin(i * 0.13) * 0.75;
        const plan = createRoPEPlan(headDim, base, { maxSeqLen: 5 });

        for (const startPos of [0, 1, 4, 5]) {
          const cached = new Float32Array(input);
          const direct = new Float32Array(input);
          applyRoPEWithPlan(cached, plan, { startPos, seqLen });
          applyRoPE(direct, headDim, { base, startPos, seqLen });
          for (let i = 0; i < cached.length; i++) {
            close(cached[i], direct[i], `cache boundary dim=${headDim} base=${base} pos=${startPos} i=${i}`);
          }
        }
      }
    }
  });

  it('custom base changes output', () => {
    const a = new Float32Array([1,2,3,4,5,6,7,8]);
    const b = new Float32Array(a);
    applyRoPE(a, 8, { startPos: 13, seqLen: 1, base: 10000 });
    applyRoPE(b, 8, { startPos: 13, seqLen: 1, base: 500000 });
    assert.ok(a.some((v, i) => Math.abs(v - b[i]) > 1e-5));
  });

  it('handles long sequences', () => {
    const x = new Float32Array(4096 * 64);
    for (let i = 0; i < x.length; i++) x[i] = (i % 17) / 17 - 0.5;
    const before = new Float32Array(x);
    applyRoPE(x, 64);
    assert.ok(verifyNormPreservation(before, x, 1e-4));
  });

  it('rejects invalid inputs', () => {
    assert.throws(() => applyRoPE([1,2,3,4], 4), /Float32Array/);
    assert.throws(() => applyRoPE(new Float32Array(9), 3), /positive even integer/);
    assert.throws(() => applyRoPE(new Float32Array(4), 0), /positive even integer/);
    assert.throws(() => applyRoPE(new Float32Array(4), 4, { startPos: -1 }), /startPos/);
    assert.throws(() => applyRoPE(new Float32Array(4), 4, { seqLen: 2 }), /exceeds tensor length/);
    assert.throws(() => createRoPEPlan(4, 0), /base/);
    assert.throws(() => createRoPEPlan(4, 10000, { maxSeqLen: -1 }), /maxSeqLen/);
    assert.throws(() => createRoPEPlan(4, 10000, { maxSeqLen: 1.5 }), /maxSeqLen/);
  });
});

describe('packed Q/K RoPE', () => {
  for (const layout of ['adjacent', 'split-half']) {
    it(`matches independent per-head reference for ${layout}, including cache boundary`, () => {
      const headDim = 8, qHeads = 4, kvHeads = 2, seqLen = 5, startPos = 3;
      const q = Float32Array.from({ length: seqLen * qHeads * headDim }, (_, i) => Math.sin(i / 9));
      const k = Float32Array.from({ length: seqLen * kvHeads * headDim }, (_, i) => Math.cos(i / 11));
      const expectedQ = new Float32Array(q), expectedK = new Float32Array(k);
      const scalar = layout === 'adjacent' ? ref : refSplitHalf;
      for (const [tensor, heads] of [[expectedQ, qHeads], [expectedK, kvHeads]]) {
        for (let pos = 0; pos < seqLen; pos++) {
          for (let head = 0; head < heads; head++) {
            const offset = (pos * heads + head) * headDim;
            tensor.set(scalar(tensor.subarray(offset, offset + headDim), headDim, {
              startPos: startPos + pos, seqLen: 1,
            }), offset);
          }
        }
      }
      const returned = applyRoPEQK(q, k, createRoPEPlan(headDim, DEFAULT_BASE, { maxSeqLen: 5 }), {
        qHeads, kvHeads, seqLen, startPos, layout,
      });
      assert.equal(returned.q, q);
      assert.equal(returned.k, k);
      for (let i = 0; i < q.length; i++) close(q[i], expectedQ[i], `Q ${i}`);
      for (let i = 0; i < k.length; i++) close(k[i], expectedK[i], `K ${i}`);
    });
  }

  it('keeps token zero as identity and accepts nonoverlapping views of one buffer', () => {
    const buffer = new ArrayBuffer(16 * Float32Array.BYTES_PER_ELEMENT);
    const q = new Float32Array(buffer, 0, 8).fill(0.5);
    const k = new Float32Array(buffer, 8 * Float32Array.BYTES_PER_ELEMENT, 8).fill(0.25);
    applyRoPEQK(q, k, createRoPEPlan(4), { qHeads: 2, kvHeads: 2 });
    assert.deepEqual(Array.from(q), Array(8).fill(0.5));
    assert.deepEqual(Array.from(k), Array(8).fill(0.25));
  });

  it('rejects bad shapes and overlapping views before modifying either input', () => {
    const plan = createRoPEPlan(4);
    const q = new Float32Array(8).fill(1), k = new Float32Array(4).fill(2);
    const opts = { qHeads: 2, kvHeads: 1 };
    assert.throws(() => applyRoPEQK(q, k, plan, { ...opts, seqLen: 2 }), /lengths/);
    assert.throws(() => applyRoPEQK(q, k, plan, { ...opts, layout: 'other' }), /layout/);
    assert.throws(() => applyRoPEQK(q, k, plan, { ...opts, qHeads: 0 }), /qHeads/);
    assert.throws(() => applyRoPEQK(q, k, plan, { ...opts, startPos: Number.MAX_SAFE_INTEGER }), /lengths/);
    const shared = new Float32Array(12);
    assert.throws(() => applyRoPEQK(shared.subarray(0, 8), shared.subarray(4, 8), plan, opts), /overlap/);
    assert.deepEqual(Array.from(q), Array(8).fill(1));
    assert.deepEqual(Array.from(k), Array(4).fill(2));
  });
});
