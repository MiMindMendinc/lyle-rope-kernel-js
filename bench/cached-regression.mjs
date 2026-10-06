import { performance } from 'node:perf_hooks';
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { applyRoPEWithPlan, createRoPEPlan } from '../src/rope-kernel.js';

function bestRate(plan, seed, headDim, seqLen) {
  let best = 0;
  for (let sample = 0; sample < 3; sample++) {
    const tensor = new Float32Array(seed);
    const start = performance.now();
    for (let iteration = 0; iteration < 30; iteration++) {
      applyRoPEWithPlan(tensor, plan, { startPos: iteration + 1, seqLen });
    }
    const elapsedSeconds = (performance.now() - start) / 1000;
    const pairs = seqLen * (headDim / 2) * 30;
    best = Math.max(best, pairs / elapsedSeconds);
  }
  return best;
}

describe('benchmark regression guard', () => {
  it('keeps the opt-in trig cache materially faster than the math path', () => {
    const headDim = 128;
    const seqLen = 512;
    const seed = new Float32Array(headDim * seqLen);
    for (let i = 0; i < seed.length; i++) seed[i] = Math.sin(i * 0.017);

    const mathPlan = createRoPEPlan(headDim);
    const cachedPlan = createRoPEPlan(headDim, 10000, { maxSeqLen: seqLen + 30 });

    // Warm both paths before timing so this guards the steady-state hot path.
    bestRate(mathPlan, seed, headDim, seqLen);
    bestRate(cachedPlan, seed, headDim, seqLen);

    const mathRate = bestRate(mathPlan, seed, headDim, seqLen);
    const cachedRate = bestRate(cachedPlan, seed, headDim, seqLen);
    assert.ok(
      cachedRate > mathRate * 1.5,
      `cached path regressed: ${cachedRate.toFixed(0)} pairs/sec vs ${mathRate.toFixed(0)} pairs/sec`,
    );
  });
});
