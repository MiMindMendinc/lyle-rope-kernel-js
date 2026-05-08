#!/usr/bin/env node
/**
 * Simple throughput benchmark for lyle-rope-kernel
 * Reports pairs/sec (the key metric)
 */

import { applyRoPE } from './src/rope-kernel.js';
import { performance } from 'perf_hooks';

const headDims = [64, 128, 256];
const seqLens = [512, 2048, 8192];
const iterations = 50;

console.log('lyle-rope-kernel Benchmark');
console.log('==========================\n');

for (const headDim of headDims) {
  for (const seqLen of seqLens) {
    const tensor = new Float32Array(seqLen * headDim);
    for (let i = 0; i < tensor.length; i++) tensor[i] = Math.random() * 2 - 1;

    // Warmup
    for (let w = 0; w < 5; w++) {
      applyRoPE(new Float32Array(tensor), headDim);
    }

    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
      applyRoPE(new Float32Array(tensor), headDim); // fresh copy each time for fairness
    }
    const elapsed = performance.now() - start;

    const pairs = seqLen * (headDim / 2) * iterations;
    const pairsPerSec = (pairs / (elapsed / 1000)) / 1e6;

    console.log(`headDim=${headDim}  seqLen=${seqLen}  →  ${pairsPerSec.toFixed(1)} M pairs/sec  (${elapsed.toFixed(1)} ms total)`);
  }
}

console.log('\n✅ Benchmark complete. Expect 133–140 M pairs/sec on modern hardware.');