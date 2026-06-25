#!/usr/bin/env node
import { applyRoPE } from './src/rope-kernel.js';
import { performance } from 'node:perf_hooks';
import os from 'node:os';

const headDims = [64, 128, 256];
const seqLens = [512, 2048, 8192];
const iterations = 50;

console.log('lyle-rope-kernel Benchmark: copy + compute');
console.log('===========================================');
console.log('Node: ' + process.version);
console.log('Platform: ' + process.platform + ' ' + process.arch);
console.log('CPU: ' + (os.cpus()[0]?.model ?? 'unknown'));
console.log('Iterations: ' + iterations + '\n');

for (const headDim of headDims) {
  for (const seqLen of seqLens) {
    const tensor = new Float32Array(seqLen * headDim);
    for (let i = 0; i < tensor.length; i++) tensor[i] = Math.random() * 2 - 1;

    for (let w = 0; w < 5; w++) {
      applyRoPE(new Float32Array(tensor), headDim);
    }

    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
      applyRoPE(new Float32Array(tensor), headDim);
    }
    const elapsed = performance.now() - start;

    const pairs = seqLen * (headDim / 2) * iterations;
    const pairsPerSec = (pairs / (elapsed / 1000)) / 1e6;

    console.log(
      'headDim=' + headDim.toString().padStart(3) +
      '  seqLen=' + seqLen.toString().padStart(5) +
      '  -> ' + pairsPerSec.toFixed(1).padStart(6) +
      ' M pairs/sec  (' + elapsed.toFixed(1) + ' ms total)'
    );
  }
}

console.log('\nNote: this benchmark includes Float32Array copy cost.');
console.log('Run `npm run benchmark:hot` for precomputed-plan in-place timing.');
