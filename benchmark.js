#!/usr/bin/env node
import * as rope from './src/rope-kernel.js';
import { performance } from 'node:perf_hooks';

const usePlan = process.argv.includes('--mode=hot');
const headDims = [64, 128, 256];
const seqLens = [512, 2048, 8192];
const iterations = usePlan ? 100 : 50;

console.log('lyle-rope-kernel Benchmark');
console.log('Mode: ' + (usePlan ? 'precomputed plan' : 'copy plus compute'));
console.log('Node: ' + process.version);
console.log('Iterations: ' + iterations + '\n');

for (const headDim of headDims) {
  const plan = rope.createRoPEPlan(headDim);
  for (const seqLen of seqLens) {
    const tensor = new Float32Array(seqLen * headDim);
    for (let i = 0; i < tensor.length; i++) tensor[i] = Math.random() * 2 - 1;
    const start = performance.now();
    if (usePlan) {
      for (let i = 0; i < iterations; i++) rope.applyRoPEWithPlan(tensor, plan, { startPos: i, seqLen });
    } else {
      for (let i = 0; i < iterations; i++) rope.applyRoPE(new Float32Array(tensor), headDim);
    }
    const elapsed = performance.now() - start;
    const pairs = seqLen * (headDim / 2) * iterations;
    const rate = (pairs / (elapsed / 1000)) / 1e6;
    console.log('headDim=' + headDim + ' seqLen=' + seqLen + ' -> ' + rate.toFixed(1) + ' M pairs/sec');
  }
}
