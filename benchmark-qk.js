#!/usr/bin/env node
import { performance } from 'node:perf_hooks';
import { applyRoPEQK, applyRoPEWithPlan, createRoPEPlan } from './src/rope-kernel.js';

const headDim = 128, qHeads = 32, kvHeads = 8, seqLen = 128, iterations = 100;
const qInput = Float32Array.from({ length: seqLen * qHeads * headDim }, (_, i) => Math.sin(i));
const kInput = Float32Array.from({ length: seqLen * kvHeads * headDim }, (_, i) => Math.cos(i));
const plan = createRoPEPlan(headDim);

function measure(run) {
  for (let i = 0; i < 10; i++) run();
  const start = performance.now();
  for (let i = 0; i < iterations; i++) run();
  return (performance.now() - start) / iterations;
}

function naivePerHead() {
  const q = new Float32Array(qInput), k = new Float32Array(kInput);
  for (const [tensor, heads] of [[q, qHeads], [k, kvHeads]]) {
    for (let pos = 0; pos < seqLen; pos++) {
      for (let head = 0; head < heads; head++) {
        const offset = (pos * heads + head) * headDim;
        applyRoPEWithPlan(tensor.subarray(offset, offset + headDim), plan, {
          startPos: 1024 + pos, seqLen: 1,
        });
      }
    }
  }
}

function packedQK() {
  const q = new Float32Array(qInput), k = new Float32Array(kInput);
  applyRoPEQK(q, k, plan, { qHeads, kvHeads, startPos: 1024 });
}

console.log(`Node ${process.version}; ${iterations} timed runs after 10 warmups`);
console.log(`headDim=${headDim}, seqLen=${seqLen}, qHeads=${qHeads}, kvHeads=${kvHeads}`);
console.log('Both modes include input copies; neither includes plan construction.');
console.log(`Per-head calls: ${measure(naivePerHead).toFixed(3)} ms/run`);
console.log(`Packed Q/K:     ${measure(packedQK).toFixed(3)} ms/run`);
