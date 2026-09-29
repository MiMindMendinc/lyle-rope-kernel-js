import { applyRoPEQK, createRoPEPlan } from '../src/rope-kernel.js';

// A 4:1 grouped-query attention example: [tokens, heads, headDim].
const headDim = 64;
const qHeads = 4;
const kvHeads = 1;
const tokens = 2;
const q = Float32Array.from({ length: tokens * qHeads * headDim }, (_, i) => Math.sin(i));
const k = Float32Array.from({ length: tokens * kvHeads * headDim }, (_, i) => Math.cos(i));
const v = new Float32Array(k).fill(0.25); // Values are never rotated.
const plan = createRoPEPlan(headDim, 10000, { maxSeqLen: 128 });

applyRoPEQK(q, k, plan, { qHeads, kvHeads, startPos: 42 });
console.log({ qLength: q.length, kLength: k.length, vUnchanged: v.every(x => x === 0.25) });
