/**
 * Minimal Transformer Block using lyle-rope-kernel
 * Shows drop-in usage for real attention layers
 */

import { applyRoPE } from '../src/rope-kernel.js';

export class SimpleAttention {
  constructor(headDim = 128, nHeads = 8) {
    this.headDim = headDim;
    this.nHeads = nHeads;
    this.scale = 1 / Math.sqrt(headDim);
  }

  forward(q, k, v, startPos = 0) {
    const seqLen = q.length / this.headDim;
    
    // Apply RoPE to Q and K (in-place, zero alloc)
    applyRoPE(q, this.headDim, { startPos });
    applyRoPE(k, this.headDim, { startPos });

    // Simple scaled dot-product attention (for demo)
    const scores = new Float32Array(seqLen * seqLen);
    let idx = 0;
    
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < seqLen; j++) {
        let dot = 0;
        for (let d = 0; d < this.headDim; d++) {
          dot += q[i * this.headDim + d] * k[j * this.headDim + d];
        }
        scores[idx++] = dot * this.scale;
      }
    }

    // Softmax (naive)
    for (let i = 0; i < seqLen; i++) {
      let max = -Infinity;
      for (let j = 0; j < seqLen; j++) max = Math.max(max, scores[i * seqLen + j]);
      
      let sum = 0;
      for (let j = 0; j < seqLen; j++) {
        const exp = Math.exp(scores[i * seqLen + j] - max);
        scores[i * seqLen + j] = exp;
        sum += exp;
      }
      for (let j = 0; j < seqLen; j++) {
        scores[i * seqLen + j] /= sum;
      }
    }

    // Weighted sum with V
    const out = new Float32Array(seqLen * this.headDim);
    for (let i = 0; i < seqLen; i++) {
      for (let d = 0; d < this.headDim; d++) {
        let val = 0;
        for (let j = 0; j < seqLen; j++) {
          val += scores[i * seqLen + j] * v[j * this.headDim + d];
        }
        out[i * this.headDim + d] = val;
      }
    }

    return out;
  }
}

// Example usage
if (import.meta.url === `file://${process.argv[1]}`) {
  const attn = new SimpleAttention(128, 8);
  const seqLen = 256;
  const q = new Float32Array(seqLen * 128);
  const k = new Float32Array(seqLen * 128);
  const v = new Float32Array(seqLen * 128);
  
  // Fill with dummy data
  for (let i = 0; i < q.length; i++) {
    q[i] = k[i] = v[i] = Math.random() - 0.5;
  }
  
  console.log('Running minimal transformer attention with RoPE...');
  const out = attn.forward(q, k, v, 0);
  console.log('Output shape:', out.length, 'elements');
  console.log('Success! RoPE kernel integrated cleanly.');
}