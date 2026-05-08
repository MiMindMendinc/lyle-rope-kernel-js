import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { applyRoPE, applyToHead, verifyNormPreservation, DEFAULT_BASE } from '../src/rope-kernel.js';

describe('lyle-rope-kernel', () => {
  it('preserves L2 norm on random input (Llama-style)', () => {
    const headDim = 128;
    const seqLen = 64;
    const tensor = new Float32Array(seqLen * headDim);

    // Fill with random values in [-1, 1]
    for (let i = 0; i < tensor.length; i++) {
      tensor[i] = Math.random() * 2 - 1;
    }

    const copy = new Float32Array(tensor);
    applyRoPE(tensor, headDim, { startPos: 0 });

    assert.ok(verifyNormPreservation(copy, tensor), 'RoPE must preserve vector norm');
  });

  it('produces correct rotation for known small case (headDim=4, pos=0)', () => {
    const headDim = 4;
    const tensor = new Float32Array([1, 0, 0, 1]); // simple test vector
    const expected = new Float32Array([1, 0, 0, 1]); // at pos=0, theta=0 → identity

    applyRoPE(tensor, headDim, { startPos: 0, seqLen: 1 });

    for (let i = 0; i < 4; i++) {
      assert.ok(Math.abs(tensor[i] - expected[i]) < 1e-6, `index ${i}`);
    }
  });

  it('supports startPos for KV-cache continuation', () => {
    const headDim = 8;
    const tensor = new Float32Array(headDim).fill(0.5);
    const pos = 1024;

    applyRoPE(tensor, headDim, { startPos: pos, seqLen: 1 });

    // Just ensure it runs without error and changes values (non-zero rotation)
    let changed = false;
    for (let i = 0; i < headDim; i++) {
      if (Math.abs(tensor[i] - 0.5) > 1e-6) {
        changed = true;
        break;
      }
    }
    assert.ok(changed, 'RoPE with startPos should rotate the vector');
  });

  it('handles large sequences without precision collapse', () => {
    const headDim = 64;
    const seqLen = 4096;
    const tensor = new Float32Array(seqLen * headDim);

    for (let i = 0; i < tensor.length; i++) tensor[i] = (i % 17) / 17 - 0.5;

    const before = new Float32Array(tensor);
    applyRoPE(tensor, headDim, { startPos: 0 });

    assert.ok(verifyNormPreservation(before, tensor, 1e-4), 'Norm should be preserved even on long sequences');
  });
});