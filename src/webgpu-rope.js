import { applyRoPE } from './rope-kernel.js';

export const WEBGPU_ROPE_STATUS = 'preview-fallback';
export const WGSL_ROPE_SHADER = '';

export function applyRoPEWebGPU(tensor, headDim, options = {}) {
  return applyRoPE(tensor, headDim, options);
}
