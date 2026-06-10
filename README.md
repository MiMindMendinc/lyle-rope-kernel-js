# lyle-rope-kernel-js

**Zero-dependency, in-place RoPE (Rotary Position Embedding) kernel in pure JavaScript — bit-exact against the Llama reference, with a reproducible benchmark and live interactive demo.**

[![Tests](https://img.shields.io/badge/Tests-4%2F4_passing-brightgreen.svg)](test/rope.test.js)
[![Benchmark](https://img.shields.io/badge/Benchmark-reproducible_(benchmark.js)-blue.svg)](benchmark.js)
[![Zero deps](https://img.shields.io/badge/Dependencies-0-success.svg)](package.json)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[▶ Open the live demo](demo/index.html)** — sliders for `headDim`/`seqLen`/`nHeads`, real-time before/after tensor view, attention-score visualization, instant throughput readout.

---

## Why this exists

RoPE is the positional encoding inside every modern LLM (Llama, Mistral, Gemma, Qwen). Most JavaScript implementations allocate on every call, break on KV-cache continuation (`startPos > 0`), or drift from the reference math. This kernel is built the way you'd build it for a real inference hot loop:

- **True in-place, zero-allocation hot path** — near-zero GC pressure in autoregressive decoding
- **Full `startPos` support** for KV-cache continuation
- **Bit-exact** against official Llama reference rotations (float32 tolerance), norm preservation verified
- Hand-tuned `Float32Array` access patterns, `applyToHead` helper, full JSDoc

## Performance — measured, reproducible

Run it yourself: `npm run benchmark` (harness: [`benchmark.js`](benchmark.js))

| Hardware | Throughput |
| --- | --- |
| Modern desktop CPU | ~133–140M rotation pairs/sec |
| Shared cloud vCPU (independent re-run) | ~31–42M pairs/sec |

Numbers vary with `headDim`/`seqLen`; the harness sweeps Llama-3-8B/70B, 128-dim (GPT-style), and 64-dim (mobile) configs. No cross-library comparison is claimed here — run the harness against your own baseline.

## Correctness

`npm test` — 4/4 passing:
- Llama reference parity (bit-exact within float32 tolerance)
- Norm preservation (required for attention stability)
- Edge cases: KV-cache continuation, seqLen > 2048, startPos > 0, odd headDim

## Install & use

```bash
npm install lyle-rope-kernel
```

```js
import { applyRoPE } from 'lyle-rope-kernel';

const headDim = 128;
const seqLen = 512;
const q = new Float32Array(seqLen * headDim);
const k = new Float32Array(seqLen * headDim);

applyRoPE(q, headDim, { startPos: 0 });        // in-place, fastest path
applyRoPE(k, headDim, { startPos: 0 });

// KV-cache continuation
applyRoPE(q, headDim, { startPos: 1024, seqLen: 8 });
```

Full drop-in attention block: [`examples/minimal-transformer.js`](examples/minimal-transformer.js)

## Development

```bash
git clone https://github.com/MiMindMendinc/lyle-rope-kernel-js.git
cd lyle-rope-kernel-js
npm install
npm test           # correctness suite
npm run benchmark  # throughput + memory profiling
```

## Roadmap

- [x] v1.0.0 — Core kernel, tests, benchmark, interactive demo
- [ ] v1.1.0 — WebGPU compute-shader path ([`src/webgpu-rope.js`](src/webgpu-rope.js))
- [ ] v1.2.0 — Transformer-block integration package
- [ ] v2.0.0 — Multi-head batching + tiled attention experiments

## Related work

Part of a privacy-first, offline-capable AI stack by [Michigan MindMend Inc.](https://github.com/MiMindMendinc) — see [DominusUltra](https://github.com/MiMindMendinc/DominusUltra) (Triton fused RoPE + causal attention) and [TrustLayer](https://github.com/MiMindMendinc/TrustLayer) (LLM safety gateway).

## License

MIT
