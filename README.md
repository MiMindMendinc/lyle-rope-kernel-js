# lyle-rope-kernel-js

> Zero-dependency, in-place Rotary Position Embedding kernel for JavaScript inference experiments.

![Tests](https://img.shields.io/badge/tests-17%2F17%20passing-brightgreen)
![Runtime](https://img.shields.io/badge/runtime-Node%2020%2B-blue)
![Dependencies](https://img.shields.io/badge/dependencies-0-success)
![Module](https://img.shields.io/badge/module-ESM-purple)
![License](https://img.shields.io/badge/license-MIT-yellow)
![Status](https://img.shields.io/badge/status-verified-success)

## Plaques

| Plaque | Status |
| --- | --- |
| Correctness Gate | 15/15 core correctness tests; 17/17 total suite |
| Reference Gate | deterministic scalar reference parity |
| Stability Gate | L2 norm preservation checked |
| KV Cache Gate | startPos continuation checked |
| Hot Path Gate | reusable frequency and optional trig-cache plans |
| Dependency Gate | zero runtime dependencies |
| Claim Hygiene Gate | no unsupported fastest or fake benchmark claims |
| WebGPU Gate | preview fallback only |

## What is verified

- In-place Float32Array rotation.
- Reusable precomputed RoPE plan API.
- Optional precomputed trig cache for bounded context windows.
- KV-cache continuation through startPos.
- Deterministic known-value tests.
- Independent scalar reference parity tests.
- Norm preservation tests.
- Partial seqLen tests.
- Invalid input validation.

## API

Exports:

- `applyRoPE(tensor, headDim, options)`
- `applyRoPESplitHalf(tensor, headDim, options)`
- `applyRoPESplitHalfWithPlan(tensor, plan, options)`
- `createRoPEPlan(headDim, base, options)`
- `applyRoPEWithPlan(tensor, plan, options)`
- `applyToHead(head, pos, headDim, base)`
- `verifyNormPreservation(original, afterRoPE, tolerance)`

Use `applyRoPE` for adjacent-pair layouts. Use `applyRoPESplitHalf` for stacks that pair the first and second halves of each head. Use `createRoPEPlan` plus `applyRoPEWithPlan` in tight loops where the same `headDim` and `base` are reused.

```js
import { applyRoPE, createRoPEPlan, applyRoPEWithPlan } from 'lyle-rope-kernel';

const headDim = 128;
const seqLen = 512;
const q = new Float32Array(seqLen * headDim);
const k = new Float32Array(seqLen * headDim);

applyRoPE(q, headDim, { startPos: 0 });
applyRoPE(k, headDim, { startPos: 0 });

const plan = createRoPEPlan(headDim);
applyRoPEWithPlan(q, plan, { startPos: 1024, seqLen: 8 });
applyRoPEWithPlan(k, plan, { startPos: 1024, seqLen: 8 });
```

For a bounded context window, add `maxSeqLen` when creating a plan. This precomputes the exact sine/cosine values once, so subsequent calls avoid trigonometry for positions inside that window. Calls outside it remain correct and use the normal math path.

```js
const cachedPlan = createRoPEPlan(128, 10000, { maxSeqLen: 8192 });
applyRoPEWithPlan(q, cachedPlan, { startPos: 0, seqLen: 512 });
```

The cache stores two `Float64Array`s of `maxSeqLen * (headDim / 2)` entries: `16 * maxSeqLen * (headDim / 2)` bytes in total. Use it only when that one-time memory cost is appropriate for a reused context window. The cache is opt-in; the default plan keeps the compact frequency-only representation.

## Support and scope

- Standard RoPE only. Scaling variants such as Llama 3, YaRN, and NTK are intentionally outside the core API for now.
- The core is portable ESM using only standard JavaScript and typed arrays, so it can run in modern Node and browser runtimes.
- Node 20 and Node 22 are verified in CI. WebGPU remains a preview fallback; the JavaScript path is the production path.

## Benchmarks

Commands:

```bash
npm test
npm run benchmark
npm run benchmark:hot
npm run benchmark:cached
```

Local validation baseline measured on Node `v22.14.0`. These numbers are not universal hardware claims; they are included as a reproducible marker for the current implementation. Run the commands above to reproduce them on your own machine.

### Copy plus compute

| headDim | seqLen | throughput |
| ---: | ---: | ---: |
| 64 | 512 | 42.9 M pairs/sec |
| 64 | 2048 | 45.6 M pairs/sec |
| 64 | 8192 | 41.5 M pairs/sec |
| 128 | 512 | 61.5 M pairs/sec |
| 128 | 2048 | 51.8 M pairs/sec |
| 128 | 8192 | 47.1 M pairs/sec |
| 256 | 512 | 63.2 M pairs/sec |
| 256 | 2048 | 55.4 M pairs/sec |
| 256 | 8192 | 48.0 M pairs/sec |

### Precomputed plan hot path

| headDim | seqLen | throughput |
| ---: | ---: | ---: |
| 64 | 512 | 59.0 M pairs/sec |
| 64 | 2048 | 54.7 M pairs/sec |
| 64 | 8192 | 47.7 M pairs/sec |
| 128 | 512 | 68.5 M pairs/sec |
| 128 | 2048 | 57.5 M pairs/sec |
| 128 | 8192 | 48.8 M pairs/sec |
| 256 | 512 | 70.5 M pairs/sec |
| 256 | 2048 | 58.2 M pairs/sec |
| 256 | 8192 | 50.6 M pairs/sec |

### Cached-plan hot path

`npm run benchmark:cached` builds the trig cache before timing and measures only application. It is the relevant mode for workloads that repeatedly use a bounded context window.

| headDim | seqLen | throughput |
| ---: | ---: | ---: |
| 64 | 512 | 220.5 M pairs/sec |
| 64 | 2048 | 462.2 M pairs/sec |
| 64 | 8192 | 396.7 M pairs/sec |
| 128 | 512 | 476.1 M pairs/sec |
| 128 | 2048 | 444.8 M pairs/sec |
| 128 | 8192 | 399.6 M pairs/sec |
| 256 | 512 | 470.8 M pairs/sec |
| 256 | 2048 | 475.3 M pairs/sec |
| 256 | 8192 | 475.0 M pairs/sec |

`npm test` also includes a steady-state regression guard. It checks that the cached path remains at least 1.5× faster than the frequency-only plan for a fixed representative workload. It is a relative check, not a cross-machine throughput claim.

## Tests

The suite covers known values, scalar reference parity, split-half parity, cached-plan parity across cache boundaries and supported dimensions, norm preservation, position-zero identity, `startPos`, partial `seqLen`, `applyToHead` parity, planned API parity, custom base behavior, long sequences, invalid inputs, and a relative hot-path regression guard.

Current marker: `17 tests / 17 passing`.

## Layout note

`applyRoPE` rotates adjacent pairs inside each row-major head: `x0/x1`, `x2/x3`, and so on. `applyRoPESplitHalf` rotates split halves: `x0/x(half)`, `x1/x(half+1)`, and so on.

## WebGPU status

The WebGPU entrypoint is preview-fallback only. Treat the JavaScript path as the production path until full GPU bindings are implemented.

## Release checklist

- [x] Package exports defined
- [x] Zero runtime dependencies
- [x] Strict input validation
- [x] Planned hot path API
- [x] Reproducible benchmark commands
- [x] Benchmarks recorded with environment note
- [x] Demo copy cleaned
- [x] README claims aligned with tests
- [x] CI uses lockfile-free install path

## License

MIT
