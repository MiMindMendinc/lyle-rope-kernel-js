# lyle-rope-kernel-js

> Zero-dependency, in-place Rotary Position Embedding kernel for JavaScript inference experiments.

![Tests](https://img.shields.io/badge/tests-14%2F14%20passing-brightgreen)
![Runtime](https://img.shields.io/badge/runtime-Node%2020%2B-blue)
![Dependencies](https://img.shields.io/badge/dependencies-0-success)
![Module](https://img.shields.io/badge/module-ESM-purple)
![License](https://img.shields.io/badge/license-MIT-yellow)
![Status](https://img.shields.io/badge/status-release%20candidate-success)

## Plaques

| Plaque | Status |
| --- | --- |
| Correctness Gate | 14/14 tests passing |
| Reference Gate | deterministic scalar reference parity |
| Stability Gate | L2 norm preservation checked |
| KV Cache Gate | startPos continuation checked |
| Hot Path Gate | reusable precomputed RoPE plan |
| Dependency Gate | zero runtime dependencies |
| Claim Hygiene Gate | no unsupported fastest or fake benchmark claims |
| WebGPU Gate | preview fallback only |

## What is verified

- In-place Float32Array rotation.
- Reusable precomputed RoPE plan API.
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
- `createRoPEPlan(headDim, base)`
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

## Benchmarks

Commands:

```bash
npm test
npm run benchmark
npm run benchmark:hot
```

Local validation baseline from Node `v22.16.0` in this session. These numbers are not universal hardware claims; they are included as a reproducible marker for the current implementation.

### Copy plus compute

| headDim | seqLen | throughput |
| ---: | ---: | ---: |
| 64 | 512 | 25.4 M pairs/sec |
| 64 | 2048 | 31.6 M pairs/sec |
| 64 | 8192 | 33.5 M pairs/sec |
| 128 | 512 | 46.9 M pairs/sec |
| 128 | 2048 | 40.5 M pairs/sec |
| 128 | 8192 | 38.2 M pairs/sec |
| 256 | 512 | 50.9 M pairs/sec |
| 256 | 2048 | 46.9 M pairs/sec |
| 256 | 8192 | 39.6 M pairs/sec |

### Precomputed plan hot path

| headDim | seqLen | throughput |
| ---: | ---: | ---: |
| 64 | 512 | 50.1 M pairs/sec |
| 64 | 2048 | 48.3 M pairs/sec |
| 64 | 8192 | 40.9 M pairs/sec |
| 128 | 512 | 60.3 M pairs/sec |
| 128 | 2048 | 48.5 M pairs/sec |
| 128 | 8192 | 43.2 M pairs/sec |
| 256 | 512 | 56.8 M pairs/sec |
| 256 | 2048 | 44.9 M pairs/sec |
| 256 | 8192 | 41.4 M pairs/sec |

## Tests

The suite covers known values, scalar reference parity, split-half parity, norm preservation, position-zero identity, `startPos`, partial `seqLen`, `applyToHead` parity, planned API parity, custom base behavior, long sequences, and invalid inputs.

Current marker: `14 tests / 14 passing`.

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
