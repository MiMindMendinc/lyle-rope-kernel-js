# lyle-rope-kernel-js

Experimental, zero-dependency JavaScript RoPE library with in-place rotation,
reusable plans, and packed Q/K support.

[![CI](https://github.com/MiMindMendinc/lyle-rope-kernel-js/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MiMindMendinc/lyle-rope-kernel-js/actions/workflows/ci.yml)

The badge tracks `main`; proposed changes have their own pull-request checks.
This is an experimental CPU component, not a production-certified inference engine.

## Scope and evidence

| Area | Scope | Evidence |
| --- | --- | --- |
| Rotation | In-place Float32 tensors; adjacent and split-half layouts | `test/rope.test.js` |
| Numerical behavior | Float64 frequencies and arithmetic; Float32 output; explicit tolerance and sampled position range | `docs/NUMERICS.md`, `test/hardening.test.js` |
| Packed Q/K | `[tokens, heads, headDim]`, unequal Q/K head counts, shared angles | Both correctness test files |
| Example | Single-head, non-causal attention on small deterministic inputs | `test/example.test.js` |
| Packaging | Actual tarball installed into a fresh offline consumer | `npm run test:package` |
| Performance | Repeated local measurements against this package's per-head path | `npm run evidence` |

Not implemented: GPU execution, a transformer, model loading, an attention engine,
a KV cache, Llama 3/YaRN/NTK scaling, or end-to-end model-quality validation.
`startPos` supplies an absolute position; it does not implement a KV cache.

## Run from a checkout

Use Node 22 or newer. The CI matrix targets Node 22 and 24 on Linux and Windows;
check the actual workflow run for its results. No runtime or development packages
are required beyond Node and npm.

```sh
git clone https://github.com/MiMindMendinc/lyle-rope-kernel-js.git
cd lyle-rope-kernel-js
npm ci --offline --ignore-scripts --no-audit --no-fund
npm run verify
npm run example
npm run evidence
```

The clone requires network access. Subsequent commands are local. `verify` runs
correctness tests and a fresh tarball install; it does not publish anything.
Evidence is saved under ignored `reports/` as JSON plus a raw TAP test log.

To install your own local build into another project:

```sh
npm pack --offline --ignore-scripts
# In the consumer project, use the actual path printed by npm pack:
npm install /absolute/path/to/lyle-rope-kernel-1.1.0-rc.1.tgz --offline --ignore-scripts --no-audit --no-fund
```

The version in this branch is a release candidate, not a claim of an npm publication.

## API

```js
import { createRoPEPlan, applyRoPEQK } from 'lyle-rope-kernel';

const headDim = 4;
const plan = createRoPEPlan(headDim);
const q = Float32Array.from([1, 0, 0, 1, 0.5, -0.5, 1, 0]);
const k = Float32Array.from([1, 0, 0.5, 0.5]);
// One token, two query heads, one key head. Every head uses position 8192.
applyRoPEQK(q, k, plan, {
  qHeads: 2, kvHeads: 1, startPos: 8192, layout: 'adjacent',
});
```

| Export | Purpose |
| --- | --- |
| `createRoPEPlan(headDim, base = 10000, { maxSeqLen = 0 })` | Reusable frequency plan; optional bounded trig cache |
| `applyRoPE(tensor, headDim, options)` | Adjacent-pair rotation; creates a plan per call |
| `applyRoPESplitHalf(tensor, headDim, options)` | Split-half rotation; creates a plan per call |
| `applyRoPEWithPlan(tensor, plan, options)` | Adjacent-pair rotation with a reused plan |
| `applyRoPESplitHalfWithPlan(tensor, plan, options)` | Split-half rotation with a reused plan |
| `applyRoPEQK(q, k, plan, options)` | Packed Q/K rotation, returning the original `{ q, k }` |
| `applyToHead(head, pos, headDim, base)` | Rotate exactly one head |
| `verifyNormPreservation(before, after, tolerance = 1e-5)` | Absolute whole-tensor L2 diagnostic, not an angle-correctness oracle |
| `DEFAULT_BASE` | `10000` |

Single-tensor options: `startPos` and `seqLen`; direct calls also accept `base`.
Single-tensor shapes are `[seqLen, headDim]`, **not** flattened multi-head tensors.
Use `applyRoPEQK` for packed multi-head data. Its options require positive
`qHeads`/`kvHeads`, and accept `startPos`, `seqLen`, and `layout` (`adjacent` or
`split-half`). Packed tensor lengths must exactly match the requested shapes;
Q and K must not overlap. V is not an argument and is not rotated.

All operations rotate their inputs in place. Do not apply RoPE twice to already
rotated keys. Single-tensor `seqLen` may select a prefix of complete rows, leaving
the remaining rows unchanged. Incomplete rows are rejected even with explicit
`seqLen`. All positions and lengths, including the exclusive end
`startPos + seqLen`, must fit the safe-integer range.

## Precision, caching and migration

Read the [numerical contract](docs/NUMERICS.md) before integrating with a model.
The tested absolute tolerance remains `1e-6` for bounded fixtures. This does not
establish bit-exact parity with a model framework or preserved model accuracy.

`createRoPEPlan(128, 10000, { maxSeqLen: 8192 })` precomputes positions 0..8191.
Positions outside the cache fall back to the same frequency/math path. The cache
is optional. Frequency storage uses `4 * headDim` bytes; both trig tables together
use `8 * maxSeqLen * headDim` bytes. A 128-dimensional, 8192-position cache is
8 MiB plus 512 frequency bytes (excluding object overhead). Budget memory before
creating a plan from untrusted sizes; safe-integer checks are not memory quotas.

Plans now use **Float64Array** frequencies. Recreate plans when upgrading;
serialized, fabricated, transferred, or other-module plans are not accepted.
The object is shallow-frozen: its typed-array contents must be treated as read-only.
Do not mutate, resize, detach, alias tensor inputs onto, or concurrently modify
plan storage. Tensor values are not scanned for finiteness in the hot path;
non-finite or overflowing data is outside the numerical contract.

## Reproducible measurements

`npm run evidence` records the source commit (or `null` outside Git), dirty state,
SHA-256 hashes of source/test/harness files, Node/V8/OS/CPU identifiers, test
counts, package-install results, numerical errors, plan-build time, cache bytes,
and seven raw timing samples per mode. It does not collect hostnames, usernames,
keys, model data, or environment-variable dumps, and does not upload results.
Review generated reports before sharing them.

The harness checks all compared outputs outside the timed region. It compares
per-head, packed, and packed-cached calls **within this package**, for one-token
and 128-token workloads in both layouts. Every mode includes fresh input copies;
plan creation is timed separately. Mode order rotates across samples. Results
are machine/workload-specific, not claims against external libraries or GPUs.

Legacy `benchmark`, `benchmark:hot`, `benchmark:cached`, and `benchmark:qk`
commands remain exploratory tools. Use the evidence harness for retained raw
samples and environment information. The historical cached-path timing guard is
now opt-in (`npm run test:performance`), not a correctness CI gate.

## Examples and browser status

`npm run example` checks a single-head attention result against an analytic
answer. It uses a quadratic score matrix and is educational, non-causal, and
unsuitable as a production attention implementation. The former `SimpleAttention`
name remains an alias; a second `nHeads` argument now throws instead of being ignored.
`node examples/packed-qk.js` demonstrates packed shapes only.

`demo/index.html` is a static project information page, not an executing browser
benchmark. Browser runtime correctness is not yet covered by this CI matrix.
The `lyle-rope-kernel/webgpu` compatibility entry point is a **CPU fallback**:
its shader is empty and it does not request a GPU device or run GPU computation.

## Release gate

Before merging or publishing, require passing PR correctness and package checks,
review the numerical contract and migration notes, and retain a source-identified
evidence report. See [CHANGELOG.md](CHANGELOG.md). No general security audit,
model integration certification, or universal speed claim is implied.

MIT. Copyright information is retained in [LICENSE](LICENSE).
