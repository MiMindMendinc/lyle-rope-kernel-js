# lyle-rope-kernel-js

Experimental, zero-dependency JavaScript RoPE library with in-place rotation,
reusable plans, and packed Q/K support.

**[Live browser demo](https://mimindmendinc.github.io/lyle-rope-kernel-js/playground.html)**:
rotate synthetic data with the real ES module in your browser, see the angles, check
the result against the scalar reference, and time it on your device. No analytics,
no network requests beyond the page's own files.
· [Benchmarks](#benchmarks-one-machine-indicative-only) · [Install](#install)

[![CI](https://github.com/MiMindMendinc/lyle-rope-kernel-js/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MiMindMendinc/lyle-rope-kernel-js/actions/workflows/ci.yml)

The badge tracks `main`; proposed changes have their own pull-request checks.
This is an experimental CPU component, not a production-certified inference engine.

**Status vocabulary:** local `npm run verify` = TESTED LOCALLY; green PR checks = CI VERIFIED for that revision; green `main` badge = verified main. Do not conflate these.

Experimental prerelease / install / migration notes: [docs/PRERELEASE_NOTES.md](https://github.com/MiMindMendinc/lyle-rope-kernel-js/blob/main/docs/PRERELEASE_NOTES.md). Report problems via [GitHub Issues](https://github.com/MiMindMendinc/lyle-rope-kernel-js/issues) (include Node version, OS, commit SHA, and failing command).

## Scope and evidence

| Area | Scope | Evidence |
| --- | --- | --- |
| Rotation | In-place Float32 tensors; adjacent and split-half layouts | `test/rope.test.js` |
| Numerical behavior | Float64 frequencies and arithmetic; Float32 output; explicit tolerance and sampled position range; scaled inputs to 1e30 (abs + rel tolerance); wrong types, empty tensors and NaN pinned | `docs/NUMERICS.md`, `test/hardening.test.js`, `test/scaled-input.test.js`, `test/input-contract.test.js` |
| Packed Q/K | `[tokens, heads, headDim]`, unequal Q/K head counts, shared angles | Both correctness test files |
| Example | Single-head, non-causal attention on small deterministic inputs | `test/example.test.js` |
| Packaging | Actual tarball installed into a fresh offline consumer | `npm run test:package` |
| Performance | Repeated local measurements against this package's per-head path | `npm run evidence` |
| Throughput | Median ns per rotated pair over a fixed shape matrix; each case reference-checked first | `npm run bench`, `evidence/bench-*.json` |
| Browser demo | Same kernel file served by Pages; in-page reference check | `demo/playground.html`, `test/demo.test.js` |

Not implemented: GPU execution, a transformer, model loading, an attention engine,
a KV cache, Llama 3/YaRN/NTK scaling, or end-to-end model-quality validation.
`startPos` supplies an absolute position; it does not implement a KV cache.

## Install

> **Check npm for published versions first.** The
> [`lyle-rope-kernel` page on npmjs](https://www.npmjs.com/package/lyle-rope-kernel) lists
> every version that has actually been published. If `1.1.0-rc.1` is not listed there,
> use a checkout or a local tarball ([Run from a checkout](#run-from-a-checkout)).
> Prereleases are meant to be published under the `next` dist-tag (`publishConfig` in
> `package.json`), so install an explicit version. When it is listed, use:

```sh
npm install lyle-rope-kernel@1.1.0-rc.1
```

```js
import { createRoPEPlan, applyRoPEWithPlan } from 'lyle-rope-kernel';

const headDim = 64, seqLen = 128;
const plan = createRoPEPlan(headDim, 10000, { maxSeqLen: seqLen }); // optional trig cache
const x = new Float32Array(seqLen * headDim); // one head, shape [seqLen, headDim]
applyRoPEWithPlan(x, plan); // rotates in place for positions 0..127
```

It is a single ES module with no dependencies and no install scripts. Node 22+ is
required and CI-tested on Node 22 and 24; browsers are exercised only by the live demo's
own in-page check, not by CI.

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

The version in `package.json` is a release candidate; it is not by itself a claim that
that version was published.

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

## Benchmarks (one machine, indicative only)

Measured with `npm run bench` on **one machine**: Intel(R) Xeon(R) Processor (8 logical
CPUs, shared Linux sandbox VM, not dedicated benchmark hardware), Node v22.23.3 (V8 12.4.254.21-node.57),
linux 6.12.94+ x64, 2026-10-07, source commit `89583f8be19d`
(clean tree). Raw samples and settings: [`evidence/bench-2026-10-07.json`](evidence/bench-2026-10-07.json).
**These numbers are from one machine and one run, and are indicative only.** Your hardware,
Node version and workload will give different numbers.

Values are the **median nanoseconds per rotated (x0, x1) pair** (lower means less time
per pair) over 11 samples of about 40 ms each, after a 150 ms warmup. The timed region is
in-place rotation of one reused `Float32Array` starting at position 0; plan construction
and input copies are not timed. Packed Q/K counts both Q and K pairs. Before timing, every
kernel case is checked against the scalar reference within the tested `1e-6` tolerance.

| Shape [seq, headDim] | applyRoPEWithPlan, no cache | applyRoPEWithPlan, cached | applyRoPESplitHalfWithPlan, cached | Naive JS reference in this repo |
| --- | ---: | ---: | ---: | ---: |
| [128, 64] | 14.2 | 2.91 | 3.29 | 95.4 |
| [512, 64] | 18.3 | 2.72 | 3.05 | 100 |
| [2048, 64] | 22.1 | 2.71 | 2.98 | 107 |
| [128, 128] | 15.2 | 2.63 | 3.06 | 94.2 |
| [512, 128] | 17.9 | 2.74 | 3.03 | 106 |
| [2048, 128] | 22.9 | 2.62 | 3.08 | 111 |

| Packed Q/K shape [seq, Q heads / KV heads, headDim] | applyRoPEQK, no cache | applyRoPEQK, cached |
| --- | ---: | ---: |
| [128, 8/8, 64] | 4.14 | 3.26 |
| [128, 32/8, 64] | 3.46 | 2.96 |
| [512, 8/8, 64] | 4.32 | 3.13 |
| [512, 32/8, 64] | 3.49 | 2.97 |
| [2048, 8/8, 64] | 4.90 | 3.22 |
| [2048, 32/8, 64] | 4.62 | 4.16 |
| [128, 8/8, 128] | 4.02 | 3.51 |
| [128, 32/8, 128] | 3.46 | 2.98 |
| [512, 8/8, 128] | 4.26 | 3.16 |
| [512, 32/8, 128] | 3.53 | 3.15 |
| [2048, 8/8, 128] | 5.69 | 4.16 |
| [2048, 32/8, 128] | 4.62 | 4.26 |

"Naive JS reference in this repo" is `support/reference.mjs`, the scalar correctness
oracle used by the tests. It allocates a new output array and recomputes `Math.pow` per
element; it is not an optimized implementation, and no other library was benchmarked.
A second back-to-back run on the same commit differed from this one by a median of 2.6%
per cell and by up to 41% in the noisiest cell, so treat small differences as noise.
Reproduce with:

```sh
npm run bench                       # prints the tables; JSON goes to ignored reports/
npm run bench -- --out my-run.json  # choose the JSON path
```

## Precision, caching and migration

Read the [numerical contract](docs/NUMERICS.md) before integrating with a model.
The tested absolute tolerance remains `1e-6` for bounded fixtures in `[-1, 1)`;
inputs scaled up to `1e30` are tested against `1e-6 + 4 Float32 ulps * |reference|`. This does not
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

`npm run bench` is the throughput matrix shown above; it writes JSON to ignored
`reports/` unless given `--out`. Legacy `benchmark`, `benchmark:hot`, `benchmark:cached`,
and `benchmark:qk` commands remain exploratory tools. Use the evidence harness for retained raw
samples and environment information. The historical cached-path timing guard is
now opt-in (`npm run test:performance`), not a correctness CI gate.

## Examples and browser status

`npm run example` checks a single-head attention result against an analytic
answer. It uses a quadratic score matrix and is educational, non-causal, and
unsuitable as a production attention implementation. The former `SimpleAttention`
name remains an alias; a second `nHeads` argument now throws instead of being ignored.
`node examples/packed-qk.js` demonstrates packed shapes only.

The Pages site is built by `npm run site:build` (copies `demo/` plus the unmodified
`src/rope-kernel.js` and `support/reference.mjs` into ignored `_site/`). Preview it locally
with `npm run site:serve` (loopback only, http://127.0.0.1:8080/). `demo/playground.html`
runs the kernel in the browser on seeded synthetic data and reports its own reference
check and browser timing; its logic is unit-tested in Node by `test/demo.test.js`.
Browser runtime correctness is still not covered by this CI matrix.
The `lyle-rope-kernel/webgpu` compatibility entry point is a **CPU fallback**:
its shader is empty and it does not request a GPU device or run GPU computation.

## Release gate

Before merging or publishing, require passing PR correctness and package checks,
review the numerical contract and migration notes, and retain a source-identified
evidence report. See [CHANGELOG.md](CHANGELOG.md). No general security audit,
model integration certification, or universal speed claim is implied.

MIT. Copyright information is retained in [LICENSE](LICENSE).
