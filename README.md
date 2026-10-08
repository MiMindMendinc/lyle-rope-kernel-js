# lyle-rope-kernel-js

Experimental, zero-dependency JavaScript RoPE library with in-place rotation,
reusable plans, and packed Q/K support.

**[Live browser demo](https://mimindmendinc.github.io/lyle-rope-kernel-js/playground.html)**:
rotate synthetic data with the real ES module in your browser, see the angles, check
the result against the scalar reference, and time it on your device. No analytics,
no network requests beyond the page's own files.

[Live demo](https://mimindmendinc.github.io/lyle-rope-kernel-js/playground.html) · [Benchmarks](#benchmarks-one-machine-indicative-only) · [Install](#install)

[![CI](https://github.com/MiMindMendinc/lyle-rope-kernel-js/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MiMindMendinc/lyle-rope-kernel-js/actions/workflows/ci.yml)

This is an experimental CPU component, not a production-certified inference engine.
The badge shows CI status for `main`; run `npm run verify` to check a local checkout.

Prerelease, install and migration notes: [docs/PRERELEASE_NOTES.md](https://github.com/MiMindMendinc/lyle-rope-kernel-js/blob/main/docs/PRERELEASE_NOTES.md).
Report problems via [GitHub Issues](https://github.com/MiMindMendinc/lyle-rope-kernel-js/issues) (include Node version, OS, commit SHA, and the failing command).

## Scope and evidence

| Area | Scope | Evidence |
| --- | --- | --- |
| Rotation | In-place Float32 tensors; adjacent and split-half layouts | `test/rope.test.js` |
| Numerical behavior | Float64 frequencies and arithmetic; Float32 output; explicit tolerance and sampled position range; scaled inputs to 1e30 (abs + rel tolerance); wrong types, empty tensors and NaN pinned | `docs/NUMERICS.md`, `test/hardening.test.js`, `test/scaled-input.test.js`, `test/input-contract.test.js` |
| Packed Q/K | `[tokens, heads, headDim]`, unequal Q/K head counts, shared angles | `test/rope.test.js`, `test/hardening.test.js`, `test/scaled-input.test.js` |
| Example | Single-head, non-causal attention on small deterministic inputs | `test/example.test.js` |
| Packaging | Actual tarball installed into a fresh offline consumer | `npm run test:package` |
| Performance | Per-head vs packed timings within this package, with source hashes and raw samples | `npm run evidence` |
| Throughput | Median ns per rotated pair over a fixed shape matrix; each case reference-checked first | `npm run bench`, `evidence/bench-*.json` |
| Browser demo | Same kernel file served by Pages; in-page reference check | `demo/playground.html`, `test/demo.test.js` |

Not implemented: GPU execution, a transformer, model loading, an attention engine,
a KV cache, Llama 3/YaRN/NTK scaling, or end-to-end model-quality validation.
`startPos` supplies an absolute position; it does not implement a KV cache.

## Install

> **Not yet published to npm.** Install from a [checkout or a local tarball](#run-from-a-checkout).
> Once a prerelease is published it will use the `next` dist-tag, so you would install it by
> explicit version:

```sh
# Future install command; this does not work until the package is published.
npm install lyle-rope-kernel@1.1.0-rc.1
```

Quick start (after installing a local tarball; in a checkout, import from `./src/rope-kernel.js`):

```js
import { createRoPEPlan, applyRoPEWithPlan } from 'lyle-rope-kernel';

const headDim = 64, seqLen = 128;
const plan = createRoPEPlan(headDim, 10000, { maxSeqLen: seqLen }); // optional trig cache
// One head, shape [seqLen, headDim], filled with non-zero demo data.
const x = Float32Array.from({ length: seqLen * headDim }, (_, i) => Math.sin(i));
applyRoPEWithPlan(x, plan); // rotates x in place for positions 0..127
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
linux 6.12.94+ x64, 2026-10-08, source commit `40906c0dd032` (clean tree), a PR #8 branch commit that was squash-merged
to `main` as `4c92b89` with identical `src/` and `bench/` measurement code.
**5 back-to-back runs** were combined with `npm run bench:aggregate`; every run and raw sample is in
[`evidence/bench-2026-10-08.json`](https://github.com/MiMindMendinc/lyle-rope-kernel-js/blob/main/evidence/bench-2026-10-08.json).
**These numbers are from one machine and are indicative only.** Your hardware, Node version
and workload will give different numbers. No other library was benchmarked.

Each cell is the **median, across the 5 runs, of each run's median nanoseconds per rotated
(x0, x1) pair**, with the **min–max of those per-run medians** in parentheses; lower means less
time per pair. Each run takes 11 samples of about 40 ms per case after a 150 ms warmup. The timed
region is in-place rotation of one reused `Float32Array` starting at position 0; plan construction
and input copies are not timed. Packed Q/K counts both Q and K pairs. Before timing, every case,
including the baseline, is checked against the scalar reference within the tested `1e-6` tolerance.

| Shape [seq, headDim] | applyRoPEWithPlan, no cache | applyRoPEWithPlan, cached | applyRoPESplitHalfWithPlan, cached | Plain in-place scalar loop (baseline, not this package) |
| --- | ---: | ---: | ---: | ---: |
| [128, 64] | 16.3 (14.2–24.4) | 2.96 (2.81–4.44) | 3.07 (2.99–4.35) | 14.4 (14.2–15.7) |
| [512, 64] | 18.4 (18.3–18.8) | 2.99 (2.74–3.22) | 3.04 (3.01–3.68) | 18.7 (18.5–18.8) |
| [2048, 64] | 22.8 (22.3–22.8) | 2.73 (2.69–2.86) | 3.12 (3.01–4.07) | 22.6 (21.9–25.1) |
| [128, 128] | 14.8 (14.0–20.2) | 2.85 (2.67–4.00) | 3.01 (3.00–4.37) | 15.3 (15.0–22.5) |
| [512, 128] | 20.3 (18.2–24.6) | 2.73 (2.68–2.78) | 3.03 (2.97–3.06) | 17.8 (17.5–27.3) |
| [2048, 128] | 22.7 (21.3–23.6) | 2.71 (2.67–3.36) | 3.03 (3.00–3.57) | 22.0 (21.3–29.9) |

| Packed Q/K shape [seq, Q heads / KV heads, headDim] | applyRoPEQK, no cache | applyRoPEQK, cached |
| --- | ---: | ---: |
| [128, 8/8, 64] | 4.15 (4.04–6.32) | 3.03 (3.02–3.73) |
| [128, 32/8, 64] | 3.30 (3.29–3.35) | 2.92 (2.89–3.31) |
| [512, 8/8, 64] | 4.36 (4.27–4.47) | 3.12 (3.04–5.46) |
| [512, 32/8, 64] | 3.58 (3.54–5.78) | 2.96 (2.89–4.73) |
| [2048, 8/8, 64] | 4.66 (4.56–7.59) | 3.17 (3.04–5.44) |
| [2048, 32/8, 64] | 4.44 (4.34–6.88) | 3.75 (3.46–5.62) |
| [128, 8/8, 128] | 4.02 (3.87–6.91) | 3.01 (2.97–5.51) |
| [128, 32/8, 128] | 3.54 (3.34–5.87) | 3.03 (2.98–5.34) |
| [512, 8/8, 128] | 4.48 (4.30–7.52) | 3.40 (3.13–5.99) |
| [512, 32/8, 128] | 3.64 (3.50–7.17) | 3.11 (3.03–6.54) |
| [2048, 8/8, 128] | 5.70 (5.46–9.27) | 5.22 (4.10–6.84) |
| [2048, 32/8, 128] | 4.97 (4.51–7.36) | 3.93 (3.88–6.54) |

The **plain in-place scalar loop** (`bench/baseline.mjs`) is not part of this package. It is what a
straightforward hand-written implementation does: inverse frequencies once per call into a reused
scratch array, `Math.cos`/`Math.sin` for every pair, results written back in place, no output
allocation. Its numbers are in the same range as this package's no-cache path, which does the same
per-pair arithmetic plus input validation. The cached columns read precomputed cos/sin values from
the optional trig table instead of calling `Math.cos`/`Math.sin` for every pair.

Not shown: the JSON also times `support/reference.mjs` (group `oracle`), the correctness oracle
used by the tests. It allocates a new output array and recomputes `Math.pow` per element, so it
is not a performance baseline and is deliberately left out of these tables.

**Run-to-run noise:** this shared VM is noisy. A typical run was within 2.1% of the cell median,
but occasional slow runs widened the ranges: the median cell's min–max span was 48% of its median,
up to 113% for [512, 32/8, 128] applyRoPEQK, cached trig table. Runs 1–5 had 4, 0, 5, 9 and 20
of the 48 cells more than 20% above their median. Treat differences inside the ranges as noise.
Reproduce with:

```sh
npm run bench                       # one run: prints the tables; JSON goes to ignored reports/
npm run bench -- --out run-1.json   # choose the JSON path; repeat for run-2.json, run-3.json, ...
npm run bench:aggregate -- --out combined.json run-1.json run-2.json run-3.json
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

Plans use **Float64Array** frequencies and are not serializable: serialized, fabricated,
transferred, or other-module plans are rejected (see [CHANGELOG.md](CHANGELOG.md) when upgrading).
The object is shallow-frozen: its typed-array contents must be treated as read-only.
Do not mutate, resize, detach, alias tensor inputs onto, or concurrently modify
plan storage. Tensor values are not scanned for finiteness in the hot path;
non-finite or overflowing data is outside the numerical contract.

## Evidence harness (`npm run evidence`)

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

`npm run test:performance` runs an opt-in cached-vs-uncached timing check; it is not part of CI.

## Examples and browser status

`npm run example` checks a single-head attention result against an analytic
answer. It uses a quadratic score matrix and is educational, non-causal, and
unsuitable as a production attention implementation. `SimpleAttention` is an alias
of the example class and accepts a single constructor argument.
`node examples/packed-qk.js` demonstrates packed shapes only.

The Pages site is built by `npm run site:build` (copies `demo/` plus the unmodified
`src/rope-kernel.js`, and `support/reference.mjs` served as `support/reference.js`, into ignored `_site/`). Preview it locally
with `npm run site:serve` (loopback only, http://127.0.0.1:8080/). `demo/playground.html`
runs the kernel in the browser on seeded synthetic data and reports its own reference
check and browser timing; its logic is unit-tested in Node by `test/demo.test.js`.
Browser runtime correctness is still not covered by this CI matrix.
The `lyle-rope-kernel/webgpu` compatibility entry point is a **CPU fallback**:
its shader is empty and it does not request a GPU device or run GPU computation.

## Contributing / releases

Pull requests need passing correctness and package checks (`npm run verify`). Changes are
listed in [CHANGELOG.md](CHANGELOG.md). This package has not had a security audit or
model-integration testing.

## License

Licensed under the [MIT License](LICENSE).
