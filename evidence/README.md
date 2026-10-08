# Hardening evidence (2026-10-05)

Tested source commit: `b9718a62b22e55e7e3b4e13508790d18279f6ff8`.
Tested Git tree: `a62b0d77b2d0d918f208f4d0e3f59978c8fa138a`.
These evidence-only files are committed after the tested implementation so that
its source identity does not depend on a self-referential report hash.

## Retained run

- [Machine-readable report](hardening-2026-10-05.json): environment, source hashes,
  38 passing correctness tests with no skips, fresh offline tarball install,
  numerical comparison results, plan/cache memory, and raw repeated timings.
- [Unedited test output](hardening-2026-10-05.json.tap): TAP from that run; its
  SHA-256 is recorded in the report.

The run used Node 22.16.0 on a shared Linux sandbox, not dedicated benchmark
hardware. Repository content was transferred through the GitHub connector because
container DNS could not resolve GitHub. The complete uploaded source tree and
GitHub commit object were hash-verified against the local tested checkout before
this run. The report records that source commit and a clean working tree.

The numerical matrix passed the existing 1e-6 absolute tolerance, without changing
it, on bounded deterministic inputs and sampled positions through 131071.
A measured error of zero in this sample is not a bit-exact guarantee for other
inputs or runtimes. See `../docs/NUMERICS.md` for the contract and limitations.

Timing compares this package's per-head and packed implementations only. It is
not a cross-library/GPU benchmark or an end-to-end model speedup. Cache construction
is measured separately; repeated application includes fresh input copies.

## Before/after regression proof

The original commit `157036c26e799539f69d6629464e16686fe9b228` passed its existing
21 tests (including its historical timing guard). Applying the new 13-test core
regression file to that unmodified kernel produced **3 passes and 10 failures**.
After the fixes, those same 13 tests passed. Five checked-example tests plus the
original 20 correctness tests produce the retained total of 38.
The timing guard was moved, unchanged, into an opt-in command; it was not counted
as a skipped correctness test. It also passed locally when run separately.

Reproduce the failing side from a Git checkout (POSIX shell; modifies only a new
worktree, not the checked-out main branch):

```sh
git worktree add --detach ../rope-before-hardening 157036c26e799539f69d6629464e16686fe9b228
mkdir -p ../rope-before-hardening/support
git show b9718a62b22e55e7e3b4e13508790d18279f6ff8:support/reference.mjs > ../rope-before-hardening/support/reference.mjs
git show b9718a62b22e55e7e3b4e13508790d18279f6ff8:test/hardening.test.js > ../rope-before-hardening/test/hardening.test.js
(cd ../rope-before-hardening && node --test test/hardening.test.js)
# Nonzero exit is expected. Run npm run verify on the proposed branch for the fixed side.
```

For a fresh report on the proposed branch, run `npm run evidence`. Generated
reports stay in ignored `reports/`; nothing is uploaded or published by the harness.
The committed run is local evidence, not a substitute for inspecting the PR's
Linux/Windows Node 22/24 CI outcomes. No release tag or npm publication is implied.

# Throughput run (2026-10-07)

[`bench-2026-10-07.json`](bench-2026-10-07.json) is one `npm run bench` run on source commit
`89583f8be19d7b78fc10ba9ffd9d0b39edcea2a2` (clean tree), Node 22.23.3 on a shared Linux
sandbox VM (Intel Xeon, 8 logical CPUs). It contains every raw sample, the method, and the
maximum absolute error of each timed kernel case against `support/reference.mjs`
(all within `1e-6`). It is one machine and one run: indicative only, not a cross-library,
GPU or end-to-end model comparison. The README benchmark tables are copied from this run.
