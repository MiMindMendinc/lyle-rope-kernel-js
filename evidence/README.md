# Evidence

Committed measurement and test evidence for this repository. Nothing here is uploaded or
published by the harnesses; generated reports stay in ignored `reports/`.

## Hardening evidence (2026-10-05)

Tested source commit: `b9718a62b22e55e7e3b4e13508790d18279f6ff8`.
Tested Git tree: `a62b0d77b2d0d918f208f4d0e3f59978c8fa138a`.
These evidence-only files are committed after the tested implementation so that
its source identity does not depend on a self-referential report hash.

### Retained run

- [Machine-readable report](hardening-2026-10-05.json): environment, source hashes,
  38 passing correctness tests with no skips, fresh offline tarball install,
  numerical comparison results, plan/cache memory, and raw repeated timings.
- [Unedited test output](hardening-2026-10-05.json.tap): TAP from that run; its
  SHA-256 is recorded in the report.

The run used Node 22.16.0 on a shared Linux VM, not dedicated benchmark hardware.
The report records the source commit and a clean working tree.

The numerical matrix passed the existing 1e-6 absolute tolerance, without changing
it, on bounded deterministic inputs and sampled positions through 131071.
A measured error of zero in this sample is not a bit-exact guarantee for other
inputs or runtimes. See `../docs/NUMERICS.md` for the contract and limitations.

Timing compares this package's per-head and packed implementations only. It is
not a cross-library/GPU benchmark or an end-to-end model speedup. Cache construction
is measured separately; repeated application includes fresh input copies.

### Before/after regression proof

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
# Nonzero exit is expected. Run npm run verify on main for the fixed side.
```

For a fresh report, run `npm run evidence`. The committed run is local evidence;
the Linux/Windows Node 22/24 CI runs are the cross-platform check.

## Throughput runs (2026-10-08)

[`bench-2026-10-08.json`](bench-2026-10-08.json) combines **5 back-to-back `npm run bench`
runs** on source commit `40906c0dd0321048d92cfefb51b5e77e0d435697` (every run records a clean
tree; that PR #8 branch commit was squash-merged to `main` as `4c92b89`, with identical `src/`,
`bench/throughput.mjs` and `bench/baseline.mjs`), Node 22.23.3 on a shared Linux sandbox VM (Intel Xeon, 8 logical CPUs). It was built
with `npm run bench:aggregate`, which refuses fewer than 3 runs, dirty trees, mixed commits,
mixed Node/CPU, or any case that failed its reference check. The file holds:

- `runs`: each complete per-run report (environment, method, every raw sample, and the
  maximum absolute error of each timed case against `support/reference.mjs`, all within `1e-6`);
- `summary.cases`: per case, the five per-run medians, their median (the README value) and
  their min–max (the README range);
- `summary.variability`: the figures quoted in the README noise sentence (median and maximum
  min–max spread as a percentage of the cell median, the median typical deviation, and how
  many cells were more than 20% above their median in each run).

`test/bench-evidence.test.js` recomputes the summary from the stored runs and checks that the
README tables and noise sentence match it, so the README cannot drift from this file.
The comparison column is `bench/baseline.mjs`, a plain in-place scalar loop that is not part of
the package. Group `oracle` times the allocating reference oracle for completeness only; it is
not a performance baseline and is not shown in the README.

One machine, one session: indicative only, not a cross-library, GPU or end-to-end model
comparison. The aggregation code was finalized after the runs (it only post-processes the
stored JSON); the measured bench code is unchanged from the recorded commit.

[`bench-2026-10-07.json`](bench-2026-10-07.json) is an earlier single run on `89583f8` (before
the baseline column existed). It is kept for history and is not used by the README.
