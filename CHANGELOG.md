# Changelog

## 1.1.0-rc.1 (proposed; not published)

- Preserve inverse-frequency precision in Float64 while retaining Float32 tensors.
- Add an independent long-position reference matrix without relaxing the 1e-6
  absolute tolerance on bounded fixtures; test both layouts, cache boundaries
  and packed unequal Q/K heads.
- Validate safe integers, byte/shape arithmetic, exclusive position ends,
  complete rows, options, authentic module-local plans and intact plan storage.
- Replace the misleading multi-head example with checked, single-head non-causal
  attention. Preserve `SimpleAttention` as an alias; reject its old ignored
  second constructor argument. Preserve the wrapper's caller inputs.
- Add a fresh offline tarball-consumer check covering exports, numeric output,
  unsafe-position rejection and the explicitly CPU-only WebGPU fallback.
- Add source-identified numerical and repeated benchmark JSON/TAP evidence.
- Move the timing regression guard out of correctness CI into an opt-in command.
- Replace static verification/test-count badges and plaques with scoped status,
  a live main-branch CI badge, migration notes and reproducible commands.
- Target Node 22/24 on Linux/Windows in CI; disable install lifecycle scripts.
- Add `npm run bench`: a Node-built-ins throughput matrix (headDim 64/128, seq
  128/512/2048, packed 8/8 and 32/8 heads) that checks each case against the scalar
  reference before timing; retain one run under `evidence/` with machine details.
- Add a live browser demo (`demo/playground.html`) that imports the unmodified ES
  module, checks it against the reference in the page and times it in the browser;
  Pages now builds `_site/` with `npm run site:build`. No analytics or network requests.
  Number fields re-run automatically; invalid or empty values clear the previous
  verdict, timing and visuals; positions are capped at 131071, the highest position
  the numerical tests cover.
- Packaging metadata for a future publish: `sideEffects: false`, a
  `./package.json` export, more specific keywords and a `prepublishOnly` verify gate.
  The package is still not published.

### Compatibility notes

This is an experimental release candidate. Recreate plans after upgrading:
`invFreq` is now Float64 and plans are module-local, not serialized structures.
Incomplete rows and unsafe positions that older versions accepted now throw.
The declared Node minimum is 22. Public CPU function names and return conventions
are retained. No GPU implementation, release tag or registry publication is added.
