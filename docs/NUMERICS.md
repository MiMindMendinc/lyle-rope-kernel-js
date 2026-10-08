# Numerical and input contract

## Defined operation

For absolute token position `p`, pair index `i`, head dimension `D`, and base `b`:

```
angle = p / b^(2*i/D)
y0 = x0*cos(angle) - x1*sin(angle)
y1 = x0*sin(angle) + x1*cos(angle)
```

The independent oracle in `support/reference.mjs` evaluates this scalar formula
without using implementation plans, frequencies, caches or rotation routines.
Adjacent layout pairs `(2*i, 2*i+1)`; split-half pairs `(i, i+D/2)`.
All heads of a packed token use the same absolute position.

The implementation stores `1 / b^(2*i/D)` in Float64, multiplies by the position,
evaluates JavaScript `Math.sin`/`Math.cos`, and rounds into Float32 tensor storage.
Multiplication by the reciprocal and direct division need not be bit-identical.
The contract is tolerance-based, not exact framework equivalence.

## Tested envelope

The regression matrix uses deterministic Float32 inputs in `[-1, 1]`, dimensions
`2, 4, 8, 64, 96, 128, 256`, bases `1, 10000, 500000`, both layouts, and 3-token
windows starting at `0, 1, 17, 1024, 8191, 8192, 32768, 65536, 131069`.
This samples absolute positions through **131071**; it does not exhaust every
position or every possible input. It checks frequency-only and cached plans
(`maxSeqLen=8193`), including windows crossing the cache boundary. The packed
matrix separately checks Q:K head counts 4:1 at high positions.

Acceptance: every tested output has maximum absolute error **<= 1e-6** against
the independent Float32-output oracle. The tolerance was not loosened from the
original short-position tests. Original tests for known values, norm preservation,
partial sequence operation, layouts and packed views remain in the suite.

Scaled inputs are tested separately (`test/scaled-input.test.js`): the same fixtures
multiplied by `1e3`, `1e6`, `1e12` and `1e30`, both layouts, cached and frequency-only
plans, packed Q/K, positions through 131069, against a combined bound
`|actual - reference| <= 1e-6 + 4 * 2^-23 * |reference|` (absolute plus four Float32
ulps relative). A purely absolute `1e-6` bound is not meaningful at large magnitudes:
at `|x| ~ 1e12` one Float32 ulp is about `1.2e5`.

Safe integer positions outside this tested envelope are accepted when their
angles are finite. That acceptance is not an accuracy guarantee for those
positions. Different input magnitudes, runtimes, extreme bases and actual models
require separate evaluation. Norm preservation alone does not validate angles.
No MMLU/perplexity, training, quantized-model, GPU, or scaled-RoPE result is claimed.

## Validation and mutation

Dimensions must be positive even safe integers. Counts, offsets and cache sizes
must be non-negative safe integers; packed head counts must be positive safe
integers. Relevant length/byte multiplications and `startPos + seqLen` (the
exclusive end, including for packed Q/K) are checked before tensor writes.
A zero-length operation may use `Number.MAX_SAFE_INTEGER` as its start; a
one-token operation may not use that start because its exclusive end overflows.

The single-tensor API requires complete rows; explicit `seqLen` can rotate fewer
rows but cannot legitimize a malformed tensor. Packed Q/K lengths must match
exactly. Overlapping Q/K views are rejected before either is changed. Invalid
options, unsafe ranges, fabricated plans and detached plan storage are rejected.

A positive finite base is required. Non-finite inverse frequencies or angles are
rejected, but finite input data can still overflow Float32 on output. Tensor
values are not scanned in the kernel. Callers must supply appropriate finite,
bounded inputs, retain exclusive ownership while rotating, and budget allocations.
`test/input-contract.test.js` pins the current behavior: a NaN or infinite element
makes both elements of its rotated pair non-finite (positions > 0) and leaves every
other element unchanged; near-`Float32` maximum finite inputs can overflow to
`Infinity`. Tensors must be `Float32Array` from the same JavaScript realm (the check
is `instanceof`), so `Float64Array`, integer arrays and cross-realm arrays throw
`TypeError`. Empty tensors are no-ops, except `applyToHead`, which needs one head.
Length arithmetic checks do not prevent all resource-exhaustion conditions.

## Plans and compatibility

`invFreq` changed from Float32Array to Float64Array. Frequency memory is
`8*(D/2)` bytes, versus `4*(D/2)` previously: for D=128, **512 vs 256 bytes**.
Cached sine/cosine storage remains two Float64 arrays, together
`16*maxSeqLen*(D/2)` bytes. Plan creation is separate from application latency.

Recreate plans using this module's `createRoPEPlan`. Plans are module-local runtime
objects, not a portable serialization format. Their typed arrays are exposed for
inspection but must not be mutated, detached, resized, shared for concurrent writes,
or overlapped by input tensors. Shallow freezing is not deep immutability.

These changes can affect low-order output bits and reject inputs that older
versions silently accepted. Revalidate integrations after upgrading. The package
remains experimental; the release candidate is not a production certification.
