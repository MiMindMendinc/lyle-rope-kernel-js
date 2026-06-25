# lyle-rope-kernel-js

Zero-dependency, in-place RoPE kernel in pure JavaScript for local transformer experiments.

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

- applyRoPE(tensor, headDim, options)
- createRoPEPlan(headDim, base)
- applyRoPEWithPlan(tensor, plan, options)
- applyToHead(head, pos, headDim, base)
- verifyNormPreservation(original, afterRoPE, tolerance)

Use applyRoPE for simple calls. Use createRoPEPlan plus applyRoPEWithPlan in tight loops where the same headDim and base are reused.

## Benchmarks

Run npm run benchmark for copy plus compute timing.

Run npm run benchmark:hot for the precomputed-plan path.

The default benchmark includes Float32Array copy cost. The hot path reuses a RoPE plan.

## Tests

Run npm test.

The suite covers known values, scalar reference parity, norm preservation, position-zero identity, startPos, partial seqLen, applyToHead parity, planned API parity, custom base behavior, long sequences, and invalid inputs.

## Layout note

This implementation rotates adjacent pairs inside each row-major head: x0/x1, x2/x3, and so on. Some model stacks use split-half rotary layouts, so confirm the target layout before integration.

## WebGPU status

The WebGPU file is preview-only. Treat the JavaScript path as the production path until full GPU bindings are implemented.

## License

MIT
