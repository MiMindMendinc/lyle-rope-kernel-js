# Prerelease notes: lyle-rope-kernel 1.1.0-rc.1

`1.1.0-rc.1` is an experimental release candidate. It is **not yet published to npm**;
install it from a checkout or a local tarball as described below. The full list of
changes is in [CHANGELOG.md](../CHANGELOG.md).

Author: Lyle Perrien II  
Package: `lyle-rope-kernel` (repository: `MiMindMendinc/lyle-rope-kernel-js`)

## What this is

An experimental, zero-dependency **CPU / JavaScript** RoPE (rotary position embedding)
kernel for inference experiments:

- In-place Float32 tensor rotation
- Adjacent-pair and split-half layouts
- Reusable plans with an optional trig cache
- Packed Q/K (`applyRoPEQK`) with unequal head counts

It is **not** a transformer, model loader, attention engine, KV-cache implementation,
GPU kernel, or production inference stack.

## Installation

These steps install from a checkout or a local tarball; see the README for npm status.

### From a git checkout

```sh
git clone https://github.com/MiMindMendinc/lyle-rope-kernel-js.git
cd lyle-rope-kernel-js
# Node >= 22 (engines field). CI covers Node 22 and 24 on Linux and Windows.
npm ci --offline --ignore-scripts --no-audit --no-fund
npm run verify
npm run example
```

### From a packed tarball

```sh
# In the repository checkout:
npm pack --offline --ignore-scripts
# In a fresh project, using the path that npm pack printed:
npm install /absolute/path/to/lyle-rope-kernel-1.1.0-rc.1.tgz \
  --offline --ignore-scripts --no-audit --no-fund
```

`npm run test:package` performs the same steps automatically: it packs the repository,
installs the tarball into a fresh offline project and exercises the public exports.

### Minimal API check after install

```js
import { createRoPEPlan, applyRoPE, applyRoPEQK, DEFAULT_BASE } from 'lyle-rope-kernel';

const headDim = 4;
const plan = createRoPEPlan(headDim, DEFAULT_BASE, { maxSeqLen: 16 });
const tensor = Float32Array.from([1, 0, 0, 1]);
applyRoPE(tensor, headDim, { startPos: 1 });

const q = Float32Array.from([1, 0, 0, 1, 0.5, -0.5, 1, 0]);
const k = Float32Array.from([1, 0, 0.5, 0.5]);
applyRoPEQK(q, k, plan, { qHeads: 2, kvHeads: 1, startPos: 2, layout: 'adjacent' });
```

## Migration notes (from earlier experimental revisions)

- **Recreate plans** after upgrading: `invFreq` is Float64, and plans are module-local
  (WeakSet-authenticated), not serializable or hand-built structures.
- Incomplete rows and unsafe positions that older versions accepted **now throw**.
- The minimum Node version is **22**.
- Public CPU function names and in-place return conventions are unchanged.
- `lyle-rope-kernel/webgpu` remains an explicit **CPU fallback** (empty shader, no device
  request). It is not GPU support.
- The absolute tolerance for bounded fixtures remains **1e-6**. See
  [docs/NUMERICS.md](NUMERICS.md).

## Known limitations

- CPU / JavaScript only: no GPU acceleration, and browser runtimes are not part of CI.
- No end-to-end model inference, no Llama 3 / YaRN / NTK scaling, no model-quality validation.
- `startPos` is an absolute position offset, not a KV-cache subsystem.
- Tensor values are not scanned for finiteness on the hot path.
- Timings from `npm run evidence` and `npm run bench` are machine- and workload-specific
  comparisons within this package, not comparisons with other libraries or GPUs.
- The version in `package.json` is a release candidate, not a registry publication.

## Reporting issues

Open an issue at https://github.com/MiMindMendinc/lyle-rope-kernel-js/issues and include
your Node version, OS, the exact commit SHA, the failing command, and whether you installed
from a checkout or a tarball.
