# Experimental prerelease notes — lyle-rope-kernel 1.1.0-rc.1

**Status labels (honest):** PROPOSED / TESTED LOCALLY (Node 22 on this handoff machine).  
**Not claimed:** CI VERIFIED on the handoff branch until PR checks complete; verified `main`; npm published; GPU/browser execution; model-quality or E2E inference.

Author: Lyle Perrien II  
Package: `lyle-rope-kernel` (repo: `MiMindMendinc/lyle-rope-kernel-js`)  
Source tip tested: `f9b70514446f1821562ad5e552e79df3d394a617` (and any subsequent handoff-branch commits listed in the draft PR)

## What this is

An experimental, zero-dependency **CPU / JavaScript** RoPE (rotary position embedding) kernel for inference experiments:

- In-place Float32 tensor rotation
- Adjacent-pair and split-half layouts
- Reusable plans with optional trig cache
- Packed Q/K (`applyRoPEQK`) with unequal head counts

It is **not** a transformer, model loader, attention engine, KV-cache implementation, GPU kernel, or production inference stack.

## Installation (do not publish yet)

These notes describe local / tarball install only. **Do not** `npm publish` from this handoff.

### From a git checkout

```sh
git clone https://github.com/MiMindMendinc/lyle-rope-kernel-js.git
cd lyle-rope-kernel-js
# Use Node >= 22 (engines field). CI also covers Node 24.
npm ci --offline --ignore-scripts --no-audit --no-fund
npm run verify
npm run example
```

### From a packed tarball (offline consumer)

```sh
# In the package tree:
npm pack --offline --ignore-scripts
# Copy the printed .tgz somewhere durable, then in a fresh project:
npm install /absolute/path/to/lyle-rope-kernel-1.1.0-rc.1.tgz \
  --offline --ignore-scripts --no-audit --no-fund
```

Verified offline install on this handoff used:

```sh
npm install /workspace/showcase-handoff/lyle/lyle-rope-kernel-1.1.0-rc.1.tgz \
  --offline --ignore-scripts --no-audit --no-fund --cache "$TMP/npm-cache"
```

SHA-256 of the handoff tarball: see `lyle-rope-kernel-1.1.0-rc.1.tgz.sha256` next to the artifact.

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

- **Recreate plans** after upgrading: `invFreq` is Float64; plans are module-local (WeakSet-authenticated), not serializable/fabricated structures.
- Incomplete rows and unsafe positions that older versions may have accepted **now throw**.
- Node engines minimum is **22**.
- Public CPU names and in-place return conventions are retained.
- `lyle-rope-kernel/webgpu` remains an explicit **CPU fallback** (empty shader; no device request). Do not treat it as GPU support.
- Numerical absolute tolerance for bounded fixtures remains **1e-6**. See `docs/NUMERICS.md`.

## Known limitations

- CPU / JS only — no GPU acceleration claim; no browser runtime CI coverage.
- No end-to-end model inference, no Llama 3 / YaRN / NTK scaling, no model-quality certification.
- `startPos` is an absolute position offset, not a full KV-cache subsystem.
- Tensor values are not scanned for finiteness on the hot path.
- Performance numbers from `npm run evidence` are machine/workload-specific comparisons **within this package**, not claims vs external libraries or GPUs.
- Release candidate version in `package.json` is **not** an npm registry publication.

## How to report issues

Open an issue at: https://github.com/MiMindMendinc/lyle-rope-kernel-js/issues  

Include Node version, OS, exact commit SHA, failing command, and whether you used checkout vs tarball install.

## Evidence map (local handoff box)

| Command | Exit | Evidence path |
| --- | --- | --- |
| `npm test` | 0 | `/workspace/showcase-handoff/lyle/npm-test.txt` |
| `npm run example` | 0 | `/workspace/showcase-handoff/lyle/npm-run-example.txt` |
| `npm run test:package` | 0 | `/workspace/showcase-handoff/lyle/npm-run-test-package.txt` |
| `npm run evidence` | 0 | `/workspace/showcase-handoff/lyle/npm-run-evidence.txt`, `reports/` |
| `npm run verify` | 0 | `/workspace/showcase-handoff/lyle/npm-run-verify.txt` |
| `npm pack` | 0 | `/workspace/showcase-handoff/lyle/npm-pack.txt` + `.tgz` |
| Offline install + API | 0 | `offline-install.txt`, `offline-api-example.txt` |

**Verdict intent for reviewers:** Ready for code review of experimental package handoff. Not ready to claim published or merged.
