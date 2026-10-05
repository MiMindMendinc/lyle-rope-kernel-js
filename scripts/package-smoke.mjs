// Install the actual tarball in a fresh consumer with network and lifecycle scripts disabled.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const npmCli = process.env.npm_execpath;
if (!npmCli) throw new Error('Run this check with npm run test:package');
const temp = mkdtempSync(join(tmpdir(), 'rope-package-'));
const npm = (args, cwd) => execFileSync(process.execPath, [npmCli, ...args,
  '--offline', '--ignore-scripts', '--no-audit', '--no-fund', '--cache', join(temp, 'cache')], {
  cwd, encoding: 'utf8', timeout: 60000, stdio: ['ignore', 'pipe', 'pipe'],
});
try {
  const [packed] = JSON.parse(npm(['pack', '--json', '--pack-destination', temp], root));
  const names = packed.files.map(file => file.path).sort();
  assert.deepEqual(names, ['CHANGELOG.md', 'LICENSE', 'README.md', 'docs/NUMERICS.md',
    'package.json', 'src/rope-kernel.js', 'src/webgpu-rope.js'].sort());
  const archive = resolve(temp, packed.filename);
  const consumer = join(temp, 'consumer');
  mkdirSync(consumer);
  writeFileSync(join(consumer, 'package.json'), JSON.stringify({ private: true, type: 'module' }));
  npm(['install', '--save=false', '--package-lock=false', archive], consumer);
  writeFileSync(join(consumer, 'probe.mjs'), `
import assert from 'node:assert/strict';
import { lstatSync } from 'node:fs';
import * as rope from 'lyle-rope-kernel';
import { applyRoPEWebGPU, WEBGPU_ROPE_STATUS, WGSL_ROPE_SHADER } from 'lyle-rope-kernel/webgpu';
assert.equal(lstatSync('node_modules/lyle-rope-kernel').isSymbolicLink(), false);
for (const name of ['applyRoPE', 'applyRoPESplitHalf', 'applyRoPEWithPlan',
  'applyRoPESplitHalfWithPlan', 'applyRoPEQK', 'applyToHead', 'createRoPEPlan',
  'verifyNormPreservation']) assert.equal(typeof rope[name], 'function', name);
assert.equal(rope.DEFAULT_BASE, 10000);
assert.ok(rope.createRoPEPlan(4).invFreq instanceof Float64Array);
const input = Float32Array.from([1, 2, 3, 4]);
const gpuFallback = new Float32Array(input);
rope.applyRoPE(input, 4, { startPos: 1 });
const expected = [-1.1426396637, 1.9220755965, 2.9598506689, 4.0297995008];
for (let i = 0; i < 4; i++) assert.ok(Math.abs(input[i] - expected[i]) <= 1e-6);
assert.throws(() => rope.applyRoPE(new Float32Array(4), 4, { startPos: 2 ** 53 }), RangeError);
assert.equal(WEBGPU_ROPE_STATUS, 'preview-fallback');
assert.equal(WGSL_ROPE_SHADER, '');
applyRoPEWebGPU(gpuFallback, 4, { startPos: 1 });
assert.deepEqual(gpuFallback, input);
console.log('PASS: installed package exports, numerical fixture, range guard and CPU fallback');
`);
  const probe = execFileSync(process.execPath, ['probe.mjs'], {
    cwd: consumer, encoding: 'utf8', timeout: 30000,
  }).trim();
  console.log(JSON.stringify({ status: 'pass', version: packed.version, files: names,
    tarballSha256: createHash('sha256').update(readFileSync(archive)).digest('hex'),
    packedBytes: packed.size, unpackedBytes: packed.unpackedSize,
    install: 'fresh tarball consumer; offline; fresh npm cache; lifecycle scripts disabled', probe,
  }, null, 2));
} finally {
  rmSync(temp, { recursive: true, force: true });
}
