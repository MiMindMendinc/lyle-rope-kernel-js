// Local evidence only: no uploads, telemetry, hostnames, usernames or environment dumps.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from 'node:fs';
import { cpus, platform, release, arch, availableParallelism } from 'node:os';
import { dirname, resolve, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { applyRoPEQK, applyRoPEWithPlan, applyRoPESplitHalfWithPlan, createRoPEPlan } from '../src/rope-kernel.js';
import { referenceRoPE, fixture, maxAbsError } from '../support/reference.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const args = process.argv.slice(2);
if (args.length && (args.length !== 2 || args[0] !== '--output')) {
  throw new Error('Usage: npm run evidence -- [--output reports/local.json]');
}
const output = resolve(root, args[1] ?? 'reports/local.json');
mkdirSync(dirname(output), { recursive: true });
const hash = value => createHash('sha256').update(value).digest('hex');
const child = (command, argv) => spawnSync(command, argv, {
  cwd: root, encoding: 'utf8', timeout: 120000, maxBuffer: 8 * 1024 * 1024,
});
const git = argv => {
  const result = child('git', argv);
  return result.status === 0 ? result.stdout.trim() : null;
};
function sourceFiles() {
  const files = [];
  function walk(folder) {
    for (const entry of readdirSync(resolve(root, folder), { withFileTypes: true })) {
      const name = folder + '/' + entry.name;
      if (entry.isDirectory()) walk(name);
      else if (entry.isFile()) files.push(name);
    }
  }
  for (const folder of ['src', 'test', 'support', 'scripts', 'bench', 'examples']) walk(folder);
  files.push('package.json', 'package-lock.json');
  return Object.fromEntries(files.sort().map(name => [name, hash(readFileSync(resolve(root, name)))]));
}
const status = git(['status', '--porcelain']);
const report = {
  schemaVersion: 1, status: 'running', generatedAt: new Date().toISOString(),
  source: { commit: git(['rev-parse', 'HEAD']), dirty: status === null ? null : status !== '',
    fileSha256: sourceFiles() },
  environment: { node: process.version, v8: process.versions.v8, platform: platform(),
    osRelease: release(), arch: arch(), cpuModel: cpus()[0]?.model ?? 'unknown',
    logicalCpus: cpus().length, availableParallelism: availableParallelism(),
    qualification: 'Reported environment only; dedicated hardware is not established.' },
};
function numericalCheck() {
  const results = [];
  for (const layout of ['adjacent', 'split-half']) {
    let worst = 0, cases = 0;
    const apply = layout === 'adjacent' ? applyRoPEWithPlan : applyRoPESplitHalfWithPlan;
    for (const headDim of [2, 4, 8, 64, 96, 128, 256]) {
      for (const base of [1, 10000, 500000]) {
        const plans = [createRoPEPlan(headDim, base), createRoPEPlan(headDim, base, { maxSeqLen: 8193 })];
        for (const startPos of [0, 1, 17, 1024, 8191, 8192, 32768, 65536, 131069]) {
          const input = fixture(3 * headDim, startPos + headDim);
          const expected = referenceRoPE(input, headDim, { startPos, base, layout });
          for (const plan of plans) {
            const actual = new Float32Array(input);
            apply(actual, plan, { startPos });
            worst = Math.max(worst, maxAbsError(actual, expected));
            cases++;
          }
        }
      }
    }
    assert.ok(worst <= 1e-6, `numerical contract failed: ${layout} ${worst}`);
    results.push({ layout, cases, maxAbsError: worst, tolerance: 1e-6 });
  }
  return { inputs: 'deterministic Float32 values in [-1, 1]',
    positions: 'sampled 3-token windows, absolute positions up to 131071; not exhaustive', results };
}
function benchmark(seqLen, layout) {
  const headDim = 128, qHeads = 32, kvHeads = 8, startPos = 1024;
  const qSeed = fixture(seqLen * qHeads * headDim, 11);
  const kSeed = fixture(seqLen * kvHeads * headDim, 29);
  let t = performance.now();
  const plan = createRoPEPlan(headDim);
  const frequencyPlanMs = performance.now() - t;
  t = performance.now();
  const cached = createRoPEPlan(headDim, 10000, { maxSeqLen: startPos + seqLen });
  const cachedPlanMs = performance.now() - t;
  const apply = layout === 'adjacent' ? applyRoPEWithPlan : applyRoPESplitHalfWithPlan;
  const modes = {
    perHead() {
      const q = new Float32Array(qSeed), k = new Float32Array(kSeed);
      for (const [tensor, heads] of [[q, qHeads], [k, kvHeads]]) {
        for (let token = 0; token < seqLen; token++) {
          for (let head = 0; head < heads; head++) {
            const offset = (token * heads + head) * headDim;
            apply(tensor.subarray(offset, offset + headDim), plan, { startPos: startPos + token });
          }
        }
      }
      return { q, k };
    },
    packed() {
      return applyRoPEQK(new Float32Array(qSeed), new Float32Array(kSeed), plan,
        { qHeads, kvHeads, startPos, layout });
    },
    packedCached() {
      return applyRoPEQK(new Float32Array(qSeed), new Float32Array(kSeed), cached,
        { qHeads, kvHeads, startPos, layout });
    },
  };
  const eq = referenceRoPE(qSeed, headDim, { startPos, heads: qHeads, layout });
  const ek = referenceRoPE(kSeed, headDim, { startPos, heads: kvHeads, layout });
  const names = Object.keys(modes), warmups = 10, samples = 7;
  const iterations = seqLen === 1 ? 100 : 20;
  let checksum = 0;
  const consume = name => { const { q, k } = modes[name](); checksum += q[q.length - 1] + k[k.length - 1]; };
  for (const name of names) {
    const { q, k } = modes[name]();
    assert.ok(maxAbsError(q, eq) <= 1e-6 && maxAbsError(k, ek) <= 1e-6, name);
    for (let i = 0; i < warmups; i++) consume(name);
  }
  const timings = Object.fromEntries(names.map(name => [name, []]));
  for (let sample = 0; sample < samples; sample++) {
    // Rotate mode order across samples rather than always timing the baseline first.
    for (let index = 0; index < names.length; index++) {
      const name = names[(sample + index) % names.length];
      const begin = performance.now();
      for (let i = 0; i < iterations; i++) consume(name);
      timings[name].push((performance.now() - begin) / iterations);
    }
  }
  assert.ok(Number.isFinite(checksum));
  return { headDim, seqLen, qHeads, kvHeads, startPos, layout, warmups, samples, iterations,
    method: 'All modes include fresh Q/K copies and checksum consumption. Plan construction excluded. No external library or GPU baseline.',
    plans: { frequencyPlanMs, cachedPlanMs, frequencyBytes: plan.invFreq.byteLength,
      cachedFrequencyBytes: cached.invFreq.byteLength,
      cachedTrigBytes: cached.cosTable.byteLength + cached.sinTable.byteLength,
      cacheMaxSeqLen: cached.maxSeqLen },
    timing: Object.fromEntries(names.map(name => {
      const sorted = [...timings[name]].sort((a, b) => a - b);
      return [name, { rawMsPerRun: timings[name], medianMs: sorted[3], minMs: sorted[0], maxMs: sorted[6] }];
    })), checksum };
}
try {
  const tests = child(process.execPath, ['--test', '--test-reporter=tap']);
  const tap = (tests.stdout ?? '') + (tests.stderr ?? '');
  writeFileSync(output + '.tap', tap);
  report.tests = { command: 'node --test --test-reporter=tap', exitCode: tests.status,
    tapSha256: hash(tap), tapFile: relative(dirname(output), output + '.tap') };
  for (const key of ['tests', 'suites', 'pass', 'fail', 'cancelled', 'skipped', 'todo']) {
    report.tests[key] = Number(tap.match(new RegExp(`^# ${key} (\\d+)$`, 'm'))?.[1] ?? NaN);
  }
  assert.equal(tests.status, 0, tests.error?.message ?? 'correctness suite failed');
  assert.ok(report.tests.tests > 0 && report.tests.pass === report.tests.tests);
  assert.equal(report.tests.skipped, 0);
  const packed = child(process.execPath, ['scripts/package-smoke.mjs']);
  assert.equal(packed.status, 0, (packed.stderr ?? '') || packed.error?.message || 'package smoke failed');
  report.package = JSON.parse(packed.stdout);
  report.numerics = numericalCheck();
  report.benchmarks = [];
  for (const layout of ['adjacent', 'split-half']) {
    for (const seqLen of [1, 128]) report.benchmarks.push(benchmark(seqLen, layout));
  }
  report.status = 'pass';
} catch (error) {
  report.status = 'fail';
  report.error = String(error.message);
  process.exitCode = 1;
} finally {
  writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ status: report.status, report: relative(root, output),
    tests: report.tests, error: report.error }, null, 2));
}
