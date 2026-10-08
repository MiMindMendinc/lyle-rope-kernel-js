// Throughput benchmark for the exported CPU kernels. Node built-ins only.
// Local measurement only: no uploads, telemetry, hostnames, usernames or env dumps.
//
//   npm run bench                         # full matrix, JSON to reports/ (git-ignored)
//   npm run bench -- --out evidence/x.json
//   npm run bench -- --smoke              # tiny run used by the test suite; no file written
//
// Every case is checked against the independent scalar reference in
// support/reference.mjs before it is timed. Timings are never asserted.
import { spawnSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { arch, cpus, loadavg, platform, release, totalmem } from 'node:os';
import { dirname, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import {
  applyRoPEQK, applyRoPESplitHalfWithPlan, applyRoPEWithPlan, createRoPEPlan,
} from '../src/rope-kernel.js';
import { fixture, maxAbsError, referenceRoPE } from '../support/reference.mjs';
import { plainScalarRoPEInPlace } from './baseline.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const args = process.argv.slice(2);
const smoke = args.includes('--smoke');
const outIndex = args.indexOf('--out');
for (const [i, arg] of args.entries()) {
  if (arg !== '--smoke' && arg !== '--out' && !(outIndex >= 0 && i === outIndex + 1)) {
    throw new Error('Usage: npm run bench -- [--smoke] [--out path.json]');
  }
}
if (outIndex >= 0 && !args[outIndex + 1]) throw new Error('--out needs a path');

const TOLERANCE = 1e-6; // Same absolute tolerance as the correctness tests.
const BASE = 10000;
const settings = smoke
  ? { samples: 3, sampleMs: 2, warmupMs: 5 }
  : { samples: 11, sampleMs: 40, warmupMs: 150 };

function localDate(date = new Date()) {
  const pad = value => String(value).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function git(argv) {
  const result = spawnSync('git', argv, { cwd: root, encoding: 'utf8', timeout: 10000 });
  return result.status === 0 ? result.stdout.trim() : null;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Warm up, pick an iteration count that fills sampleMs, then take independent samples.
function measure(fn) {
  const warmupEnd = performance.now() + settings.warmupMs;
  let warmupRuns = 0;
  while (performance.now() < warmupEnd || warmupRuns < 3) { fn(); warmupRuns++; }
  let iterations = 1;
  for (;;) {
    const start = performance.now();
    for (let i = 0; i < iterations; i++) fn();
    const elapsed = performance.now() - start;
    if (elapsed >= settings.sampleMs || iterations >= 1 << 20) break;
    iterations = Math.min(1 << 20, Math.max(iterations * 2,
      Math.ceil(iterations * settings.sampleMs / Math.max(elapsed, 0.001))));
  }
  const nsPerCall = [];
  for (let s = 0; s < settings.samples; s++) {
    const start = performance.now();
    for (let i = 0; i < iterations; i++) fn();
    nsPerCall.push(((performance.now() - start) * 1e6) / iterations);
  }
  return { iterations, nsPerCall };
}

function summarize(timing, pairs) {
  const nsPerPair = timing.nsPerCall.map(ns => ns / pairs);
  const med = median(nsPerPair);
  return {
    iterationsPerSample: timing.iterations,
    medianNsPerCall: median(timing.nsPerCall),
    medianNsPerPair: med,
    minNsPerPair: Math.min(...nsPerPair),
    maxNsPerPair: Math.max(...nsPerPair),
    medianMillionPairsPerSecond: 1e3 / med,
    samplesNsPerCall: timing.nsPerCall,
  };
}

function check(condition, message) {
  if (!condition) throw new Error('Correctness check failed: ' + message);
}

const singleShapes = smoke
  ? [{ headDim: 8, seqLen: 16 }]
  : [64, 128].flatMap(headDim => [128, 512, 2048].map(seqLen => ({ headDim, seqLen })));
const packedShapes = smoke
  ? [{ headDim: 8, seqLen: 4, qHeads: 2, kvHeads: 1 }]
  : [64, 128].flatMap(headDim => [128, 512, 2048].flatMap(seqLen => [
    { headDim, seqLen, qHeads: 8, kvHeads: 8 },
    { headDim, seqLen, qHeads: 32, kvHeads: 8 },
  ]));

const results = [];
let seed = 1;

for (const { headDim, seqLen } of singleShapes) {
  const input = fixture(seqLen * headDim, seed++);
  const pairs = seqLen * headDim / 2;
  const mathPlan = createRoPEPlan(headDim, BASE);
  const planStart = performance.now();
  const cachedPlan = createRoPEPlan(headDim, BASE, { maxSeqLen: seqLen });
  const cachedPlanBuildMs = performance.now() - planStart;
  const expected = {
    adjacent: referenceRoPE(input, headDim, { base: BASE }),
    'split-half': referenceRoPE(input, headDim, { base: BASE, layout: 'split-half' }),
  };
  const scratch = new Float64Array(headDim / 2);
  const cases = [
    { api: 'applyRoPEWithPlan', layout: 'adjacent', plan: 'frequencies only', run: t => applyRoPEWithPlan(t, mathPlan) },
    { api: 'applyRoPEWithPlan', layout: 'adjacent', plan: 'cached trig table', run: t => applyRoPEWithPlan(t, cachedPlan) },
    { api: 'applyRoPESplitHalfWithPlan', layout: 'split-half', plan: 'cached trig table', run: t => applyRoPESplitHalfWithPlan(t, cachedPlan) },
    // Not this package: a plain in-place scalar loop (bench/baseline.mjs), checked like the kernels.
    { api: 'plainScalarRoPEInPlace (baseline, not this package)', layout: 'adjacent', plan: 'none',
      run: t => plainScalarRoPEInPlace(t, headDim, BASE, scratch) },
  ];
  for (const c of cases) {
    const error = maxAbsError(c.run(new Float32Array(input)), expected[c.layout]);
    check(error <= TOLERANCE, `${c.api} ${c.plan} d=${headDim} seq=${seqLen}: ${error}`);
    // Timed region: in-place rotation of one reused buffer; no copies, no plan creation.
    const buffer = new Float32Array(input);
    results.push({ group: 'single', api: c.api, layout: c.layout, plan: c.plan, headDim, seqLen,
      qHeads: 1, kvHeads: 0, rotatedPairsPerCall: pairs, maxAbsErrorVsReference: error,
      ...(c.plan === 'cached trig table' ? { cachedPlanBuildMs } : {}),
      ...summarize(measure(() => c.run(buffer)), pairs) });
  }
  // Correctness oracle cost only: it allocates its output and recomputes Math.pow per element,
  // so it is not a performance baseline and is not shown in the README tables.
  results.push({ group: 'oracle', api: 'referenceRoPE (correctness oracle; allocates; not a baseline)',
    layout: 'adjacent', plan: 'none', headDim, seqLen, qHeads: 1, kvHeads: 0,
    rotatedPairsPerCall: pairs, maxAbsErrorVsReference: 0,
    ...summarize(measure(() => referenceRoPE(input, headDim, { base: BASE })), pairs) });
}

for (const { headDim, seqLen, qHeads, kvHeads } of packedShapes) {
  const qInput = fixture(seqLen * qHeads * headDim, seed++);
  const kInput = fixture(seqLen * kvHeads * headDim, seed++);
  const pairs = seqLen * (qHeads + kvHeads) * headDim / 2;
  const mathPlan = createRoPEPlan(headDim, BASE);
  const cachedPlan = createRoPEPlan(headDim, BASE, { maxSeqLen: seqLen });
  const qExpected = referenceRoPE(qInput, headDim, { base: BASE, heads: qHeads });
  const kExpected = referenceRoPE(kInput, headDim, { base: BASE, heads: kvHeads });
  for (const [label, plan] of [['frequencies only', mathPlan], ['cached trig table', cachedPlan]]) {
    const options = { qHeads, kvHeads };
    const { q, k } = applyRoPEQK(new Float32Array(qInput), new Float32Array(kInput), plan, options);
    const error = Math.max(maxAbsError(q, qExpected), maxAbsError(k, kExpected));
    check(error <= TOLERANCE, `applyRoPEQK ${label} d=${headDim} seq=${seqLen} q=${qHeads} kv=${kvHeads}: ${error}`);
    const qBuffer = new Float32Array(qInput), kBuffer = new Float32Array(kInput);
    results.push({ group: 'packed', api: 'applyRoPEQK', layout: 'adjacent', plan: label,
      headDim, seqLen, qHeads, kvHeads, rotatedPairsPerCall: pairs, maxAbsErrorVsReference: error,
      ...summarize(measure(() => applyRoPEQK(qBuffer, kBuffer, plan, options)), pairs) });
  }
}

const status = git(['status', '--porcelain']);
const report = {
  schemaVersion: 1,
  kind: smoke ? 'smoke' : 'throughput',
  generatedAt: new Date().toISOString(),
  source: { commit: git(['rev-parse', 'HEAD']), dirty: status === null ? null : status !== '' },
  environment: {
    node: process.version, v8: process.versions.v8, platform: platform(), osRelease: release(),
    arch: arch(), cpuModel: cpus()[0]?.model ?? 'unknown', logicalCpus: cpus().length,
    totalMemoryGiB: Math.round(totalmem() / 2 ** 30 * 10) / 10,
    loadAverageAtStart: loadavg().map(value => Math.round(value * 100) / 100),
  },
  method: {
    timer: 'node:perf_hooks performance.now()',
    warmupMsPerCase: settings.warmupMs, targetMsPerSample: settings.sampleMs,
    samplesPerCase: settings.samples, statistic: 'median of samples',
    unit: 'ns per rotated (x0, x1) pair; packed counts Q and K pairs',
    timedRegion: 'in-place rotation of one reused Float32Array buffer from startPos 0; ' +
      'plan creation and input copies are outside the timed region (position 0 is the identity and is skipped by the kernel)',
    baseline: 'bench/baseline.mjs plainScalarRoPEInPlace: a straightforward in-place scalar loop (not this ' +
      'package): inverse frequencies once per call into a reused scratch array, Math.cos/Math.sin per pair, ' +
      'no output allocation; reference-checked like the kernel cases',
    oracle: 'group "oracle" times support/reference.mjs referenceRoPE, the correctness oracle; it allocates a ' +
      'new output array and recomputes Math.pow per element, so it is recorded for completeness only and is ' +
      'not a performance baseline',
    correctness: `every timed kernel case matched the reference within ${TOLERANCE} absolute before timing`,
    base: BASE,
    caveat: 'One machine; this file is one run. Indicative only; not a comparison with other libraries, GPUs or models.',
  },
  results,
};

const fmt = value => value >= 100 ? value.toFixed(0) : value >= 10 ? value.toFixed(1) : value.toFixed(2);
const lines = [
  `Node ${report.environment.node}, ${report.environment.cpuModel} (${report.environment.logicalCpus} logical CPUs), ` +
    `${report.environment.platform} ${report.environment.osRelease} ${report.environment.arch}`,
  '',
  '| Shape [seq, headDim] | applyRoPEWithPlan, no cache | applyRoPEWithPlan, cached | applyRoPESplitHalfWithPlan, cached | Plain in-place scalar loop (baseline) |',
  '| --- | ---: | ---: | ---: | ---: |',
];
for (const { headDim, seqLen } of singleShapes) {
  const cell = (api, plan) => {
    const r = results.find(x => x.group === 'single' && x.headDim === headDim &&
      x.seqLen === seqLen && x.api.startsWith(api) && x.plan === plan);
    return fmt(r.medianNsPerPair);
  };
  lines.push(`| [${seqLen}, ${headDim}] | ${cell('applyRoPEWithPlan', 'frequencies only')} | ` +
    `${cell('applyRoPEWithPlan', 'cached trig table')} | ${cell('applyRoPESplitHalfWithPlan', 'cached trig table')} | ` +
    `${cell('plainScalarRoPEInPlace', 'none')} |`);
}
lines.push('', '| Packed Q/K shape [seq, Q heads / KV heads, headDim] | applyRoPEQK, no cache | applyRoPEQK, cached |',
  '| --- | ---: | ---: |');
for (const { headDim, seqLen, qHeads, kvHeads } of packedShapes) {
  const cell = plan => fmt(results.find(x => x.group === 'packed' && x.headDim === headDim &&
    x.seqLen === seqLen && x.qHeads === qHeads && x.kvHeads === kvHeads && x.plan === plan).medianNsPerPair);
  lines.push(`| [${seqLen}, ${qHeads}/${kvHeads}, ${headDim}] | ${cell('frequencies only')} | ${cell('cached trig table')} |`);
}
lines.push('', 'Median nanoseconds per rotated (x0, x1) pair; lower means less time per pair. One machine, one run; indicative only.',
  'For the README, combine several runs with `npm run bench:aggregate` (median and min-max across runs).');

if (smoke) {
  process.stdout.write(JSON.stringify(report) + '\n');
} else {
  const output = resolve(root, outIndex >= 0 ? args[outIndex + 1] : `reports/bench-${localDate()}.json`);
  mkdirSync(dirname(output), { recursive: true });
  writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
  console.log(lines.join('\n'));
  console.log(`\nWrote ${output}`);
}
