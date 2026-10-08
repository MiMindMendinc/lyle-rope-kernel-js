// Combine several `npm run bench` JSON runs into one evidence file and render the README tables.
//   npm run bench:aggregate -- --out evidence/bench-YYYY-MM-DD.json run1.json run2.json run3.json ...
// Requires >= 3 runs of the full matrix from the same clean source commit and Node version.
// Each README cell is the median of the per-run medians, with the min-max of those per-run
// medians. Node built-ins only; nothing is uploaded.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const MIN_RUNS = 3;
const TOLERANCE = 1e-6;

export function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export const caseKey = r =>
  [r.group, r.api, r.plan, r.headDim, r.seqLen, r.qHeads, r.kvHeads].join('|');

const round = (value, digits = 1) => Math.round(value * 10 ** digits) / 10 ** digits;

export function aggregateRuns(runs) {
  if (runs.length < MIN_RUNS) throw new Error(`need at least ${MIN_RUNS} runs, got ${runs.length}`);
  const first = runs[0];
  for (const [i, run] of runs.entries()) {
    const where = `run ${i + 1}`;
    if (run.kind !== 'throughput') throw new Error(`${where}: not a full throughput run`);
    if (run.source.dirty !== false) throw new Error(`${where}: source tree was not clean`);
    if (run.source.commit !== first.source.commit) throw new Error(`${where}: different source commit`);
    if (run.environment.node !== first.environment.node || run.environment.cpuModel !== first.environment.cpuModel) {
      throw new Error(`${where}: different Node version or CPU`);
    }
    const keys = run.results.map(caseKey).join('\n');
    if (keys !== first.results.map(caseKey).join('\n')) throw new Error(`${where}: different case matrix`);
    for (const r of run.results) {
      if (!(r.maxAbsErrorVsReference <= TOLERANCE)) throw new Error(`${where}: ${caseKey(r)} failed the reference check`);
    }
  }
  const cases = first.results.map((r, index) => {
    const perRun = runs.map(run => run.results[index].medianNsPerPair);
    const med = median(perRun);
    const min = Math.min(...perRun), max = Math.max(...perRun);
    return {
      group: r.group, api: r.api, layout: r.layout, plan: r.plan, headDim: r.headDim, seqLen: r.seqLen,
      qHeads: r.qHeads, kvHeads: r.kvHeads,
      perRunMedianNsPerPair: perRun,
      medianOfRunMediansNsPerPair: med, minRunMedianNsPerPair: min, maxRunMedianNsPerPair: max,
      spreadPercentOfMedian: (max - min) / med * 100,
      maxAbsErrorVsReference: Math.max(...runs.map(run => run.results[index].maxAbsErrorVsReference)),
    };
  });
  const shown = cases.filter(c => c.group !== 'oracle');
  const spreads = shown.map(c => c.spreadPercentOfMedian);
  const worst = shown.reduce((a, b) => (b.spreadPercentOfMedian > a.spreadPercentOfMedian ? b : a));
  // Typical deviation: per cell, the median over runs of |run - cell median| / cell median.
  const typical = shown.map(c => median(c.perRunMedianNsPerPair.map(v => Math.abs(v - c.medianOfRunMediansNsPerPair)))
    / c.medianOfRunMediansNsPerPair * 100);
  const slowCellsPerRun = runs.map((_, r) => shown.filter(c =>
    c.perRunMedianNsPerPair[r] > 1.2 * c.medianOfRunMediansNsPerPair).length);
  return {
    runCount: runs.length,
    sourceCommit: first.source.commit,
    allRunsCleanTree: true,
    node: first.environment.node,
    cpuModel: first.environment.cpuModel,
    statistic: 'per cell: median of the per-run medians; range = min-max of the per-run medians',
    variability: {
      definition: '(max - min) / median of the per-run medians, in percent, over README cells (oracle excluded)',
      medianSpreadPercent: round(median(spreads)),
      maxSpreadPercent: round(worst.spreadPercentOfMedian),
      maxSpreadCase: label(worst),
      medianTypicalDeviationPercent: round(median(typical)),
      slowCellsPerRunDefinition: 'per run, README cells more than 20% above the cell median',
      slowCellsPerRun,
      readmeCellCount: shown.length,
    },
    cases,
  };
}

function label(c) {
  const shape = c.group === 'packed' ? `[${c.seqLen}, ${c.qHeads}/${c.kvHeads}, ${c.headDim}]` : `[${c.seqLen}, ${c.headDim}]`;
  return `${shape} ${c.api.split(' ')[0]}, ${c.plan}`;
}

export const fmt = value => (value >= 100 ? value.toFixed(0) : value >= 10 ? value.toFixed(1) : value.toFixed(2));
const cellText = c => `${fmt(c.medianOfRunMediansNsPerPair)} (${fmt(c.minRunMedianNsPerPair)}–${fmt(c.maxRunMedianNsPerPair)})`;

// Markdown for the two README tables; test/bench-evidence.test.js checks the README matches.
export function renderReadmeTables(summary) {
  const find = (group, filter) => {
    const hit = summary.cases.find(c => c.group === group && filter(c));
    if (!hit) throw new Error('missing case');
    return cellText(hit);
  };
  const single = summary.cases.filter(c => c.group === 'single' && c.api === 'applyRoPEWithPlan' && c.plan === 'frequencies only');
  const lines = [
    '| Shape [seq, headDim] | applyRoPEWithPlan, no cache | applyRoPEWithPlan, cached | applyRoPESplitHalfWithPlan, cached | Plain in-place scalar loop (baseline, not this package) |',
    '| --- | ---: | ---: | ---: | ---: |',
  ];
  for (const { headDim, seqLen } of single) {
    const at = (api, plan) => find('single', c => c.headDim === headDim && c.seqLen === seqLen &&
      c.api.startsWith(api) && c.plan === plan);
    lines.push(`| [${seqLen}, ${headDim}] | ${at('applyRoPEWithPlan', 'frequencies only')} | ` +
      `${at('applyRoPEWithPlan', 'cached trig table')} | ${at('applyRoPESplitHalfWithPlan', 'cached trig table')} | ` +
      `${at('plainScalarRoPEInPlace', 'none')} |`);
  }
  lines.push('', '| Packed Q/K shape [seq, Q heads / KV heads, headDim] | applyRoPEQK, no cache | applyRoPEQK, cached |',
    '| --- | ---: | ---: |');
  const packed = summary.cases.filter(c => c.group === 'packed' && c.plan === 'frequencies only');
  for (const { headDim, seqLen, qHeads, kvHeads } of packed) {
    const at = plan => find('packed', c => c.headDim === headDim && c.seqLen === seqLen &&
      c.qHeads === qHeads && c.kvHeads === kvHeads && c.plan === plan);
    lines.push(`| [${seqLen}, ${qHeads}/${kvHeads}, ${headDim}] | ${at('frequencies only')} | ${at('cached trig table')} |`);
  }
  return lines.join('\n');
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const args = process.argv.slice(2);
  const outIndex = args.indexOf('--out');
  if (outIndex < 0 || !args[outIndex + 1]) throw new Error('Usage: npm run bench:aggregate -- --out out.json run1.json run2.json run3.json ...');
  const out = resolve(args[outIndex + 1]);
  const files = args.filter((_, i) => i !== outIndex && i !== outIndex + 1);
  const runs = files.map(file => JSON.parse(readFileSync(file, 'utf8')));
  const summary = aggregateRuns(runs);
  const evidence = { schemaVersion: 1, kind: 'throughput-aggregate', summary, runs };
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, JSON.stringify(evidence) + '\n');
  console.log(renderReadmeTables(summary));
  console.log('\n' + JSON.stringify(summary.variability, null, 2));
  console.log(`\nWrote ${out} (${runs.length} runs, commit ${summary.sourceCommit})`);
}
