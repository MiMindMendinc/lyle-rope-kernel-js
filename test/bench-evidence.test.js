import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { MIN_RUNS, aggregateRuns, renderReadmeTables } from '../bench/aggregate.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
// Normalize CRLF: Windows checkouts (core.autocrlf) convert README line endings.
const readme = readFileSync(root + 'README.md', 'utf8').replace(/\r\n/g, '\n');

describe('README benchmark tables match the committed evidence', () => {
  const link = readme.match(/\(https:\/\/github\.com\/MiMindMendinc\/lyle-rope-kernel-js\/blob\/main\/(evidence\/bench-[\d-]+\.json)\)/);
  it('links the aggregated evidence file with an absolute URL (works on npmjs)', () => {
    assert.ok(link, 'README must link evidence/bench-*.json by absolute GitHub URL');
  });
  const evidence = JSON.parse(readFileSync(root + link[1], 'utf8'));

  it('holds at least three clean runs from one source commit', () => {
    assert.equal(evidence.kind, 'throughput-aggregate');
    assert.ok(evidence.runs.length >= MIN_RUNS);
    for (const run of evidence.runs) {
      assert.equal(run.source.dirty, false);
      assert.equal(run.source.commit, evidence.summary.sourceCommit);
      for (const r of run.results) assert.ok(r.maxAbsErrorVsReference <= 1e-6);
    }
  });

  it('stores exactly the summary recomputed from its runs', () => {
    assert.deepEqual(aggregateRuns(evidence.runs), evidence.summary);
  });

  it('README tables, run count and noise sentence are the committed numbers', () => {
    const { summary } = evidence;
    const v = summary.variability;
    assert.ok(readme.includes(renderReadmeTables(summary)), 'README tables differ from the evidence');
    const text = readme.replace(/\s+/g, ' ');
    assert.ok(text.includes(`**${summary.runCount} back-to-back runs**`));
    assert.ok(text.includes(`\`${summary.sourceCommit.slice(0, 12)}\` (clean tree)`));
    // Spreads are quoted as whole percentages; the typical deviation keeps one decimal.
    const pct = value => `${Math.round(value)}%`;
    const slow = v.slowCellsPerRun;
    const slowList = slow.length > 1 ? `${slow.slice(0, -1).join(', ')} and ${slow.at(-1)}` : `${slow[0]}`;
    assert.ok(text.includes(`within ${v.medianTypicalDeviationPercent}% of the cell median`), 'typical deviation');
    assert.ok(text.includes(`min–max span was ${pct(v.medianSpreadPercent)} of its median, ` +
      `up to ${pct(v.maxSpreadPercent)} for ${v.maxSpreadCase}`), 'spread');
    assert.ok(text.includes(`Runs 1–${summary.runCount} had ${slowList} of the ${v.readmeCellCount} cells ` +
      'more than 20% above their median'), 'slow cells per run');
    // No speed-up multiplier claims in the README prose.
    assert.doesNotMatch(readme, /\b\d+(\.\d+)?\s*[x×]\s*(faster|speed)/i);
  });
});
