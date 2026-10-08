import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, readdirSync, existsSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import * as kernel from '../src/rope-kernel.js';
import * as reference from '../support/reference.mjs';
import {
  FIELD_LABELS, LIMITS, NUMBER_FIELDS, TOLERANCE, configFromFields, cosineGrid, maxStartPosFor, pairAt,
  parseNumberField, runCheck, seededInput, timeKernel, validateConfig, validateConfigFields,
} from '../demo/playground-core.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const FIELD_IDS_HAVE_LABELS = field => typeof FIELD_LABELS[field] === 'string';
const base = { api: 'single', layout: 'adjacent', headDim: 64, seqLen: 33, startPos: 0,
  base: 10000, cached: false, seed: 7, qHeads: 4, kvHeads: 2 };

describe('browser demo core (runs the same logic as the page)', () => {
  it('matches the scalar reference for both APIs, layouts, plans and position ranges', () => {
    for (const api of ['single', 'packed']) {
      for (const layout of ['adjacent', 'split-half']) {
        for (const cached of [false, true]) {
          for (const startPos of [0, 1, 8191, LIMITS.maxStartPos - 40, maxStartPosFor(base.seqLen)]) {
            const cfg = { ...base, api, layout, cached, startPos };
            const result = runCheck(kernel, reference, cfg);
            assert.equal(result.pass, true, JSON.stringify(cfg));
            assert.ok(result.maxAbsError <= TOLERANCE);
            assert.ok(result.relativeNormChange < 1e-5);
          }
        }
      }
    }
  });

  it('reports a failure when the kernel output is wrong', () => {
    const broken = { ...kernel, applyRoPEWithPlan: tensor => { tensor[3] += 0.01; return tensor; } };
    const result = runCheck(broken, reference, base);
    assert.equal(result.pass, false);
    assert.ok(result.maxAbsError > TOLERANCE);
  });

  it('generates deterministic synthetic data in [-1, 1)', () => {
    const a = seededInput(4096, 42);
    assert.deepEqual(a, seededInput(4096, 42));
    assert.notDeepEqual(a, seededInput(4096, 43));
    for (const value of a) assert.ok(value >= -1 && value < 1);
  });

  it('rejects out-of-range configurations before allocating', () => {
    assert.deepEqual(validateConfig(base), []);
    assert.ok(validateConfig({ ...base, headDim: 6 }).length);
    assert.ok(validateConfig({ ...base, seqLen: 0 }).length);
    assert.ok(validateConfig({ ...base, seqLen: LIMITS.maxSeqLen + 1 }).length);
    assert.ok(validateConfig({ ...base, startPos: LIMITS.maxStartPos + 1 }).length);
    assert.ok(validateConfig({ ...base, api: 'packed', qHeads: 0 }).length);
    assert.ok(validateConfig({ ...base, api: 'packed', seqLen: 4096, qHeads: 64, headDim: 256 }).length);
    assert.ok(validateConfig({ ...base, cached: true, startPos: 131071, headDim: 256 }).length);
    assert.throws(() => runCheck(kernel, reference, { ...base, layout: 'x' }), RangeError);
  });

  it('treats empty or non-numeric form fields as validation errors, not as 0', () => {
    assert.ok(Number.isNaN(parseNumberField('')));
    assert.ok(Number.isNaN(parseNumberField('   ')));
    assert.ok(Number.isNaN(parseNumberField('abc')));
    assert.ok(Number.isNaN(parseNumberField(undefined)));
    assert.equal(parseNumberField('0'), 0);
    assert.equal(parseNumberField(' 42 '), 42);
    const fields = { api: 'single', layout: 'adjacent', headDim: '64', cached: false,
      seqLen: '33', startPos: '0', base: '10000', qHeads: '4', kvHeads: '2', seed: '7' };
    assert.deepEqual(configFromFields(fields), base);
    assert.deepEqual(validateConfig(configFromFields(fields)), []);
    for (const name of NUMBER_FIELDS) {
      const api = name === 'qHeads' || name === 'kvHeads' ? 'packed' : 'single';
      for (const blank of ['', '  ', 'x']) {
        const errors = validateConfig(configFromFields({ ...fields, api, [name]: blank }));
        assert.equal(errors.length, 1, `${name}=${JSON.stringify(blank)}: ${errors}`);
        assert.ok(errors[0].startsWith(`${FIELD_LABELS[name]} is empty or not a number`), errors[0]);
        assert.equal(validateConfigFields(configFromFields({ ...fields, api, [name]: blank }))[0].field, name);
        assert.throws(() => runCheck(kernel, reference, configFromFields({ ...fields, api, [name]: blank })),
          RangeError);
      }
    }
    // A literal 0 is still a valid startPos and seed.
    assert.deepEqual(validateConfig(configFromFields({ ...fields, startPos: '0', seed: '0' })), []);
  });

  it('names the field behind each error, with capitalized sentence-style messages', () => {
    const cache = validateConfigFields({ ...base, cached: true, startPos: 131071 - 32, headDim: 256 });
    assert.deepEqual(cache.map(error => error.field), ['cached']);
    assert.match(cache[0].message, /^Cached plan would exceed 64 MiB/);
    const position = validateConfigFields({ ...base, seqLen: 4096, startPos: 131071 });
    assert.deepEqual(position.map(error => error.field), ['startPos']);
    const many = validateConfigFields({ ...base, seqLen: 0, base: 1, headDim: 6 });
    assert.deepEqual(many.map(error => error.field).sort(), ['base', 'headDim', 'seqLen']);
    for (const { field, message } of many) {
      assert.match(message, /^[A-Z]/, message);
      assert.ok(!message.endsWith('.'), message);
      assert.ok(FIELD_IDS_HAVE_LABELS(field));
    }
  });

  it('keeps every rotated position within the range covered by the numerical tests', () => {
    assert.equal(LIMITS.maxPosition, 131071);
    assert.equal(maxStartPosFor(1), 131071);
    assert.equal(maxStartPosFor(4096), 131071 - 4095);
    assert.deepEqual(validateConfig({ ...base, seqLen: 1, startPos: 131071 }), []);
    assert.ok(validateConfig({ ...base, seqLen: 1, startPos: 131072 }).length);
    for (const seqLen of [2, 72, 513, LIMITS.maxSeqLen]) {
      const highest = { ...base, seqLen, startPos: maxStartPosFor(seqLen) };
      assert.deepEqual(validateConfig(highest), [], `seqLen ${seqLen}`);
      const over = validateConfig({ ...highest, startPos: highest.startPos + 1 });
      assert.equal(over.length, 1, `seqLen ${seqLen}`);
      assert.match(over[0], /must be at most 131071/);
      assert.match(over[0], new RegExp(`use startPos ${maxStartPosFor(seqLen)} or lower`));
    }
    // The case found in QA: startPos 131071 with 4096 tokens would reach position 135166.
    const qa = validateConfig({ ...base, seqLen: 4096, startPos: 131071 });
    assert.equal(qa.length, 1);
    assert.match(qa[0], /\(the last position, 135166\)/);
    assert.throws(() => runCheck(kernel, reference, { ...base, seqLen: 4096, startPos: 131071 }), RangeError);
    for (const api of ['single', 'packed']) {
      for (const layout of ['adjacent', 'split-half']) {
        const cfg = { ...base, api, layout, seqLen: 8, startPos: maxStartPosFor(8) };
        assert.equal(runCheck(kernel, reference, cfg).pass, true, JSON.stringify(cfg));
        assert.throws(() => runCheck(kernel, reference, { ...cfg, startPos: cfg.startPos + 1 }), RangeError);
      }
    }
  });

  it('builds visualization data from the plan', () => {
    const plan = kernel.createRoPEPlan(8);
    const grid = cosineGrid(plan, 0, 3);
    assert.equal(grid.length, 12);
    assert.deepEqual(Array.from(grid.slice(0, 4)), [1, 1, 1, 1]);
    assert.equal(grid[4], Math.cos(1));
    const tensor = Float32Array.from([0, 1, 2, 3, 4, 5, 6, 7]);
    assert.deepEqual(pairAt(tensor, { ...base, headDim: 8 }, 0, 1), [2, 3]);
    assert.deepEqual(pairAt(tensor, { ...base, headDim: 8, layout: 'split-half' }, 0, 1), [1, 5]);
  });

  it('produces finite timing samples without asserting speed', () => {
    const t = timeKernel(kernel, { ...base, seqLen: 8 }, {
      now: () => performance.now(), targetMs: 1, samples: 3, warmupMs: 1,
    });
    assert.equal(t.samples.length, 3);
    for (const value of t.samples) assert.ok(Number.isFinite(value) && value >= 0);
  });
});

describe('static Pages site build', () => {
  it('ships the unmodified kernel and resolvable same-origin imports only', () => {
    const out = mkdtempSync(join(tmpdir(), 'rope-site-'));
    try {
      execFileSync(process.execPath, [join(root, 'scripts/build-site.mjs'), out], { stdio: 'pipe' });
      for (const file of ['index.html', 'playground.html', 'playground.css', 'playground.js',
        'playground-core.js', 'playground-fallback.js', 'src/rope-kernel.js', 'support/reference.js']) {
        assert.ok(existsSync(join(out, file)), file);
      }
      assert.deepEqual(readFileSync(join(out, 'src/rope-kernel.js')),
        readFileSync(join(root, 'src/rope-kernel.js')));
      assert.deepEqual(readFileSync(join(out, 'support/reference.js')),
        readFileSync(join(root, 'support/reference.mjs')));
      const files = readdirSync(out, { recursive: true }).map(String)
        .filter(name => /\.(html|js|css)$/.test(name));
      for (const name of files) {
        const text = readFileSync(join(out, name), 'utf8');
        for (const match of text.matchAll(/\bfrom\s+['"]([^'"]+)['"]/g)) {
          assert.ok(match[1].startsWith('./'), `${name}: non-relative import ${match[1]}`);
          assert.ok(existsSync(resolve(dirname(join(out, name)), match[1])), `${name}: missing ${match[1]}`);
        }
        // Network-capable APIs and remote subresources are not allowed in the site.
        assert.doesNotMatch(text, /\b(fetch|XMLHttpRequest|sendBeacon|WebSocket|EventSource)\s*\(/, name);
        assert.doesNotMatch(text, /<(script|link|img|iframe|source)\b[^>]*\b(src|href)=["']?(https?:)?\/\//i, name);
        assert.doesNotMatch(text, /url\(\s*['"]?(https?:)?\/\//i, name);
        assert.doesNotMatch(text, /import\s*\(?\s*['"](https?:)?\/\//, name);
      }
      const page = readFileSync(join(out, 'playground.html'), 'utf8');
      assert.match(page, /connect-src 'none'/);
      assert.match(page, /<script type="module" src="playground.js"><\/script>/);
      // A classic script reports a module that failed to load; the module marks itself ready.
      assert.match(page, /<script src="playground-fallback.js" defer><\/script>/);
      assert.match(readFileSync(join(out, 'playground.js'), 'utf8'), /dataset\.demo = 'ready'/);
      // The page names the reference file it actually loads.
      assert.match(page, /<code>support\/reference\.js<\/code>/);
      for (const html of ['index.html', 'playground.html']) {
        const text = readFileSync(join(out, html), 'utf8');
        for (const tag of ['name="description"', 'property="og:title"', 'property="og:description"',
          'property="og:url"', 'property="og:type"']) {
          assert.ok(text.includes(tag), `${html}: ${tag}`);
        }
        assert.match(text, /Content-Security-Policy/, html);
        assert.doesNotMatch(text, /og:image/, `${html}: no external preview image`);
      }
    } finally {
      rmSync(out, { recursive: true, force: true });
    }
  });
});

describe('benchmark script smoke run (correctness only; timings are not asserted)', () => {
  it('checks every benchmarked case against the reference and emits a report', () => {
    const report = JSON.parse(execFileSync(process.execPath,
      [join(root, 'bench/throughput.mjs'), '--smoke'], { encoding: 'utf8', timeout: 60000 }));
    assert.equal(report.kind, 'smoke');
    assert.ok(report.results.length >= 6);
    for (const result of report.results) {
      assert.ok(result.maxAbsErrorVsReference <= 1e-6, result.api);
      assert.ok(Number.isFinite(result.medianNsPerPair), result.api);
    }
    assert.equal('hostname' in report.environment, false);
  });
});
