// DOM glue for the live demo. Imports only same-origin files built into the Pages site.
import * as kernel from './src/rope-kernel.js';
import * as reference from './support/reference.js';
import {
  LIMITS, cosineGrid, pairAt, runCheck, timeKernel, validateConfig,
} from './playground-core.js';

const $ = id => document.getElementById(id);
const form = $('controls');
const numberFields = ['seqLen', 'startPos', 'qHeads', 'kvHeads', 'seed'];

function readConfig() {
  const cfg = {
    api: $('api').value,
    layout: $('layout').value,
    headDim: Number($('headDim').value),
    base: Number($('base').value),
    cached: $('cached').checked,
  };
  for (const name of numberFields) cfg[name] = Number($(name).value);
  return cfg;
}

function showConfigErrors(errors) {
  const box = $('config-error');
  box.hidden = errors.length === 0;
  box.textContent = errors.join('. ');
  return errors.length === 0;
}

function fmt(value, digits = 3) {
  if (value === 0) return '0';
  if (Math.abs(value) < 1e-3) return value.toExponential(2);
  return value.toFixed(digits);
}

function row(label, value) {
  const tr = document.createElement('tr');
  const th = document.createElement('th');
  const td = document.createElement('td');
  th.scope = 'row';
  th.textContent = label;
  td.textContent = value;
  tr.append(th, td);
  return tr;
}

function describe(cfg) {
  const shape = cfg.api === 'single'
    ? `[${cfg.seqLen}, ${cfg.headDim}]`
    : `Q [${cfg.seqLen}, ${cfg.qHeads}, ${cfg.headDim}], K [${cfg.seqLen}, ${cfg.kvHeads}, ${cfg.headDim}]`;
  const api = cfg.api === 'packed' ? 'applyRoPEQK'
    : cfg.layout === 'adjacent' ? 'applyRoPEWithPlan' : 'applyRoPESplitHalfWithPlan';
  return { api, shape };
}

function renderCheck(cfg, result) {
  const { api, shape } = describe(cfg);
  const box = $('check');
  box.replaceChildren();
  const verdict = document.createElement('p');
  verdict.id = 'verdict';
  verdict.className = 'verdict ' + (result.pass ? 'pass' : 'fail');
  verdict.dataset.status = result.pass ? 'pass' : 'fail';
  verdict.textContent = result.pass
    ? 'PASS: kernel output matches the scalar reference'
    : 'FAIL: kernel output differs from the scalar reference';
  const table = document.createElement('table');
  table.append(
    row('Function', api),
    row('Shape', shape),
    row('Layout / startPos / base', `${cfg.layout} / ${cfg.startPos} / ${cfg.base}`),
    row('Plan', cfg.cached ? `cached trig table (maxSeqLen ${cfg.startPos + cfg.seqLen})` : 'frequencies only'),
    row('Max absolute error vs reference', `${fmt(result.maxAbsError)} (tolerance ${result.tolerance})`),
    row('Relative L2 norm change (rotation should be ~0)', fmt(result.relativeNormChange)),
    row('Synthetic input', `seeded uniform [-1, 1), seed ${cfg.seed}`),
  );
  box.append(verdict, table);
}

function drawHeatmap(cfg, plan) {
  const canvas = $('heatmap');
  const ctx = canvas.getContext('2d');
  const rows = Math.min(cfg.seqLen, 256);
  const cols = plan.halfDim;
  const grid = cosineGrid(plan, cfg.startPos, rows);
  const image = ctx.createImageData(cols, rows);
  for (let i = 0; i < grid.length; i++) {
    const v = grid[i];
    const t = Math.abs(v);
    // Diverging map: orange (-1) -> near-black (0) -> blue (+1).
    const [r, g, b] = v >= 0 ? [24 + 35 * t, 24 + 106 * t, 27 + 219 * t] : [24 + 225 * t, 24 + 91 * t, 27 - 5 * t];
    image.data.set([r, g, b, 255], i * 4);
  }
  const off = document.createElement('canvas');
  off.width = cols;
  off.height = rows;
  off.getContext('2d').putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  canvas.setAttribute('aria-label', `Heatmap of cos(theta) for positions ${cfg.startPos} to ` +
    `${cfg.startPos + rows - 1} (rows) and ${cols} frequency pairs (columns)`);
}

function drawDials(cfg, result) {
  const canvas = $('dials');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const half = cfg.headDim / 2;
  const count = Math.min(8, half);
  const pairs = Array.from({ length: count }, (_, j) => Math.round(j * (half - 1) / Math.max(1, count - 1)));
  const token = cfg.seqLen - 1;
  const position = cfg.startPos + token;
  const before = cfg.api === 'single' ? result.inputs.x : result.inputs.q;
  const after = cfg.api === 'single' ? result.outputs.x : result.outputs.q;
  const colsN = 4, cellW = canvas.width / colsN, cellH = canvas.height / 2;
  const radius = Math.min(cellW, cellH) / 2 - 22;
  ctx.font = '12px system-ui, sans-serif';
  ctx.textAlign = 'center';
  pairs.forEach((pair, j) => {
    const cx = (j % colsN) * cellW + cellW / 2;
    const cy = Math.floor(j / colsN) * cellH + cellH / 2 - 6;
    ctx.strokeStyle = '#3f3f46';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.stroke();
    const [b0, b1] = pairAt(before, cfg, token, pair);
    const [a0, a1] = pairAt(after, cfg, token, pair);
    const scale = radius / Math.max(Math.hypot(b0, b1), 1e-9);
    for (const [x, y, color, width] of [[b0, b1, '#a1a1aa', 2], [a0, a1, '#60a5fa', 3]]) {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + x * scale, cy - y * scale);
      ctx.stroke();
    }
    ctx.fillStyle = '#e4e4e7';
    ctx.fillText(`pair ${pair}`, cx, cy + radius + 16);
  });
  const caption = `Token ${token} (position ${position})` + (cfg.api === 'packed' ? ', Q head 0' : '') +
    `: ${count} of ${half} pairs, before (grey) and after (blue), each scaled to the dial`;
  $('dials-caption').textContent = caption;
  canvas.setAttribute('aria-label', caption);
}

function setBusy(busy) {
  for (const id of ['run', 'time', 'reseed']) $(id).disabled = busy;
}

function runAndRender() {
  const cfg = readConfig();
  if (!showConfigErrors(validateConfig(cfg))) return;
  try {
    const result = runCheck(kernel, reference, cfg);
    renderCheck(cfg, result);
    drawHeatmap(cfg, result.plan);
    drawDials(cfg, result);
  } catch (error) {
    $('check').textContent = 'Error: ' + error.message;
  }
}

function timeAndRender() {
  const cfg = readConfig();
  if (!showConfigErrors(validateConfig(cfg))) return;
  setBusy(true);
  $('timing').textContent = 'Timing… the page may be unresponsive for a moment.';
  // Yield once so the status text paints before the main thread is busy.
  setTimeout(() => {
    try {
      const t = timeKernel(kernel, cfg, { now: () => performance.now(), targetMs: 50, samples: 5 });
      const { api, shape } = describe(cfg);
      const box = $('timing');
      box.replaceChildren();
      const table = document.createElement('table');
      table.append(
        row('Function / shape', `${api} ${shape}`),
        row('Median time per rotated pair (your browser)', `${fmt(t.medianNsPerPair, 2)} ns`),
        row('Median time per call (your browser)', `${fmt(t.medianNsPerPair * t.pairsPerCall / 1e3, 1)} µs`),
        row('Samples (ns per pair)', t.samples.map(v => v.toFixed(2)).join(', ')),
        row('Method', `${t.iterations} in-place calls per sample after warmup; plan creation excluded`),
      );
      const note = document.createElement('p');
      note.className = 'muted';
      note.textContent = 'Measured in your browser on this device, on the main thread, with coarse browser ' +
        'timers. Indicative only; not comparable across devices or with the Node numbers in the README.';
      box.append(table, note);
    } catch (error) {
      $('timing').textContent = 'Error: ' + error.message;
    } finally {
      setBusy(false);
    }
  }, 30);
}

function syncPackedFields() {
  form.classList.toggle('hidden-packed', $('api').value !== 'packed');
}

form.addEventListener('submit', event => { event.preventDefault(); runAndRender(); });
$('time').addEventListener('click', timeAndRender);
$('reseed').addEventListener('click', () => {
  $('seed').value = String(crypto.getRandomValues(new Uint32Array(1))[0]);
  runAndRender();
});
$('api').addEventListener('change', () => { syncPackedFields(); runAndRender(); });
for (const id of ['layout', 'headDim', 'cached']) $(id).addEventListener('change', runAndRender);
$('seqLen').max = String(LIMITS.maxSeqLen);
syncPackedFields();
runAndRender();
