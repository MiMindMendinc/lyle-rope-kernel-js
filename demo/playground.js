// DOM glue for the live demo. Imports only same-origin files built into the Pages site.
import * as kernel from './src/rope-kernel.js';
import * as reference from './support/reference.js';
import {
  FIELD_LABELS, LAYOUT_LABELS, LIMITS, NUMBER_FIELDS, configFromFields, cosineGrid, maxStartPosFor, pairAt,
  runCheck, timeKernel, validateConfigFields,
} from './playground-core.js';

const $ = id => document.getElementById(id);
const form = $('controls');
const AUTO_RUN_DELAY_MS = 350;
let pendingRun = 0;
let lastRenderedKey = null;
let lastTimedKey = null;

function readConfig() {
  const fields = { api: $('api').value, layout: $('layout').value, headDim: $('headDim').value,
    cached: $('cached').checked };
  // Raw strings: an empty field must stay empty (NaN) instead of becoming 0.
  for (const name of NUMBER_FIELDS) fields[name] = $(name).value;
  return configFromFields(fields);
}

function configKey(cfg) {
  const { api, layout, headDim, cached, seqLen, startPos, base, seed } = cfg;
  const heads = api === 'packed' ? [cfg.qHeads, cfg.kvHeads] : [];
  return JSON.stringify([api, layout, headDim, cached, seqLen, startPos, base, seed, ...heads]);
}

// Keep the native spinner limits consistent with the tested position bound.
function syncPositionLimits(cfg) {
  const seqOk = Number.isSafeInteger(cfg.seqLen) && cfg.seqLen >= 1;
  const startOk = Number.isSafeInteger(cfg.startPos) && cfg.startPos >= 0;
  $('startPos').max = String(seqOk ? Math.max(0, Math.min(LIMITS.maxStartPos, maxStartPosFor(cfg.seqLen)))
    : LIMITS.maxStartPos);
  $('seqLen').max = String(startOk ? Math.max(1, Math.min(LIMITS.maxSeqLen, LIMITS.maxPosition - cfg.startPos + 1))
    : LIMITS.maxSeqLen);
}

const FIELD_IDS = ['api', 'layout', 'headDim', 'cached', ...NUMBER_FIELDS];

// A message slot right under each control, so an error appears next to the field that caused it.
function fieldErrorSlot(id) {
  const existing = $(id + '-error');
  if (existing) return existing;
  const slot = document.createElement('span');
  slot.id = id + '-error';
  slot.className = 'field-error error';
  // Hidden from the label's accessible name; read out through the control's aria-describedby.
  slot.setAttribute('aria-hidden', 'true');
  slot.hidden = true;
  $(id).closest('label').append(slot);
  return slot;
}

function showConfigErrors(errors) {
  const byField = new Map();
  for (const { field, message } of errors) {
    byField.set(field, byField.has(field) ? byField.get(field) + ' ' + message + '.' : message + '.');
  }
  for (const id of FIELD_IDS) {
    const control = $(id);
    const slot = fieldErrorSlot(id);
    const text = byField.get(id);
    slot.hidden = !text;
    slot.textContent = text ?? '';
    if (text) {
      control.setAttribute('aria-invalid', 'true');
      control.setAttribute('aria-describedby', slot.id);
    } else {
      control.removeAttribute('aria-invalid');
      control.removeAttribute('aria-describedby');
    }
  }
  // Short summary below the buttons; the full message sits next to each field.
  const labels = [...byField.keys()].map(field => FIELD_LABELS[field]);
  const box = $('config-error');
  box.hidden = errors.length === 0;
  box.textContent = labels.length === 1 ? `Check the highlighted setting: ${labels[0]}.`
    : `Check the ${labels.length} highlighted settings: ${labels.join(', ')}.`;
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
    row('Layout / start position / base', `${LAYOUT_LABELS[cfg.layout]} / ${cfg.startPos} / ${cfg.base}`),
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

function timingPlaceholder(text) {
  const box = $('timing');
  const p = document.createElement('p');
  p.className = 'muted';
  p.textContent = text;
  box.replaceChildren(p);
  lastTimedKey = null;
}

// Nothing on screen may describe a configuration other than the current one.
function showNotRun(message) {
  const verdict = document.createElement('p');
  verdict.id = 'verdict';
  verdict.className = 'verdict not-run';
  verdict.dataset.status = 'not-run';
  verdict.textContent = 'Not run';
  const note = document.createElement('p');
  note.className = 'muted';
  note.textContent = message;
  $('check').replaceChildren(verdict, note);
  $('viz').classList.add('stale');
  $('dials-caption').textContent = 'Not run: the visuals are hidden until the configuration is valid.';
  $('heatmap').setAttribute('aria-label', 'Not run');
  $('dials').setAttribute('aria-label', 'Not run');
  timingPlaceholder('Not run: fix the configuration above, then press Time it in your browser.');
  lastRenderedKey = null;
}

function cancelPendingRun() {
  clearTimeout(pendingRun);
  pendingRun = 0;
}

// force: re-run even if the configuration has not changed (Re-run check, Enter, new seed).
function runAndRender({ force = false } = {}) {
  cancelPendingRun();
  const cfg = readConfig();
  syncPositionLimits(cfg);
  const errors = validateConfigFields(cfg);
  if (!showConfigErrors(errors)) {
    showNotRun('Fix the highlighted settings above; results from the previous run were cleared.');
    return false;
  }
  const key = configKey(cfg);
  if (!force && key === lastRenderedKey) return true;
  try {
    const result = runCheck(kernel, reference, cfg);
    renderCheck(cfg, result);
    drawHeatmap(cfg, result.plan);
    drawDials(cfg, result);
    $('viz').classList.remove('stale');
    lastRenderedKey = key;
  } catch (error) {
    showNotRun('Error: ' + error.message);
    return false;
  }
  if (lastTimedKey !== null && lastTimedKey !== key) {
    timingPlaceholder('The configuration changed since the last timing. Press Time it in your browser to time this one.');
  }
  return true;
}

function scheduleRun() {
  cancelPendingRun();
  pendingRun = setTimeout(() => runAndRender(), AUTO_RUN_DELAY_MS);
}

function timeAndRender() {
  if (!runAndRender()) return;
  const cfg = readConfig();
  const key = configKey(cfg);
  setBusy(true);
  $('timing').textContent = 'Timing… the page may be unresponsive for a moment.';
  // Yield once so the status text paints before the main thread is busy.
  setTimeout(() => {
    // The configuration may have been edited during the yield; never label old timing as current.
    if (lastRenderedKey !== key || configKey(readConfig()) !== key) {
      setBusy(false);
      timingPlaceholder('The configuration changed before timing started. Press Time it in your browser again.');
      return;
    }
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
      lastTimedKey = key;
    } catch (error) {
      timingPlaceholder('Error: ' + error.message);
    } finally {
      setBusy(false);
    }
  }, 30);
}

function syncPackedFields() {
  form.classList.toggle('hidden-packed', $('api').value !== 'packed');
}

form.addEventListener('submit', event => { event.preventDefault(); runAndRender({ force: true }); });
$('time').addEventListener('click', timeAndRender);
$('reseed').addEventListener('click', () => {
  $('seed').value = String(crypto.getRandomValues(new Uint32Array(1))[0]);
  runAndRender({ force: true });
});
$('api').addEventListener('change', () => { syncPackedFields(); runAndRender(); });
for (const id of ['layout', 'headDim', 'cached']) $(id).addEventListener('change', () => runAndRender());
// Number fields re-run on their own: debounced while typing, immediately on commit (blur, spinner).
for (const id of NUMBER_FIELDS) {
  $(id).addEventListener('input', scheduleRun);
  $(id).addEventListener('change', () => runAndRender());
}
syncPackedFields();
runAndRender();
// Tells playground-fallback.js that the module loaded and started.
document.documentElement.dataset.demo = 'ready';
