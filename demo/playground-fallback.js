// Classic (non-module) script: if the demo module never finished starting (a file failed to
// load, or the browser lacks ES module support), replace "Loading the kernel…" with a visible
// message instead of leaving the page waiting. playground.js sets data-demo="ready" on <html>.
window.addEventListener('load', () => {
  if (document.documentElement.dataset.demo === 'ready') return;
  const box = document.getElementById('check');
  if (!box) return;
  const verdict = document.createElement('p');
  verdict.id = 'verdict';
  verdict.className = 'verdict fail';
  verdict.dataset.status = 'load-error';
  verdict.textContent = 'The demo could not start';
  const note = document.createElement('p');
  note.className = 'muted';
  note.textContent = 'The kernel module did not load in this browser. Reload the page; if it still fails, ' +
    'your browser may block or not support JavaScript modules. The same checks run from a checkout ' +
    'with npm run verify.';
  box.replaceChildren(verdict, note);
  for (const id of ['run', 'time', 'reseed']) {
    const button = document.getElementById(id);
    if (button) button.disabled = true;
  }
});
