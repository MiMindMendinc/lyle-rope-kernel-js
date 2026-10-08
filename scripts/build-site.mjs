// Assemble the static GitHub Pages site. Node built-ins only; no bundler, no network.
//   npm run site:build            -> _site/
//   node scripts/build-site.mjs <outDir>
// The site serves the unmodified ESM kernel and the repo's scalar reference so the
// browser demo runs the same source files that the Node tests exercise.
import { cpSync, existsSync, mkdirSync, rmSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const out = resolve(root, process.argv[2] ?? '_site');
if (out === resolve(root) || out === resolve(root, 'demo') || out === resolve(root, 'src')) {
  throw new Error('Refusing to build into a source directory: ' + out);
}
if (existsSync(out)) rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });
cpSync(resolve(root, 'demo'), out, { recursive: true });
mkdirSync(resolve(out, 'src'), { recursive: true });
cpSync(resolve(root, 'src/rope-kernel.js'), resolve(out, 'src/rope-kernel.js'));
mkdirSync(resolve(out, 'support'), { recursive: true });
// Copied as .js so every static host serves it with a JavaScript MIME type.
cpSync(resolve(root, 'support/reference.mjs'), resolve(out, 'support/reference.js'));
console.log('Built static site in ' + out);
