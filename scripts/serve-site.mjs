// Serve the built static site for local preview. Node built-ins only.
//   npm run site:build && npm run site:serve        -> http://127.0.0.1:8080/
//   node scripts/serve-site.mjs [dir] [port]
// Binds to the loopback interface only and serves files read-only from the directory.
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const dir = resolve(root, process.argv[2] ?? '_site');
const port = Number(process.argv[3] ?? process.env.PORT ?? 8080);
const types = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.svg': 'image/svg+xml', '.png': 'image/png',
};

await stat(resolve(dir, 'index.html')).catch(() => {
  throw new Error(`No built site in ${dir}. Run npm run site:build first.`);
});

createServer(async (request, response) => {
  try {
    const path = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    let file = resolve(dir, '.' + path);
    if (file !== dir && !file.startsWith(dir + sep)) throw Object.assign(new Error(), { code: 'EACCES' });
    if ((await stat(file)).isDirectory()) file = resolve(file, 'index.html');
    const body = await readFile(file);
    response.writeHead(200, { 'content-type': types[extname(file)] ?? 'application/octet-stream',
      'cache-control': 'no-store', 'x-content-type-options': 'nosniff' });
    response.end(request.method === 'HEAD' ? undefined : body);
  } catch {
    response.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
    response.end('Not found');
  }
}).listen(port, '127.0.0.1', () => {
  console.log(`Serving ${dir} at http://127.0.0.1:${port}/ (Ctrl+C to stop)`);
});
