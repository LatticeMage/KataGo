// Static dev server with the COOP/COEP headers needed for SharedArrayBuffer (wasm threads).
// Usage: node tools/serve.js [dir=web] [port=8080]
const http = require('http'), fs = require('fs'), path = require('path');
const root = path.resolve(process.argv[2] || path.join(__dirname, '..', 'web')), port = +process.argv[3] || 8080;
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream', '.json': 'application/json', '.css': 'text/css' };
http.createServer((req, res) => {
  let p = path.join(root, decodeURIComponent(req.url.split('?')[0]));
  if (!p.startsWith(root)) { res.writeHead(403).end(); return; }
  if (fs.existsSync(p) && fs.statSync(p).isDirectory()) p = path.join(p, 'index.html');
  if (!fs.existsSync(p)) { res.writeHead(404).end('not found'); return; }
  res.writeHead(200, {
    'Content-Type': types[path.extname(p)] || 'application/octet-stream',
    'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp',
    'Cross-Origin-Resource-Policy': 'cross-origin', 'Cache-Control': 'no-cache',
  });
  fs.createReadStream(p).pipe(res);
}).listen(port, () => console.log(`serving ${root} at http://localhost:${port}/`));
