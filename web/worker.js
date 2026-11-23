// Engine worker: loads the model with onnxruntime-web and runs MCTS off the UI thread.
/* global ort, GoEngine, NNEval, GoSearch */
importScripts('engine.js?v=7', 'nn.js?v=7', 'mcts.js?v=7');

let ev = null, search = null, session = null;

async function fetchModel(url, onProgress) {
  const cache = self.caches ? await caches.open('katago-models-v1') : null;
  let res = cache ? await cache.match(url) : null;
  if (res) {
    const buf = await res.arrayBuffer();
    onProgress(buf.byteLength, buf.byteLength, true);
    return buf;
  }
  res = await fetch(url);
  if (!res.ok) throw new Error(`model download failed: ${res.status} ${url}`);
  const total = +res.headers.get('content-length') || 0;
  const reader = res.body.getReader(), chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length; onProgress(got, total, false);
  }
  const buf = new Uint8Array(got); let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  if (cache) { try { await cache.put(url, new Response(buf.slice(), { headers: { 'content-type': 'application/octet-stream' } })); } catch (e) { /* quota */ } }
  return buf.buffer;
}

async function init(m) {
  search = null; ev = null;
  if (session) { try { await session.release(); } catch (e) { /* ignore */ } session = null; }
  const webgpu = m.backend === 'webgpu';
  importScripts(webgpu ? 'ort/ort.webgpu.min.js' : 'ort/ort.wasm.min.js');
  ort.env.wasm.wasmPaths = new URL('ort/', self.location.href).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, m.threads | 0) : 1;
  ort.env.wasm.proxy = false;
  const buf = await fetchModel(m.modelUrl, (loaded, total, cached) =>
    postMessage({ type: 'progress', loaded, total, cached }));
  postMessage({ type: 'status', text: 'creating session…' });
  const t0 = performance.now();
  session = await ort.InferenceSession.create(buf, {
    executionProviders: [webgpu ? 'webgpu' : 'wasm'],
    graphOptimizationLevel: 'all',
  });
  ev = new NNEval(ort, session);
  postMessage({
    type: 'ready', backend: webgpu ? 'webgpu' : 'wasm', threads: ort.env.wasm.numThreads,
    isolated: !!self.crossOriginIsolated, loadMs: Math.round(performance.now() - t0),
  });
}

async function think(m) {
  const pos = new GoEngine.Pos(m.size, m.komi);
  for (const [c, idx] of m.moves) { pos.next = c; pos.play(idx); }
  pos.next = m.next;
  search = new GoSearch.Search({
    evaluator: ev, pos, banned: new Set(m.banned || []), batch: m.batch || 4, policyTemp: m.policyTemp || 1,
  });
  const t0 = performance.now(), e0 = ev.evals;
  const stats = (info) => ({ ...info, id: m.id, evalsPerSec: (ev.evals - e0) / ((performance.now() - t0) / 1000) });
  const budget = m.deadline ? Math.max(0, m.deadline - Date.now()) : (m.maxMs || 1e9);
  const info = await search.run(m.visits, budget, (i) => postMessage({ type: 'info', info: stats(i) }));
  postMessage({ type: 'done', id: m.id, info: stats(info) });
}

let running = Promise.resolve();
self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') await init(m);
    else if (m.type === 'think') {
      if (search) search.stopped = true;           // let the previous search wind down first
      running = running.then(() => think(m));
      await running;
    }
    else if (m.type === 'stop' && search) search.stopped = true;
  } catch (err) {
    postMessage({ type: 'error', message: String(err && err.message || err), id: m.id });
  }
};
