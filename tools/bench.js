// Benchmark raw NN throughput in onnxruntime-web (wasm) : node tools/bench.js model.onnx threads
const path = require('path');
const web = path.join(__dirname, '..', 'web');
const ort = require(path.join(web, 'node_modules', 'onnxruntime-web'));
(async () => {
  const model = process.argv[2], threads = +process.argv[3] || 1;
  ort.env.wasm.numThreads = threads;
  const s = await ort.InferenceSession.create(path.join(web, 'models', model), { executionProviders: ['wasm'] });
  for (const [n, B] of [[19, 1], [19, 8], [9, 8]]) {
    const sp = new Float32Array(B * 22 * n * n).fill(0.1), gl = new Float32Array(B * 19);
    const f = { InputSpatial: new ort.Tensor('float32', sp, [B, 22, n, n]), InputGlobal: new ort.Tensor('float32', gl, [B, 19]) };
    await s.run(f);
    const t = Date.now(); const R = 3;
    for (let i = 0; i < R; i++) await s.run(f);
    const dt = (Date.now() - t) / 1000 / R;
    console.log(`${model} thr=${threads} ${n}x${n} batch=${B}: ${(dt * 1000).toFixed(0)} ms/run = ${(B / dt).toFixed(1)} evals/s`);
  }
})();
