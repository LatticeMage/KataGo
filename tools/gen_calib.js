// Generate realistic calibration positions by sampling moves from a fast net's raw policy.
// Writes models/calib_<n>.bin = float32 [N,22,n,n] then [N,19].
const path = require('path'), fs = require('fs');
const web = path.join(__dirname, '..', 'web');
const ort = require(path.join(web, 'node_modules', 'onnxruntime-web'));
const { GoEngine: E } = require(path.join(web, 'engine.js'));
const { NNEval } = require(path.join(web, 'nn.js'));
(async () => {
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(path.join(web, 'models', 'b10c128.onnx'), { executionProviders: ['wasm'] });
  const ev = new NNEval(ort, session);
  const plan = { 9: 96, 13: 96, 19: 160 };
  for (const [nStr, want] of Object.entries(plan)) {
    const n = +nStr, n2 = n * n, sps = [], gls = [];
    while (sps.length < want) {
      const pos = new E.Pos(n, [6.5, 7, 7.5][(Math.random() * 3) | 0]);
      const maxMoves = Math.floor(n2 * (0.3 + 0.9 * Math.random()));
      for (let m = 0; m < maxMoves && pos.passes < 2 && sps.length < want; m++) {
        if (m > 4 && m % 8 === 0) {
          const sp = new Float32Array(22 * n2), gl = new Float32Array(19);
          E.fillFeatures(pos, sp, gl); sps.push(sp); gls.push(gl);
        }
        const [r] = await ev.evalBatch([pos]);
        const legal = pos.legalMoves(); if (!legal.length) break;
        const w = legal.map((i) => Math.exp(r.policy[i] - 0)); const tot = w.reduce((a, b) => a + b, 0);
        let u = Math.random() * tot, pick = legal[legal.length - 1];
        for (let k = 0; k < legal.length; k++) { u -= w[k]; if (u <= 0) { pick = legal[k]; break; } }
        pos.play(pick);
      }
    }
    const buf = Buffer.concat([...sps, ...gls].map((a) => Buffer.from(a.buffer)));
    fs.writeFileSync(path.join(__dirname, '..', 'models', `calib_${n}.bin`), buf);
    console.log(`calib_${n}.bin: ${sps.length} positions`);
  }
})();
