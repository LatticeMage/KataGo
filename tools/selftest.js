// Node smoke test: full pipeline (features -> ONNX -> MCTS) using onnxruntime-web's wasm backend.
const path = require('path');
const web = path.join(__dirname, '..', 'web');
const ort = require(path.join(web, 'node_modules', 'onnxruntime-web'));
const { GoEngine: E } = require(path.join(web, 'engine.js'));
const { NNEval } = require(path.join(web, 'nn.js'));
const { Search } = require(path.join(web, 'mcts.js'));
(async () => {
  const size = +process.argv[2] || 9, visits = +process.argv[3] || 100, moves = +process.argv[4] || 6, batch = +process.argv[5] || 4;
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(path.join(web, 'models', process.argv[6] || 'b10c128.onnx'), { executionProviders: ['wasm'] });
  const ev = new NNEval(ort, session);
  const pos = new E.Pos(size, 7.5);
  for (let m = 0; m < moves; m++) {
    const s = new Search({ evaluator: ev, pos, batch });
    const t = Date.now();
    const info = await s.run(visits, 1e9);
    const dt = (Date.now() - t) / 1000;
    const best = info.cands[0];
    console.log(`move ${m + 1} ${pos.next === 1 ? 'B' : 'W'} -> ${E.idxToGtp(size, best.move)}  visits=${info.visits} wr=${(info.winrate * 100).toFixed(1)}% score=${info.score.toFixed(1)} ${(info.visits / dt).toFixed(1)} v/s  top: ` +
      info.cands.slice(0, 4).map((c) => `${E.idxToGtp(size, c.move)}:${c.visits}`).join(' '));
    pos.play(best.move);
  }
  const rows = []; for (let y = 0; y < size; y++) rows.push([...pos.cells.slice(y * size, y * size + size)].map((v) => '.XO'[v]).join(' '));
  console.log(rows.join('\n'));
})();
