# KataGo in the browser

Play against, or analyse with, a [KataGo](https://github.com/lightvector/KataGo) neural network that runs
entirely in your browser (onnxruntime-web: WASM multi-thread CPU, or WebGPU). No server.

**Play:** https://lattice.posetmage.com/KataGo/

- **Analysis mode** — place black and white stones alternately; the engine keeps analysing the current
  position (win rate, score, territory, candidate moves, best line).
- **Game mode** — play the engine with a per-move clock (default 10 s). Engine evaluation is hidden until the
  game ends.
- Default network: `kata1-b18c384nbt` quantized to int8 (27 MB), downloaded from
  [HomunMage/model](https://github.com/HomunMage/model) and cached by the browser.

## How it works

| piece | file |
|---|---|
| Go rules, KataGo V7 input features, Benson pass-alive area | `web/engine.js` |
| ONNX evaluation with random board symmetry | `web/nn.js` |
| Batched PUCT MCTS with virtual loss | `web/mcts.js` |
| Worker: model download/cache, session, search | `web/worker.js` |
| UI | `web/index.html`, `web/app.js` |

Known gaps: ladder input features are not computed (left at zero); simple-ko inside the search tree
(positional superko is enforced on real moves); no resignation or dead-stone removal at game end.

## Model pipeline (`tools/`)

```
python3 tools/convert.py kata1-b18c384nbt-….bin.gz models/b18c384nbt.onnx   # KataGo weights -> ONNX (v8..v14)
node    tools/gen_calib.js                                                    # realistic calibration positions
python3 tools/quantize.py models/b18c384nbt.onnx models/b18c384nbt-int8.onnx  # static int8 (QLinearConv)
```

int8 vs fp32 on 88 positions: policy KL 0.015, top-1 agreement 88.6 %, win-rate difference 0.8 % (mean).
`tools/bench.js` and `tools/selftest.js` measure throughput / run a smoke game in Node.

## Run locally

```
tools/setup-web.sh          # copies onnxruntime-web + coi-serviceworker into web/
node tools/serve.js web 8080   # serves with COOP/COEP headers (needed for WASM threads)
```

Open http://localhost:8080/ . Add `?debug` for console logging, `?model=URL` to load another ONNX.

## Deploy

GitHub Actions (`.github/workflows/pages.yml`) builds and publishes `web/` to GitHub Pages.
Pages cannot send COOP/COEP headers, so `coi-serviceworker` adds them (the page reloads once on first visit).
In the repo settings, **Pages → Source** must be set to **GitHub Actions**.

## Credits and licenses

KataGo by David J Wu (lightvector), MIT. Networks from katagotraining.org. onnxruntime-web (MIT),
coi-serviceworker (MIT).
