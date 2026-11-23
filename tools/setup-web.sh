#!/usr/bin/env bash
# Copies runtime dependencies (onnxruntime-web, coi-serviceworker) from npm into web/.
set -euo pipefail
cd "$(dirname "$0")/../web"
npm install --no-audit --no-fund --silent
mkdir -p ort
D=node_modules/onnxruntime-web/dist
cp $D/ort.wasm.min.js $D/ort.webgpu.min.js \
   $D/ort-wasm-simd-threaded.mjs $D/ort-wasm-simd-threaded.wasm \
   $D/ort-wasm-simd-threaded.jsep.mjs $D/ort-wasm-simd-threaded.jsep.wasm ort/
cp node_modules/coi-serviceworker/coi-serviceworker.min.js coi-serviceworker.js
echo "web/ ready"
