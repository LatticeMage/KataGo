// Neural net evaluator on top of an onnxruntime InferenceSession (web or node).
(function (root) {
  'use strict';
  const E = root.GoEngine || require('./engine.js').GoEngine;

  class NNEval {
    constructor(ort, session) {
      this.ort = ort; this.session = session;
      this.sp = new Float32Array(E.NUM_SPATIAL * 19 * 19);
      this.gl = new Float32Array(E.NUM_GLOBAL);
      this.evals = 0;
    }
    // positions: array of Pos, all with the same board size. Returns one result per position:
    // {policy: Float32Array(n*n) logits, pass, win (-1..1), winProb, score, own: Float32Array(n*n)}
    // All values are from the point of view of the player to move.
    async evalBatch(positions, fixedSym) {
      const B = positions.length, n = positions[0].n, n2 = n * n, C = E.NUM_SPATIAL, G = E.NUM_GLOBAL;
      const inSp = new Float32Array(B * C * n2), inGl = new Float32Array(B * G);
      const syms = new Array(B);
      const sp = this.sp.subarray(0, C * n2), gl = this.gl;
      for (let b = 0; b < B; b++) {
        E.fillFeatures(positions[b], sp, gl);
        const s = fixedSym !== undefined ? fixedSym : (Math.random() * 8) | 0;
        syms[b] = s;
        const m = E.symMap(n, s), base = b * C * n2;
        for (let c = 0; c < C; c++) for (let p = 0; p < n2; p++) inSp[base + c * n2 + p] = sp[c * n2 + m[p]];
        inGl.set(gl, b * G);
      }
      const T = this.ort.Tensor;
      const out = await this.session.run({
        InputSpatial: new T('float32', inSp, [B, C, n, n]),
        InputGlobal: new T('float32', inGl, [B, G]),
      });
      this.evals += B;
      const pol = out.OutputPolicy.data, pass = out.OutputPolicyPass.data, val = out.OutputValue.data,
        sc = out.OutputScoreValue.data, own = out.OutputOwnership.data, scN = out.OutputScoreValue.dims[1],
        pc = out.OutputPolicy.dims[1], pp = out.OutputPolicyPass.dims[1];
      const res = [];
      for (let b = 0; b < B; b++) {
        const m = E.symMap(n, syms[b]);
        const policy = new Float32Array(n2), ownership = new Float32Array(n2);
        for (let p = 0; p < n2; p++) {
          policy[m[p]] = pol[b * pc * n2 + p];
          ownership[m[p]] = Math.tanh(own[b * n2 + p]);
        }
        const w = val[b * 3], l = val[b * 3 + 1], mx = Math.max(w, l);
        const ew = Math.exp(w - mx), el = Math.exp(l - mx), winProb = ew / (ew + el);
        res.push({ policy, pass: pass[b * pp], win: 2 * winProb - 1, winProb, score: sc[b * scN] * 20, own: ownership });
      }
      return res;
    }
  }
  root.NNEval = NNEval;
  if (typeof module !== 'undefined') module.exports = { NNEval };
})(typeof self !== 'undefined' ? self : globalThis);
