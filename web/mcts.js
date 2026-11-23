// PUCT Monte-Carlo tree search with batched leaf evaluation and virtual loss.
(function (root) {
  'use strict';
  const E = root.GoEngine || require('./engine.js').GoEngine;
  const { PASS } = E;

  class Node {
    constructor() {
      this.N = 0; this.W = 0; this.vl = 0;       // W: sum of values from this node's player-to-move view
      this.moves = null; this.priors = null; this.kids = null;
      this.expanded = false; this.pending = false;
      this.terminal = false; this.tv = 0;
      this.nnWin = 0; this.score = 0; this.own = null;
    }
  }

  const CPUCT = 1.0, CPUCT_LOG = 0.45, CPUCT_BASE = 500, FPU_REDUCTION = 0.2;

  class Search {
    // opts: {evaluator, pos, banned:Set(idx), policyTemp, maxVisits, batch}
    constructor(opts) {
      this.ev = opts.evaluator; this.rootPos = opts.pos; this.banned = opts.banned || new Set();
      this.temp = opts.policyTemp || 1.0; this.batch = opts.batch || 4;
      this.root = new Node(); this.stopped = false; this.n = opts.pos.n;
    }

    expand(node, pos, r) {
      const n2 = pos.n * pos.n, an = pos.analyze(), isRoot = pos === this.rootPos;
      const moves = [], logits = [];
      const T = 1 / this.temp;
      for (let i = 0; i < n2; i++) {
        if (!pos.isLegal(i, an)) continue;
        if (isRoot && this.banned.has(i)) continue;
        moves.push(i); logits.push(r.policy[i] * T);
      }
      moves.push(PASS); logits.push(r.pass * T);
      let mx = -Infinity; for (const l of logits) if (l > mx) mx = l;
      let sum = 0; const pri = new Float32Array(logits.length);
      for (let i = 0; i < logits.length; i++) { pri[i] = Math.exp(logits[i] - mx); sum += pri[i]; }
      for (let i = 0; i < pri.length; i++) pri[i] /= sum;
      node.moves = Int16Array.from(moves); node.priors = pri; node.kids = new Array(moves.length).fill(null);
      node.expanded = true; node.pending = false;
      node.nnWin = r.win; node.score = r.score; node.own = r.own;
    }

    // Choose child index of an expanded node.
    select(node) {
      const total = node.N + node.vl;
      const cpuct = CPUCT + CPUCT_LOG * Math.log((total + CPUCT_BASE) / CPUCT_BASE);
      let visitedPolicy = 0;
      for (let i = 0; i < node.kids.length; i++) { const k = node.kids[i]; if (k && k.N + k.vl > 0) visitedPolicy += node.priors[i]; }
      const parentQ = node.N > 0 ? node.W / node.N : node.nnWin;
      const fpu = parentQ - FPU_REDUCTION * Math.sqrt(visitedPolicy);
      const sq = Math.sqrt(total + 1);
      let best = -1, bestScore = -Infinity;
      for (let i = 0; i < node.kids.length; i++) {
        const k = node.kids[i]; let q, nn = 0;
        if (k && k.N + k.vl > 0) { nn = k.N + k.vl; q = (-k.W - k.vl) / nn; } else q = fpu;
        const s = q + cpuct * node.priors[i] * sq / (1 + nn);
        if (s > bestScore) { bestScore = s; best = i; }
      }
      return best;
    }

    backup(path, v) { // v: value for the player to move at the last node of path
      for (let i = path.length - 1; i >= 0; i--) {
        const nd = path[i]; nd.N++; nd.W += v; nd.vl = Math.max(0, nd.vl - 1); v = -v;
      }
    }

    // Descend once with virtual loss. Returns {path, pos, leaf} where leaf needs an eval, or null
    // if the simulation finished at a terminal/duplicate node.
    descend() {
      const pos = this.rootPos.clone(), path = [this.root];
      let node = this.root; node.vl++;
      for (;;) {
        if (node.terminal) { this.backup(path, node.tv); return null; }
        if (!node.expanded) {
          if (node.pending) { for (const p of path) p.vl--; return 'dup'; }
          node.pending = true; return { path, pos, leaf: node };
        }
        const i = this.select(node), mv = node.moves[i];
        pos.play(mv);
        let kid = node.kids[i];
        if (!kid) {
          kid = node.kids[i] = new Node();
          if (pos.passes >= 2) { kid.terminal = true; kid.tv = pos.resultFor(pos.next); }
        }
        kid.vl++; path.push(kid); node = kid;
      }
    }

    // Run until maxVisits total root visits, or time/stop. onInfo called every ~250ms.
    async run(maxVisits, maxMs, onInfo) {
      const t0 = Date.now(); let lastInfo = 0;
      if (!this.root.expanded) {
        const [r] = await this.ev.evalBatch([this.rootPos]);
        this.expand(this.root, this.rootPos, r);
        this.root.N = 1; this.root.W = r.win;
      }
      let batchMs = 0; // running estimate of one batch's duration, so we don't start one we can't finish
      while (!this.stopped && this.root.N < maxVisits && Date.now() - t0 + batchMs * 0.9 < maxMs) {
        const tb = Date.now();
        const leaves = [];
        for (let b = 0; b < this.batch && this.root.N + leaves.length < maxVisits; b++) {
          const d = this.descend();
          if (d === 'dup') break;
          if (d) leaves.push(d);
        }
        if (leaves.length) {
          const res = await this.ev.evalBatch(leaves.map((l) => l.pos));
          for (let i = 0; i < leaves.length; i++) {
            const { path, pos, leaf } = leaves[i];
            this.expand(leaf, pos, res[i]);
            this.backup(path, res[i].win);
          }
        }
        batchMs = batchMs ? 0.5 * batchMs + 0.5 * (Date.now() - tb) : Date.now() - tb;
        // session.run is synchronous inside a worker, so awaiting it only yields microtasks. Yield a real
        // macrotask so the worker can receive 'stop' / new 'think' messages between batches.
        await new Promise((r) => setTimeout(r, 0));
        const now = Date.now();
        if (onInfo && now - lastInfo > 250) { lastInfo = now; onInfo(this.info()); }
      }
      return this.info();
    }

    info() {
      const r = this.root, n = this.n, cands = [];
      if (r.expanded) for (let i = 0; i < r.moves.length; i++) {
        const k = r.kids[i];
        cands.push({ move: r.moves[i], prior: r.priors[i], visits: k ? k.N : 0, q: k && k.N ? -k.W / k.N : null });
      }
      cands.sort((a, b) => b.visits - a.visits || b.prior - a.prior);
      // principal variation
      const pv = []; let nd = r;
      while (nd && nd.expanded && pv.length < 8) {
        let bi = -1, bv = 0;
        for (let i = 0; i < nd.kids.length; i++) { const k = nd.kids[i]; if (k && k.N > bv) { bv = k.N; bi = i; } }
        if (bi < 0) break; pv.push(nd.moves[bi]); nd = nd.kids[bi];
      }
      return {
        visits: r.N, winrate: r.N ? (1 + r.W / r.N) / 2 : 0.5, // for the player to move at root
        score: r.score, cands: cands.slice(0, 12), pv, own: r.own, size: n, toPlay: this.rootPos.next,
      };
    }
  }

  root.GoSearch = { Search };
  if (typeof module !== 'undefined') module.exports = { Search };
})(typeof self !== 'undefined' ? self : globalThis);
