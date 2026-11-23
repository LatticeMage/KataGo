// Go rules + KataGo (model v8 / inputs V7) feature encoding. Classic script: usable
// from the page and from a worker via importScripts. Exposes `GoEngine` on the global.
(function (root) {
  'use strict';
  const EMPTY = 0, BLACK = 1, WHITE = 2, PASS = -1;
  const opp = (c) => 3 - c;

  const nbCache = {};
  function neighbors(n) {
    if (nbCache[n]) return nbCache[n];
    const nb = new Array(n * n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const a = [];
      if (y > 0) a.push((y - 1) * n + x);
      if (x > 0) a.push(y * n + x - 1);
      if (x < n - 1) a.push(y * n + x + 1);
      if (y < n - 1) a.push((y + 1) * n + x);
      nb[y * n + x] = Int16Array.from(a);
    }
    return (nbCache[n] = nb);
  }

  class Pos {
    constructor(n, komi) {
      this.n = n; this.komi = komi;
      this.cells = new Uint8Array(n * n);
      this.ko = -1; this.next = BLACK; this.passes = 0;
      this.hist = []; // [color, idx] — only the last few are kept on clones
      this.nb = neighbors(n);
    }
    clone() {
      const p = Object.create(Pos.prototype);
      p.n = this.n; p.komi = this.komi; p.cells = this.cells.slice();
      p.ko = this.ko; p.next = this.next; p.passes = this.passes;
      p.hist = this.hist.slice(-5); p.nb = this.nb;
      return p;
    }
    // Flood fill the group at idx. Returns {stones, libs}.
    group(idx) {
      const { cells, nb } = this, c = cells[idx];
      const stones = [idx], seen = new Set([idx]), libs = new Set();
      for (let i = 0; i < stones.length; i++) {
        for (const a of nb[stones[i]]) {
          const v = cells[a];
          if (v === EMPTY) libs.add(a);
          else if (v === c && !seen.has(a)) { seen.add(a); stones.push(a); }
        }
      }
      return { stones, libs: libs.size };
    }
    // Per-point group id and liberty counts for every group on the board.
    analyze() {
      const n2 = this.n * this.n, { cells, nb } = this;
      const gid = new Int16Array(n2).fill(-1), libsOf = [], stamp = new Int16Array(n2).fill(-1);
      const stack = [];
      for (let i = 0; i < n2; i++) {
        if (!cells[i] || gid[i] >= 0) continue;
        const g = libsOf.length, c = cells[i]; let libs = 0;
        gid[i] = g; stack.length = 0; stack.push(i);
        while (stack.length) {
          const s = stack.pop();
          for (const a of nb[s]) {
            const v = cells[a];
            if (v === EMPTY) { if (stamp[a] !== g) { stamp[a] = g; libs++; } }
            else if (v === c && gid[a] < 0) { gid[a] = g; stack.push(a); }
          }
        }
        libsOf.push(libs);
      }
      return { gid, libsOf };
    }
    isLegal(idx, an) {
      if (idx === PASS) return true;
      const { cells, nb } = this, c = this.next;
      if (cells[idx] !== EMPTY || idx === this.ko) return false;
      an = an || this.analyze();
      for (const a of nb[idx]) {
        const v = cells[a];
        if (v === EMPTY) return true;
        const l = an.libsOf[an.gid[a]];
        if (v === c ? l > 1 : l === 1) return true;
      }
      return false;
    }
    legalMoves() {
      const an = this.analyze(), out = [];
      for (let i = 0; i < this.n * this.n; i++) if (this.isLegal(i, an)) out.push(i);
      return out;
    }
    // Plays for this.next. Caller guarantees legality (use isLegal).
    play(idx) {
      const c = this.next;
      this.hist.push([c, idx]);
      if (this.hist.length > 64) this.hist.shift();
      this.next = opp(c);
      if (idx === PASS) { this.ko = -1; this.passes++; return; }
      this.passes = 0;
      const { cells, nb } = this;
      cells[idx] = c;
      let captured = 0, capIdx = -1;
      for (const a of nb[idx]) {
        if (cells[a] === opp(c)) {
          const g = this.group(a);
          if (g.libs === 0) { for (const s of g.stones) cells[s] = EMPTY; captured += g.stones.length; capIdx = g.stones[0]; }
        }
      }
      this.ko = -1;
      if (captured === 1) {
        const g = this.group(idx);
        if (g.stones.length === 1 && g.libs === 1) this.ko = capIdx;
      }
    }
    // Tromp-Taylor area score: black minus white (komi not applied).
    areaDiff() {
      const { cells, nb } = this, n2 = this.n * this.n;
      let b = 0, w = 0;
      const seen = new Uint8Array(n2);
      for (let i = 0; i < n2; i++) {
        if (cells[i] === BLACK) b++;
        else if (cells[i] === WHITE) w++;
        else if (!seen[i]) {
          const q = [i]; seen[i] = 1; let tb = false, tw = false;
          for (let k = 0; k < q.length; k++) for (const a of nb[q[k]]) {
            if (cells[a] === EMPTY) { if (!seen[a]) { seen[a] = 1; q.push(a); } }
            else if (cells[a] === BLACK) tb = true; else tw = true;
          }
          if (tb && !tw) b += q.length; else if (tw && !tb) w += q.length;
        }
      }
      return b - w;
    }
    // Final result from the point of view of `color`: +1 win, -1 loss.
    resultFor(color) {
      const s = this.areaDiff() - this.komi; // >0 black wins
      const bw = s > 0 ? 1 : s < 0 ? -1 : 0;
      return color === BLACK ? bw : -bw;
    }
    hash() { // exact board key for positional superko
      return String.fromCharCode.apply(null, this.cells);
    }
  }

  // Benson's algorithm: pass-alive stones and the territory they enclose.
  // Returns Uint8Array area[idx] in {0,BLACK,WHITE}; other stones count for their owner
  // (matches KataGo's nonPassAliveStones=true) .
  function passAliveArea(pos) {
    const { n, cells, nb } = pos, n2 = n * n;
    const area = new Uint8Array(n2);
    for (const c of [BLACK, WHITE]) {
      const an = pos.analyze();
      // chains of colour c
      const chainStones = [];
      for (let i = 0; i < n2; i++) if (cells[i] === c) {
        const g = an.gid[i]; (chainStones[g] || (chainStones[g] = [])).push(i);
      }
      // regions: connected sets of non-c points
      const regId = new Int16Array(n2).fill(-1), regions = [];
      for (let i = 0; i < n2; i++) {
        if (cells[i] === c || regId[i] >= 0) continue;
        const r = regions.length, pts = [i]; regId[i] = r;
        for (let k = 0; k < pts.length; k++) for (const a of nb[pts[k]])
          if (cells[a] !== c && regId[a] < 0) { regId[a] = r; pts.push(a); }
        regions.push(pts);
      }
      // adjacency chain<->region, and healthy (vital) relation
      const chains = chainStones.map((s, g) => (s ? g : -1)).filter((g) => g >= 0);
      const adjC = new Map(chains.map((g) => [g, new Set()])), adjR = regions.map(() => new Set());
      for (const g of chains) for (const s of chainStones[g]) for (const a of nb[s])
        if (cells[a] !== c) { adjC.get(g).add(regId[a]); adjR[regId[a]].add(g); }
      const vital = (g, r) => { // every empty point of region r is a liberty of chain g
        for (const p of regions[r]) if (cells[p] === EMPTY && !nb[p].some((a) => cells[a] === c && an.gid[a] === g)) return false;
        return true;
      };
      let aliveC = new Set(chains), aliveR = new Set(regions.keys());
      for (let changed = true; changed;) {
        changed = false;
        for (const g of [...aliveC]) {
          let v = 0;
          for (const r of adjC.get(g)) if (aliveR.has(r) && vital(g, r)) v++;
          if (v < 2) { aliveC.delete(g); changed = true; }
        }
        for (const r of [...aliveR]) {
          for (const g of adjR[r]) if (!aliveC.has(g)) { aliveR.delete(r); changed = true; break; }
        }
      }
      for (const g of aliveC) for (const s of chainStones[g]) area[s] = c;
      for (const r of aliveR) if (adjR[r].size) for (const p of regions[r]) area[p] = c;
    }
    for (let i = 0; i < n2; i++) if (!area[i]) area[i] = cells[i];
    return area;
  }

  const NUM_SPATIAL = 22, NUM_GLOBAL = 19;
  // Fills NCHW features (22*n*n) and global features (19) for the player to move.
  function fillFeatures(pos, sp, gl) {
    const { n, cells } = pos, n2 = n * n, pla = pos.next, op = opp(pla);
    sp.fill(0); gl.fill(0);
    const an = pos.analyze();
    for (let i = 0; i < n2; i++) {
      sp[i] = 1;
      const v = cells[i];
      if (v === EMPTY) continue;
      sp[(v === pla ? 1 : 2) * n2 + i] = 1;
      const l = an.libsOf[an.gid[i]];
      if (l >= 1 && l <= 3) sp[(2 + l) * n2 + i] = 1;
    }
    if (pos.ko >= 0) sp[6 * n2 + pos.ko] = 1;
    // last five moves (must alternate opp, pla, opp, ...)
    const h = pos.hist;
    for (let k = 0; k < 5 && k < h.length; k++) {
      const [c, idx] = h[h.length - 1 - k];
      if (c !== (k % 2 === 0 ? op : pla)) break;
      if (idx === PASS) gl[k] = 1; else sp[(9 + k) * n2 + idx] = 1;
    }
    // area (pass-alive) features, area scoring
    const area = passAliveArea(pos);
    for (let i = 0; i < n2; i++) {
      if (area[i] === pla) sp[18 * n2 + i] = 1; else if (area[i] === op) sp[19 * n2 + i] = 1;
    }
    const selfKomi = pla === WHITE ? pos.komi : -pos.komi;
    gl[5] = Math.max(-n2 - 1, Math.min(n2 + 1, selfKomi)) / 20;
    gl[6] = 1; gl[7] = 0.5; // positional superko
    gl[14] = pos.passes > 0 ? 1 : 0; // a pass would end the game
    // komi parity wave
    const kf = Math.max(-n2 - 1, Math.min(n2 + 1, selfKomi));
    const floorK = n2 % 2 === 0 ? Math.floor(kf / 2) * 2 : Math.floor((kf - 1) / 2) * 2 + 1;
    let d = Math.min(2, Math.max(0, kf - floorK));
    gl[18] = d < 0.5 ? d : d < 1.5 ? 1 - d : d - 2;
  }

  // Board symmetries (dihedral group). map[p] = source index for output position p.
  const symCache = {};
  function symMap(n, s) {
    const key = n * 8 + s;
    if (symCache[key]) return symCache[key];
    const m = new Int32Array(n * n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      let sx = x, sy = y;
      if (s & 4) { const t = sx; sx = sy; sy = t; }
      if (s & 1) sx = n - 1 - sx;
      if (s & 2) sy = n - 1 - sy;
      m[y * n + x] = sy * n + sx;
    }
    return (symCache[key] = m);
  }

  const idxToGtp = (n, idx) => {
    if (idx === PASS) return 'pass';
    const x = idx % n, y = (idx / n) | 0;
    return 'ABCDEFGHJKLMNOPQRST'[x] + (n - y);
  };

  root.GoEngine = { EMPTY, BLACK, WHITE, PASS, opp, Pos, passAliveArea, fillFeatures, symMap, idxToGtp, NUM_SPATIAL, NUM_GLOBAL };
  if (typeof module !== 'undefined') module.exports = { GoEngine: root.GoEngine };
})(typeof self !== 'undefined' ? self : globalThis);
