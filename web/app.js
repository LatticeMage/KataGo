// UI + game controller. The search runs in worker.js.
(function () {
  'use strict';
  const E = GoEngine, { BLACK, WHITE, PASS } = E;
  const $ = (id) => document.getElementById(id);
  const DEBUG = new URLSearchParams(location.search).has('debug'), dbg = (...a) => { if (DEBUG) console.log('[dbg]', ...a); };
  const canvas = $('board'), ctx = canvas.getContext('2d');

  let size = 19, komi = 7.5, human = BLACK;
  let pos, moves, seen, gameOver, info = null, hover = -1;
  let worker = null, ready = false, thinkId = 0, busy = null, engineGen = 0, turnStart = 0, turnStartWall = 0;
  const mode = () => $('mode').value, secs = () => Math.max(1, +$('secs').value || 10);

  // ---------- game state ----------
  function rebuild(list) {
    pos = new E.Pos(size, komi); moves = []; seen = new Set([pos.hash()]);
    for (const [c, idx] of list) { pos.next = c; pos.play(idx); moves.push([c, idx]); seen.add(pos.hash()); }
    pos.hist = pos.hist.slice(-5);
    gameOver = mode() === 'game' && pos.passes >= 2;
  }
  function legal(idx) {
    if (idx === PASS) return true;
    if (!pos.isLegal(idx)) return false;
    const c = pos.clone(); c.play(idx);
    return !seen.has(c.hash());
  }
  function bannedByKo() {
    const an = pos.analyze(), out = [];
    for (let i = 0; i < size * size; i++) if (pos.isLegal(i, an) && !legal(i)) out.push(i);
    return out;
  }
  function play(idx) {
    pos.play(idx); moves.push([pos.next === BLACK ? WHITE : BLACK, idx]); seen.add(pos.hash());
    if (pos.hist.length > 8) pos.hist = pos.hist.slice(-5);
    if (pos.passes >= 2 && mode() === 'game') gameOver = true;
    info = null; draw(); afterMove();
  }

  // ---------- engine ----------
  function startEngine() {
    engineGen++; ready = false; busy = null;
    if (worker) worker.terminate();
    worker = new Worker('worker.js?v=7');
    const gen = engineGen;
    setStatus('loading engine…');
    worker.onmessage = (e) => { if (gen === engineGen) onWorker(e.data); };
    worker.onerror = (e) => setStatus('worker error: ' + e.message);
    worker.postMessage({ type: 'init', modelUrl: $('model').value, backend: $('backend').value, threads: +$('threads').value || 1 });
  }
  function onWorker(m) {
    if (m.type !== 'info' && m.type !== 'progress') dbg('worker→', m.type, m.id, m.message || '');
    if (m.type === 'progress') {
      $('loadFill').style.width = (m.total ? (100 * m.loaded / m.total) : 50) + '%';
      setStatus(m.cached ? 'loading model from cache…' : `downloading model ${(m.loaded / 1e6).toFixed(1)} / ${(m.total / 1e6).toFixed(1)} MB`);
    } else if (m.type === 'status') setStatus(m.text);
    else if (m.type === 'ready') {
      ready = true; $('loadFill').style.width = '100%';
      let t = `ready — ${m.backend}${m.backend === 'wasm' ? `, ${m.threads} thread${m.threads > 1 ? 's' : ''}` : ''} (session in ${m.loadMs} ms)`;
      if (m.backend === 'wasm' && !m.isolated && +$('threads').value > 1) t += ' — not cross-origin isolated, using 1 thread (reload the page once)';
      setStatus(t, true); afterMove();
    } else if (m.type === 'info') { if (m.info.id === thinkId) { info = m.info; showInfo(); draw(); tick(); } }
    else if (m.type === 'done') {
      if (m.id !== thinkId) return;
      const kind = busy; busy = null; info = m.info; showInfo(); draw();
      if (kind === 'ai' && !gameOver && mode() === 'game') {
        dbg('AI answered after', ((Date.now() - turnStartWall) / 1000).toFixed(2), 's');
        const best = info.cands.find((c) => c.visits > 0) || info.cands[0];
        play(best && legal(best.move) ? best.move : PASS);
      }
    } else if (m.type === 'error') { setStatus('error: ' + m.message); busy = null; }
  }
  function think(kind) {
    if (!ready) return;
    busy = kind; thinkId++; dbg('think', kind, 'id', thinkId, 'moves', moves.length);
    worker.postMessage({
      type: 'think', id: thinkId, size, komi, moves, next: pos.next, banned: bannedByKo(),
      visits: 1e9, deadline: kind === 'ai' ? turnStartWall + secs() * 1000 : undefined,
      batch: +$('batch').value || 4,
    });
  }
  function tick() {}
  function updateClocks() {
    if (mode() !== 'game') return;
    const total = secs(), el = (performance.now() - turnStart) / 1000;
    for (const [c, id] of [[BLACK, 'B'], [WHITE, 'W']]) {
      const active = !gameOver && pos.next === c && ready;
      const left = active ? Math.max(0, total - el) : total;
      $('clk' + id + 't').textContent = left.toFixed(1);
      $('clk' + id + 'b').style.width = (100 * left / total) + '%';
      $('clk' + id).classList.toggle('active', active);
      $('clk' + id).classList.toggle('low', left <= 3);
    }
    if ($('enforce').checked && !gameOver && ready && pos.next === human && el > total) {
      gameOver = true; stopThinking(); setStatus('time up — you lose on time'); draw();
    }
  }
  setInterval(updateClocks, 100);
  function stopThinking() { if (busy) { worker.postMessage({ type: 'stop' }); thinkId++; busy = null; } }
  function afterMove() {
    turnStart = performance.now(); turnStartWall = Date.now();
    if (!ready) return;
    if (mode() === 'analysis') { setStatus('analysing', true); think('analyze'); return; }
    if (gameOver) { announceEnd(); return; }
    if (pos.next !== human) { setStatus('AI thinking…'); think('ai'); }
    else setStatus('your move', true); // no background analysis in game mode: CPU stays free for the AI's move
  }
  function announceEnd() {
    const d = pos.areaDiff() - komi;
    setStatus(`game over — area score ${d > 0 ? 'B+' + d : 'W+' + (-d)} (stones as on board)`, true);
    thinkId++; busy = 'final'; worker.postMessage({
      type: 'think', id: thinkId, size, komi, moves, next: pos.next, visits: 100, batch: +$('batch').value || 4,
    });
  }

  // The engine's evaluation (territory, candidates, win rate) is only shown when analysing or after the game.
  const reveal = () => mode() === 'analysis' || gameOver;
  const canMove = () => mode() === 'analysis' || (!gameOver && pos.next === human);

  // ---------- UI ----------
  function setStatus(t, ok) { const s = $('status'); s.textContent = t; s.className = ok ? 'ok' : ''; }
  function showInfo() {
    if (!info || !reveal()) return;
    const blackWr = info.toPlay === BLACK ? info.winrate : 1 - info.winrate;
    const blackLead = info.toPlay === BLACK ? info.score : -info.score;
    $('wr').textContent = (blackWr * 100).toFixed(1) + '%';
    $('barFill').style.width = (blackWr * 100) + '%';
    $('score').textContent = (blackLead >= 0 ? 'B+' : 'W+') + Math.abs(blackLead).toFixed(1);
    $('vis').textContent = info.visits;
    $('speed').textContent = (info.evalsPerSec || 0).toFixed(1) + ' evals/s';
    $('pv').textContent = info.pv.map((p) => E.idxToGtp(size, p)).join(' ') || '–';
  }
  function clearStats() { $('wr').textContent = $('score').textContent = $('vis').textContent = $('speed').textContent = $('pv').textContent = '–'; $('barFill').style.width = '50%'; }
  function newGame() {
    stopThinking();
    size = +$('size').value; komi = +$('komi').value; human = +$('color').value;
    rebuild([]); info = null; $('wr').textContent = $('score').textContent = $('vis').textContent = $('speed').textContent = $('pv').textContent = '–';
    setStatus(ready ? (mode() === 'analysis' ? 'analysing' : 'your move') : 'loading engine…', ready); draw(); afterMove();
  }
  function undo() {
    stopThinking();
    const list = moves.slice(); list.pop();
    if (mode() === 'game') while (list.length && (list.length % 2 === 0 ? BLACK : WHITE) !== human) list.pop();
    rebuild(list); info = null; draw(); afterMove();
  }

  // ---------- drawing ----------
  const STARS = { 9: [2, 4, 6], 13: [3, 6, 9], 19: [3, 9, 15] };
  function layout() {
    const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const w = Math.round(r.width * dpr);
    if (canvas.width !== w) { canvas.width = w; canvas.height = w; }
    const cell = canvas.width / (size + 1);
    return { cell, off: cell };
  }
  function draw() {
    const { cell, off } = layout(), W = canvas.width, n = size;
    ctx.clearRect(0, 0, W, W);
    ctx.fillStyle = '#dcb35c'; ctx.fillRect(0, 0, W, W);
    ctx.strokeStyle = '#3a2a10'; ctx.lineWidth = Math.max(1, cell / 28);
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      ctx.moveTo(off + i * cell, off); ctx.lineTo(off + i * cell, off + (n - 1) * cell);
      ctx.moveTo(off, off + i * cell); ctx.lineTo(off + (n - 1) * cell, off + i * cell);
    }
    ctx.stroke();
    ctx.fillStyle = '#3a2a10';
    for (const sx of STARS[n] || []) for (const sy of STARS[n] || []) {
      ctx.beginPath(); ctx.arc(off + sx * cell, off + sy * cell, cell * 0.09, 0, 7); ctx.fill();
    }
    ctx.font = `${Math.round(cell * 0.32)}px system-ui`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = '#5b4520';
    for (let i = 0; i < n; i++) {
      ctx.fillText('ABCDEFGHJKLMNOPQRST'[i], off + i * cell, cell * 0.4);
      ctx.fillText(n - i, cell * 0.4, off + i * cell);
    }
    const at = (idx) => [off + (idx % n) * cell, off + ((idx / n) | 0) * cell];
    // ownership heat (absolute: positive = black)
    if (reveal() && $('ownership').checked && info && info.own) {
      const sgn = info.toPlay === BLACK ? 1 : -1;
      for (let i = 0; i < n * n; i++) {
        const o = info.own[i] * sgn; if (Math.abs(o) < 0.1) continue;
        const [x, y] = at(i), s = cell * 0.34 * Math.min(1, Math.abs(o) + 0.15);
        ctx.fillStyle = o > 0 ? 'rgba(0,0,0,0.55)' : 'rgba(255,255,255,0.7)';
        ctx.fillRect(x - s, y - s, 2 * s, 2 * s);
      }
    }
    // stones
    const R = cell * 0.47;
    for (let i = 0; i < n * n; i++) {
      const c = pos.cells[i]; if (!c) continue;
      const [x, y] = at(i);
      const g = ctx.createRadialGradient(x - R * 0.3, y - R * 0.3, R * 0.1, x, y, R);
      if (c === BLACK) { g.addColorStop(0, '#555'); g.addColorStop(1, '#050505'); } else { g.addColorStop(0, '#fff'); g.addColorStop(1, '#c9c9c9'); }
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, R, 0, 7); ctx.fill();
      ctx.strokeStyle = 'rgba(0,0,0,.35)'; ctx.lineWidth = 1; ctx.stroke();
    }
    const last = moves[moves.length - 1];
    if (last && last[1] >= 0) {
      const [x, y] = at(last[1]); ctx.strokeStyle = last[0] === BLACK ? '#fff' : '#d22'; ctx.lineWidth = cell * 0.07;
      ctx.beginPath(); ctx.arc(x, y, R * 0.45, 0, 7); ctx.stroke();
    }
    // candidate moves
    if (reveal() && info && info.cands && info.visits > 0) {
      const top = info.cands[0] ? info.cands[0].visits : 1;
      info.cands.slice(0, 8).forEach((c, k) => {
        if (c.move < 0 || c.visits < 1 || pos.cells[c.move]) return;
        const [x, y] = at(c.move), f = Math.sqrt(c.visits / top);
        ctx.fillStyle = k === 0 ? 'rgba(60,170,255,.75)' : `rgba(76,195,138,${0.25 + 0.45 * f})`;
        ctx.beginPath(); ctx.arc(x, y, R * (0.45 + 0.4 * f), 0, 7); ctx.fill();
        if (c.q !== null && c.visits >= 3) {
          const wr = ((1 + c.q) / 2) * 100; // win % for the player to move
          ctx.fillStyle = '#041'; ctx.font = `bold ${Math.round(cell * 0.3)}px system-ui`;
          ctx.fillText(wr.toFixed(0), x, y);
        }
      });
    }
    if (hover >= 0 && !pos.cells[hover] && canMove() && legal(hover)) {
      const [x, y] = at(hover); ctx.globalAlpha = 0.5; ctx.fillStyle = pos.next === BLACK ? '#000' : '#fff';
      ctx.beginPath(); ctx.arc(x, y, R, 0, 7); ctx.fill(); ctx.globalAlpha = 1;
    }
  }
  function eventIdx(ev) {
    const r = canvas.getBoundingClientRect(), { cell, off } = layout(), dpr = canvas.width / r.width;
    const x = Math.round(((ev.clientX - r.left) * dpr - off) / cell), y = Math.round(((ev.clientY - r.top) * dpr - off) / cell);
    return x >= 0 && y >= 0 && x < size && y < size ? y * size + x : -1;
  }
  canvas.addEventListener('mousemove', (e) => { const h = eventIdx(e); if (h !== hover) { hover = h; draw(); } });
  canvas.addEventListener('mouseleave', () => { hover = -1; draw(); });
  canvas.addEventListener('click', (e) => {
    const idx = eventIdx(e); dbg('click', idx, 'ready', ready, 'canMove', canMove());
    if (!ready || !canMove()) return;
    if (idx < 0 || !legal(idx)) return;
    stopThinking(); play(idx);
  });
  $('pass').onclick = () => { if (ready && canMove()) { stopThinking(); play(PASS); } };
  $('undo').onclick = undo; $('newGame').onclick = newGame;
  $('ownership').onchange = draw;
  for (const id of ['model', 'backend', 'threads']) $(id).onchange = () => startEngine();
  for (const id of ['size', 'color', 'komi']) $(id).onchange = newGame;
  function applyMode() {
    const g = mode() === 'game';
    document.querySelectorAll('.gameOnly').forEach((e) => { e.hidden = !g; });
    $('modeHelp').textContent = g ? 'You play against the engine; it gets a fixed time per move.'
      : 'Click to place stones — colours alternate. The engine keeps analysing the current position.';
  }
  $('mode').onchange = () => { applyMode(); newGame(); };
  window.addEventListener('resize', draw);

  // sensible defaults per device; ?model=URL overrides
  $('threads').value = Math.min(8, Math.max(1, (navigator.hardwareConcurrency || 4) >> 1));
  if (navigator.gpu) $('backend').title = 'WebGPU available';
  const q = new URLSearchParams(location.search);
  if (q.get('model')) { const o = document.createElement('option'); o.value = o.textContent = q.get('model'); $('model').prepend(o); $('model').value = o.value; }
  if (q.get('backend')) $('backend').value = q.get('backend');
  applyMode(); rebuild([]); draw(); startEngine();
})();
