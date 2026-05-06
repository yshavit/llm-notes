function render({ model, el }) {
  // ── Styles ────────────────────────────────────────────────────────────────
  const style = document.createElement('style');
  style.textContent = `
    * { box-sizing: border-box; margin: 0; padding: 0; }

    .widget {
      font-family: 'Georgia', serif;
      background: #fafaf8;
      border: 1px solid #d8d4cc;
      border-radius: 6px;
      padding: 1.2em;
      max-width: 560px;
      color: #222;
    }

    .widget h3 {
      font-size: 0.78em;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #888;
      margin-bottom: 1em;
      font-family: monospace;
    }

    .controls {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.6em 1.2em;
      margin-bottom: 1em;
    }

    .control-group label {
      display: block;
      font-size: 0.72em;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #888;
      margin-bottom: 0.25em;
      font-family: monospace;
    }

    .control-group input[type=range] {
      width: 100%;
      accent-color: #3a6ea5;
    }

    .control-group .val {
      font-size: 0.82em;
      color: #444;
      font-family: monospace;
    }

    canvas {
      display: block;
      width: 100%;
      border: 1px solid #ddd;
      border-radius: 4px;
      background: #fff;
      margin-bottom: 0.9em;
    }

    .params {
      display: flex;
      gap: 2em;
      margin-bottom: 0.9em;
      font-family: monospace;
      font-size: 0.88em;
    }

    .param-block { display: flex; flex-direction: column; gap: 0.15em; }
    .param-label { font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.08em; color: #999; }
    .param-value { font-size: 1.05em; color: #222; }
    .param-grad  { font-size: 0.72em; color: #b04040; }

    .buttons {
      display: flex;
      gap: 0.6em;
    }

    button {
      font-family: monospace;
      font-size: 0.82em;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      padding: 0.35em 0.85em;
      border: 1px solid #aaa;
      border-radius: 3px;
      background: #f0ede8;
      cursor: pointer;
      color: #333;
      transition: background 0.12s;
    }
    button:hover { background: #e2ddd6; }
    button.primary { background: #3a6ea5; color: #fff; border-color: #2d5687; }
    button.primary:hover { background: #2d5687; }

    .iter {
      margin-left: auto;
      font-family: monospace;
      font-size: 0.78em;
      color: #aaa;
      align-self: center;
    }
  `;
  el.appendChild(style);

  // ── DOM ───────────────────────────────────────────────────────────────────
  const widget = document.createElement('div');
  widget.className = 'widget';

  widget.innerHTML = `
    <h3>Linear model · backprop demo</h3>

    <div class="controls">
      <div class="control-group">
        <label>True slope (a<sub>true</sub>)</label>
        <input type="range" id="trueA" min="-2" max="2" step="0.05" value="0.8">
        <span class="val" id="trueAVal">0.80</span>
      </div>
      <div class="control-group">
        <label>True intercept (b<sub>true</sub>)</label>
        <input type="range" id="trueB" min="-1" max="1" step="0.05" value="0.3">
        <span class="val" id="trueBVal">0.30</span>
      </div>
      <div class="control-group">
        <label>Learning rate (𝜂)</label>
        <input type="range" id="eta" min="0.001" max="1.0" step="0.001" value="0.01">
        <span class="val" id="etaVal">0.01</span>
      </div>
      <div class="control-group">
        <label>Speed</label>
        <input type="range" id="speed" min="1" max="20" step="1" value="5">
        <span class="val" id="speedVal">5 iter/s</span>
      </div>
    </div>

    <canvas id="plot" height="300"></canvas>

    <div class="params">
      <div class="param-block">
        <span class="param-label">a (learned)</span>
        <span class="param-value" id="aVal">—</span>
        <span class="param-grad"  id="aGrad"></span>
      </div>
      <div class="param-block">
        <span class="param-label">b (learned)</span>
        <span class="param-value" id="bVal">—</span>
        <span class="param-grad"  id="bGrad"></span>
      </div>
      <div class="param-block">
        <span class="param-label">Loss (avg)</span>
        <span class="param-value" id="lossVal">—</span>
      </div>
    </div>

    <div class="buttons">
      <button class="primary" id="playBtn">▶ Play</button>
      <button id="resetBtn">↺ Reset</button>
      <span class="iter" id="iterCount">iter 0</span>
    </div>
  `;

  el.appendChild(widget);

  // ── State ─────────────────────────────────────────────────────────────────
  const canvas = widget.querySelector('#plot');
  const ctx = canvas.getContext('2d');

  // Make canvas crisp on HiDPI
  function resizeCanvas() {
    const w = canvas.clientWidth;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = 300 * dpr;
    ctx.scale(dpr, dpr);
  }
  resizeCanvas();

  let trueA = 0.8, trueB = 0.3;
  let a, b;
  let eta = 0.1, speed = 5;
  let iter = 0;
  let playing = false;
  let intervalId = null;

  // Generate fixed training points from true line + small noise
  let trainPoints = [];
  function genPoints() {
    trainPoints = [];
    const nPoints = 120;
    for (let i = 0; i <= nPoints; i++) {
      const x = -1.8 + i * (3.6 / nPoints);
      const noise = (Math.random() - 0.5) * 1.5;
      trainPoints.push({ x, y: trueA * x + trueB + noise });
    }
  }

  function resetState() {
    a = (Math.random() * 2 - 1).toFixed(3) * 1;
    b = (Math.random() * 1 - 0.5).toFixed(3) * 1;
    iter = 0;
    genPoints();
    draw();
    updateParamDisplay(null, null, null);
    widget.querySelector('#iterCount').textContent = 'iter 0';
  }

  // ── Coordinate helpers ────────────────────────────────────────────────────
  const PAD = 28;
  function toCanvasX(x) {
    const w = canvas.clientWidth;
    return PAD + (x + 2) / 4 * (w - 2 * PAD);
  }
  function toCanvasY(y) {
    return PAD + (2 - y) / 4 * (300 - 2 * PAD);
  }

  // ── Drawing ───────────────────────────────────────────────────────────────
  function draw() {
    const w = canvas.clientWidth;
    const h = 300;
    ctx.clearRect(0, 0, w, h);

    // Grid lines (light)
    ctx.strokeStyle = '#efefef';
    ctx.lineWidth = 1;
    for (let v = -2; v <= 2; v++) {
      ctx.beginPath(); ctx.moveTo(toCanvasX(v), PAD); ctx.lineTo(toCanvasX(v), h - PAD); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(PAD, toCanvasY(v)); ctx.lineTo(w - PAD, toCanvasY(v)); ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(toCanvasX(-2), toCanvasY(0)); ctx.lineTo(toCanvasX(2), toCanvasY(0)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(toCanvasX(0), toCanvasY(-2)); ctx.lineTo(toCanvasX(0), toCanvasY(2)); ctx.stroke();

    // Axis tick labels
    ctx.fillStyle = '#aaa';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    for (let v = -2; v <= 2; v++) {
      if (v === 0) continue;
      ctx.fillText(v, toCanvasX(v), toCanvasY(0) + 12);
      ctx.textAlign = 'right';
      ctx.fillText(v, toCanvasX(0) - 4, toCanvasY(v) + 3);
      ctx.textAlign = 'center';
    }

    // True line (blue, 2px)
    ctx.strokeStyle = '#3a6ea5';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(toCanvasX(-2), toCanvasY(trueA * -2 + trueB));
    ctx.lineTo(toCanvasX(2), toCanvasY(trueA * 2 + trueB));
    ctx.stroke();

    // Learned line (red, 1px)
    ctx.strokeStyle = '#c0392b';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(toCanvasX(-2), toCanvasY(a * -2 + b));
    ctx.lineTo(toCanvasX(2), toCanvasY(a * 2 + b));
    ctx.stroke();

    // Training points (small dots, matching true line color)
    ctx.fillStyle = '#3a6ea5';
    for (const p of trainPoints) {
      ctx.beginPath();
      ctx.arc(toCanvasX(p.x), toCanvasY(p.y), 1.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Legend
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#3a6ea5'; ctx.fillRect(PAD + 4, PAD + 4, 14, 2); ctx.fillText('true', PAD + 20, PAD + 8);
    ctx.fillStyle = '#c0392b'; ctx.fillRect(PAD + 4, PAD + 16, 14, 1.5); ctx.fillText('learned', PAD + 20, PAD + 20);
  }

  function updateParamDisplay(gradA, gradB, loss) {
    widget.querySelector('#aVal').textContent = a.toFixed(4);
    widget.querySelector('#bVal').textContent = b.toFixed(4);
    if (gradA !== null) {
      widget.querySelector('#aGrad').textContent = `∂L/∂a = ${gradA.toFixed(4)}`;
      widget.querySelector('#bGrad').textContent = `∂L/∂b = ${gradB.toFixed(4)}`;
      widget.querySelector('#lossVal').textContent = loss.toFixed(5);
    }
    widget.querySelector('#iterCount').textContent = `iter ${iter}`;
  }

  // ── Backprop step ─────────────────────────────────────────────────────────
  // y(x) = ax + b
  // L = mean over training set of (y(x) - y_true)^2
  // ∂L/∂a = mean of 2(y(x) - y_true) * x
  // ∂L/∂b = mean of 2(y(x) - y_true)
  function step() {
    let sumGradA = 0, sumGradB = 0, sumLoss = 0;
    for (const p of trainPoints) {
      const yPred = a * p.x + b;
      const err = yPred - p.y;
      sumLoss += err * err;
      sumGradA += 2 * err * p.x;
      sumGradB += 2 * err;
    }
    const n = trainPoints.length;
    const gradA = sumGradA / n;
    const gradB = sumGradB / n;
    const loss = sumLoss / n;

    a -= eta * gradA;
    b -= eta * gradB;
    iter++;

    draw();
    updateParamDisplay(gradA, gradB, loss);
  }

  // ── Controls wiring ───────────────────────────────────────────────────────
  function bindSlider(id, valId, parse, fmt, onChange) {
    const sl = widget.querySelector(`#${id}`);
    const lbl = widget.querySelector(`#${valId}`);
    sl.addEventListener('input', () => {
      const v = parse(sl.value);
      lbl.textContent = fmt(v);
      onChange(v);
    });
    // init label
    lbl.textContent = fmt(parse(sl.value));
  }

  bindSlider('trueA', 'trueAVal', parseFloat, v => v.toFixed(2), v => { trueA = v; resetState(); });
  bindSlider('trueB', 'trueBVal', parseFloat, v => v.toFixed(2), v => { trueB = v; resetState(); });
  bindSlider('eta', 'etaVal', parseFloat, v => v.toFixed(3), v => { eta = v; });
  bindSlider('speed', 'speedVal', parseInt, v => `${v} iter/s`, v => {
    speed = v;
    if (playing) { clearInterval(intervalId); intervalId = setInterval(step, 1000 / speed); }
  });

  const playBtn = widget.querySelector('#playBtn');
  const resetBtn = widget.querySelector('#resetBtn');

  playBtn.addEventListener('click', () => {
    playing = !playing;
    playBtn.textContent = playing ? '⏸ Pause' : '▶ Play';
    if (playing) {
      intervalId = setInterval(step, 1000 / speed);
    } else {
      clearInterval(intervalId);
    }
  });

  resetBtn.addEventListener('click', () => {
    playing = false;
    clearInterval(intervalId);
    playBtn.textContent = '▶ Play';
    resetState();
  });

  // ── Init ──────────────────────────────────────────────────────────────────
  resetState();

  return () => clearInterval(intervalId);
}

export default { render };
