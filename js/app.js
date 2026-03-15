// ============================================================
// App Controller — UI logic, visualizations, training loop
// ============================================================

const app = {
  model: null,
  training: false,
  trainTimer: null,
  stepsPerFrame: 10,
  currentPage: 'learn',

  init() {
    this.model = new MiniLM();
    this.bindEvents();
    this.showPage('learn');
    this.initLearnDemos();
  },

  // --- Navigation ---
  showPage(page) {
    this.currentPage = page;
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById(`page-${page}`).classList.add('active');
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`[data-page="${page}"]`).classList.add('active');
  },

  // --- Event binding ---
  bindEvents() {
    // Navigation
    document.querySelectorAll('.nav-tab').forEach(tab => {
      tab.addEventListener('click', () => this.showPage(tab.dataset.page));
    });

    // File upload
    const fileInput = document.getElementById('file-upload');
    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        document.getElementById('training-text').value = ev.target.result;
        this.updateTextStats();
      };
      reader.readAsText(file);
    });

    // Text input
    document.getElementById('training-text').addEventListener('input', () => {
      this.updateTextStats();
    });

    // Sample texts
    document.querySelectorAll('.sample-btn').forEach(btn => {
      btn.addEventListener('click', () => this.loadSample(btn.dataset.sample));
    });

    // Train button
    document.getElementById('btn-train').addEventListener('click', () => this.startTraining());
    document.getElementById('btn-stop').addEventListener('click', () => this.stopTraining());
    document.getElementById('btn-reset').addEventListener('click', () => this.resetModel());

    // Generate
    document.getElementById('btn-generate').addEventListener('click', () => this.generateText());

    // Sliders
    document.getElementById('slider-lr').addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      document.getElementById('val-lr').textContent = val.toFixed(3);
      if (this.model) this.model.learningRate = val;
    });
    document.getElementById('slider-temp').addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      document.getElementById('val-temp').textContent = val.toFixed(2);
    });
    document.getElementById('slider-speed').addEventListener('input', (e) => {
      this.stepsPerFrame = parseInt(e.target.value);
      document.getElementById('val-speed').textContent = e.target.value;
    });
  },

  // --- Learn page interactive demos ---
  initLearnDemos() {
    this.renderTokenizationDemo();
    this.renderPredictionDemo();
  },

  renderTokenizationDemo() {
    const text = "Hello World";
    const container = document.getElementById('tokenization-demo');
    if (!container) return;

    let html = '<div class="demo-label">Input text:</div>';
    html += `<div class="demo-text-display">"${text}"</div>`;
    html += '<div class="demo-arrow">Character Tokenization</div>';
    html += '<div class="token-grid">';
    for (let i = 0; i < text.length; i++) {
      const ch = text[i] === ' ' ? '␣' : text[i];
      html += `<div class="token-cell">
        <span class="token-char">${ch}</span>
        <span class="token-id">${i}</span>
      </div>`;
    }
    html += '</div>';
    container.innerHTML = html;
  },

  renderPredictionDemo() {
    const container = document.getElementById('prediction-demo');
    if (!container) return;

    const context = "The ca";
    const predictions = [
      { char: 't', prob: 0.45 },
      { char: 'r', prob: 0.25 },
      { char: 'n', prob: 0.12 },
      { char: 'b', prob: 0.08 },
      { char: 'l', prob: 0.05 },
      { char: 'p', prob: 0.03 },
      { char: 's', prob: 0.02 },
    ];

    let html = '<div class="demo-label">Context window:</div>';
    html += '<div class="context-display">';
    for (const ch of context) {
      html += `<span class="ctx-char">${ch === ' ' ? '␣' : ch}</span>`;
    }
    html += '<span class="ctx-char ctx-next">?</span>';
    html += '</div>';
    html += '<div class="demo-arrow">Neural Network Prediction</div>';
    html += '<div class="prob-bars">';
    for (const p of predictions) {
      const pct = (p.prob * 100).toFixed(1);
      html += `<div class="prob-bar-row">
        <span class="prob-char">"${p.char}"</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width:${pct}%"></div>
        </div>
        <span class="prob-pct">${pct}%</span>
      </div>`;
    }
    html += '</div>';
    container.innerHTML = html;
  },

  // --- Lab functionality ---

  updateTextStats() {
    const text = document.getElementById('training-text').value;
    const chars = text.length;
    const unique = new Set(text).size;
    const words = text.split(/\s+/).filter(w => w.length > 0).length;
    document.getElementById('text-stats').innerHTML =
      `${chars.toLocaleString()} characters &middot; ${words.toLocaleString()} words &middot; ${unique} unique characters`;
  },

  loadSample(name) {
    const samples = {
      shakespeare: `To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them. To die, to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office, and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscovered country, from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment,
With this regard their currents turn awry
And lose the name of action.`,
      nursery: `Twinkle, twinkle, little star, how I wonder what you are!
Up above the world so high, like a diamond in the sky.
Twinkle, twinkle, little star, how I wonder what you are!

Mary had a little lamb, its fleece was white as snow.
And everywhere that Mary went, the lamb was sure to go.
It followed her to school one day, that was against the rules.
It made the children laugh and play, to see a lamb at school.

Humpty Dumpty sat on a wall, Humpty Dumpty had a great fall.
All the king's horses and all the king's men
Couldn't put Humpty together again.

Jack and Jill went up the hill to fetch a pail of water.
Jack fell down and broke his crown, and Jill came tumbling after.
Up Jack got and home did trot, as fast as he could caper.
He went to bed to mend his head with vinegar and brown paper.

Hey diddle diddle, the cat and the fiddle,
The cow jumped over the moon.
The little dog laughed to see such sport,
And the dish ran away with the spoon.

Baa, baa, black sheep, have you any wool?
Yes sir, yes sir, three bags full.
One for the master, one for the dame,
And one for the little boy who lives down the lane.`,
      code: `function fibonacci(n) {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}

function factorial(n) {
  if (n <= 1) return 1;
  return n * factorial(n - 1);
}

function isPrime(n) {
  if (n <= 1) return false;
  if (n <= 3) return true;
  if (n % 2 === 0) return false;
  for (let i = 3; i * i <= n; i += 2) {
    if (n % i === 0) return false;
  }
  return true;
}

function bubbleSort(arr) {
  const n = arr.length;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n - i - 1; j++) {
      if (arr[j] > arr[j + 1]) {
        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
      }
    }
  }
  return arr;
}

function binarySearch(arr, target) {
  let left = 0;
  let right = arr.length - 1;
  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    if (arr[mid] === target) return mid;
    if (arr[mid] < target) left = mid + 1;
    else right = mid - 1;
  }
  return -1;
}`,
    };

    const text = samples[name] || '';
    document.getElementById('training-text').value = text;
    this.updateTextStats();
  },

  // --- Training ---

  startTraining() {
    const text = document.getElementById('training-text').value.trim();
    if (text.length < 20) {
      alert('Please enter at least 20 characters of training text.');
      return;
    }

    if (!this.model.trained && this.model.step === 0) {
      const lr = parseFloat(document.getElementById('slider-lr').value);
      this.model.learningRate = lr;
      const info = this.model.prepareData(text);
      this.showModelInfo(info);
    }

    this.training = true;
    document.getElementById('btn-train').style.display = 'none';
    document.getElementById('btn-stop').style.display = 'inline-flex';
    document.getElementById('btn-reset').style.display = 'inline-flex';
    document.getElementById('training-viz').classList.add('active');
    document.getElementById('generation-section').style.display = 'block';

    this.trainLoop();
  },

  trainLoop() {
    if (!this.training) return;

    let lastResult;
    for (let i = 0; i < this.stepsPerFrame; i++) {
      lastResult = this.model.trainStep();
    }

    this.updateTrainingViz(lastResult);

    this.trainTimer = requestAnimationFrame(() => this.trainLoop());
  },

  stopTraining() {
    this.training = false;
    this.model.trained = true;
    if (this.trainTimer) cancelAnimationFrame(this.trainTimer);
    document.getElementById('btn-train').style.display = 'inline-flex';
    document.getElementById('btn-train').innerHTML = '<span class="btn-icon">&#9654;</span> Resume';
    document.getElementById('btn-stop').style.display = 'none';
  },

  resetModel() {
    this.stopTraining();
    this.model = new MiniLM();
    document.getElementById('btn-train').innerHTML = '<span class="btn-icon">&#9654;</span> Train Model';
    document.getElementById('training-viz').classList.remove('active');
    document.getElementById('generation-section').style.display = 'none';
    document.getElementById('model-info').innerHTML = '';

    // Clear canvases
    ['loss-canvas', 'embedding-canvas'].forEach(id => {
      const c = document.getElementById(id);
      if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height);
    });
    document.getElementById('live-predictions').innerHTML = '';
    document.getElementById('step-counter').textContent = 'Step: 0';
    document.getElementById('loss-value').textContent = 'Loss: —';
    document.getElementById('generation-output').innerHTML = '';
    document.getElementById('generation-steps-viz').innerHTML = '';
  },

  showModelInfo(info) {
    const paramCount = this.model.getParamCount();
    document.getElementById('model-info').innerHTML = `
      <div class="info-grid">
        <div class="info-item">
          <div class="info-label">Vocabulary</div>
          <div class="info-value">${info.vocabSize} chars</div>
        </div>
        <div class="info-item">
          <div class="info-label">Training examples</div>
          <div class="info-value">${info.numExamples.toLocaleString()}</div>
        </div>
        <div class="info-item">
          <div class="info-label">Parameters</div>
          <div class="info-value">${paramCount.toLocaleString()}</div>
        </div>
        <div class="info-item">
          <div class="info-label">Architecture</div>
          <div class="info-value">Embed(${this.model.embedDim}) → Hidden(${this.model.hiddenSize}) → Softmax</div>
        </div>
      </div>
    `;
  },

  // --- Visualization ---

  updateTrainingViz(result) {
    // Step counter & loss
    document.getElementById('step-counter').textContent = `Step: ${result.step.toLocaleString()}`;
    document.getElementById('loss-value').textContent = `Loss: ${result.avgLoss.toFixed(4)} · LR: ${result.effectiveLR.toFixed(4)}`;

    // Loss chart
    this.drawLossChart();

    // Top predictions
    this.drawPredictions(result);

    // Embedding viz (every 20 steps)
    if (result.step % 20 === 0) {
      this.drawEmbeddingViz();
    }
  },

  drawLossChart() {
    const canvas = document.getElementById('loss-canvas');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const parent = canvas.parentElement;
    // Use clientWidth to avoid feedback loop (excludes overflow)
    const W = parent.clientWidth - 32; // account for panel padding
    const H = 200;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    ctx.scale(dpr, dpr);
    const pad = { top: 20, right: 20, bottom: 30, left: 50 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    ctx.clearRect(0, 0, W, H);

    const losses = this.model.losses;
    if (losses.length < 2) return;

    // Downsample for performance
    const maxPoints = 500;
    let data = losses;
    if (losses.length > maxPoints) {
      const step = Math.ceil(losses.length / maxPoints);
      data = [];
      for (let i = 0; i < losses.length; i += step) {
        let sum = 0, count = 0;
        for (let j = i; j < Math.min(i + step, losses.length); j++) {
          sum += losses[j]; count++;
        }
        data.push(sum / count);
      }
    }

    const minL = Math.min(...data) * 0.95;
    const maxL = Math.max(...data) * 1.05;

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH * i / 4);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(W - pad.right, y);
      ctx.stroke();
    }

    // Axis labels
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '11px "IBM Plex Mono", monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const val = maxL - (maxL - minL) * i / 4;
      const y = pad.top + (plotH * i / 4);
      ctx.fillText(val.toFixed(2), pad.left - 8, y + 4);
    }
    ctx.textAlign = 'center';
    ctx.fillText('Training Steps', W / 2, H - 4);

    // Loss line
    ctx.beginPath();
    ctx.strokeStyle = '#4f8ff7';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    for (let i = 0; i < data.length; i++) {
      const x = pad.left + (i / (data.length - 1)) * plotW;
      const y = pad.top + ((maxL - data[i]) / (maxL - minL)) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Gradient fill
    const gradient = ctx.createLinearGradient(0, pad.top, 0, H - pad.bottom);
    gradient.addColorStop(0, 'rgba(79, 143, 247, 0.15)');
    gradient.addColorStop(1, 'rgba(79, 143, 247, 0)');
    ctx.lineTo(pad.left + plotW, H - pad.bottom);
    ctx.lineTo(pad.left, H - pad.bottom);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();
  },

  drawPredictions(result) {
    const container = document.getElementById('live-predictions');
    let html = `<div class="pred-context">Context: <span class="mono">"${this.escapeHtml(result.contextChars)}"</span> → Target: <span class="mono">"${this.escapeHtml(result.targetChar)}"</span></div>`;
    html += '<div class="pred-bars">';
    for (const p of result.topPredictions.slice(0, 8)) {
      const pct = (p.prob * 100).toFixed(1);
      const isTarget = p.char === result.targetChar;
      const displayChar = p.char === ' ' ? '␣' : p.char === '\n' ? '↵' : p.char;
      html += `<div class="pred-bar-row ${isTarget ? 'pred-correct' : ''}">
        <span class="pred-char">${this.escapeHtml(displayChar)}</span>
        <div class="pred-bar-track">
          <div class="pred-bar-fill" style="width:${Math.min(pct, 100)}%"></div>
        </div>
        <span class="pred-pct">${pct}%</span>
      </div>`;
    }
    html += '</div>';
    container.innerHTML = html;
  },

  drawEmbeddingViz() {
    const canvas = document.getElementById('embedding-canvas');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const parent = canvas.parentElement;
    const size = Math.min(parent.clientWidth - 32, 280);
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = size + 'px';
    canvas.style.height = size + 'px';
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, size, size);

    const points = this.model.getEmbedding2D();
    if (points.length === 0) return;

    const pad = 30;
    const plotSize = size - 2 * pad;

    // Find bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of points) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;

    // Draw points
    for (const p of points) {
      const x = pad + ((p.x - minX) / rangeX) * plotSize;
      const y = pad + ((p.y - minY) / rangeY) * plotSize;
      const displayChar = p.char === ' ' ? '␣' : p.char === '\n' ? '↵' : p.char;

      // Dot
      ctx.beginPath();
      ctx.fillStyle = this.charColor(p.char);
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();

      // Label
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.font = '11px "IBM Plex Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(displayChar, x, y - 8);
    }
  },

  charColor(ch) {
    if (/[a-z]/.test(ch)) return '#4f8ff7';
    if (/[A-Z]/.test(ch)) return '#7c5ff7';
    if (/[0-9]/.test(ch)) return '#f7a84f';
    if (/\s/.test(ch)) return '#4fc78f';
    return '#f75f8a';
  },

  // --- Generation ---

  generateText() {
    const seed = document.getElementById('prompt-input').value || '';
    const temp = parseFloat(document.getElementById('slider-temp').value);
    const length = 200;

    if (!this.model.C) {
      alert('Please train the model first!');
      return;
    }

    const { text, steps } = this.model.generate(seed, length, temp);

    // Display generated text with character-by-character animation
    const output = document.getElementById('generation-output');
    output.innerHTML = '';

    // Show seed in different style
    const seedSpan = document.createElement('span');
    seedSpan.className = 'gen-seed';
    seedSpan.textContent = seed;
    output.appendChild(seedSpan);

    const genChars = text.slice(seed.length);
    let charIdx = 0;

    const typeChar = () => {
      if (charIdx >= genChars.length) return;
      const span = document.createElement('span');
      span.className = 'gen-char';
      span.textContent = genChars[charIdx];
      // Color by confidence
      if (steps[charIdx]) {
        const conf = steps[charIdx].selectedProb;
        span.style.opacity = 0.4 + conf * 0.6;
        span.title = `"${steps[charIdx].selected}" (${(conf * 100).toFixed(1)}% confidence)`;
      }
      output.appendChild(span);
      charIdx++;
      if (charIdx < genChars.length) {
        setTimeout(typeChar, 20);
      } else {
        this.showGenerationSteps(steps);
      }
    };
    typeChar();
  },

  showGenerationSteps(steps) {
    const container = document.getElementById('generation-steps-viz');
    if (steps.length === 0) return;

    let html = '<h4>Generation Steps (click to explore)</h4>';
    html += '<div class="gen-steps-scroll">';
    for (let i = 0; i < Math.min(steps.length, 50); i++) {
      const s = steps[i];
      html += `<div class="gen-step" data-step="${i}">
        <div class="gen-step-num">#${i + 1}</div>
        <div class="gen-step-ctx">
          <span class="mono">"${this.escapeHtml(s.context)}"</span> →
          <span class="gen-step-selected">"${this.escapeHtml(s.selected)}"</span>
          <span class="gen-step-conf">(${(s.selectedProb * 100).toFixed(1)}%)</span>
        </div>
        <div class="gen-step-probs">
          ${s.topPredictions.slice(0, 5).map(p => {
            const displayChar = p.char === ' ' ? '␣' : p.char === '\n' ? '↵' : p.char;
            return `<span class="gen-prob-pill ${p.char === s.selected ? 'selected' : ''}">${this.escapeHtml(displayChar)} ${(p.prob * 100).toFixed(0)}%</span>`;
          }).join('')}
        </div>
      </div>`;
    }
    html += '</div>';
    container.innerHTML = html;
  },

  escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  },
};

document.addEventListener('DOMContentLoaded', () => app.init());
