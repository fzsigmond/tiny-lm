// ============================================================
// App Controller — UI logic, visualizations, training loop
// ============================================================

const app = {
  model: null,
  training: false,
  trainTimer: null,
  stepsPerFrame: 10,
  currentPage: 'learn',
  currentStep: 1,

  init() {
    this.model = new MiniLM();
    applyI18n();
    this.bindEvents();
    this.showPage('learn');
    this.initLearnDemos();
    this.updateStepStates();
  },

  // Re-render everything that is built in JavaScript after a language switch.
  onLangChange() {
    this.initLearnDemos();
    // Counters keep their numbers; only the words change.
    if (this.lastResult) this.updateTrainingViz(this.lastResult);
    else {
      document.getElementById('step-counter').textContent = t('js.step', { n: 0 });
      document.getElementById('loss-value').textContent = t('js.lossEmpty');
    }
    if (document.getElementById('training-text').value) this.updateTextStats();
    if (!this.training && document.getElementById('btn-train').style.display !== 'none') {
      document.getElementById('btn-train').innerHTML = `<span class="btn-icon">&#9654;</span> ${t('js.trainModel')}`;
    }
    const pause = document.getElementById('btn-pause');
    if (pause.style.display !== 'none') {
      pause.innerHTML = this.training ? `<span class="btn-icon">&#9632;</span> ${t('js.pause')}` : `<span class="btn-icon">&#9654;</span> ${t('js.resume')}`;
    }
    const tokBtn = document.getElementById('btn-tokenize');
    tokBtn.textContent = this.model && this.model.tokenizer ? t('js.retokenize') : t('js.tokenize');
    if (this.model && this.model.tokenizer) {
      this.showTokenInfo({
        vocabSize: this.model.vocabSize,
        numMerges: this.model.tokenizer.merges.length,
        compressionRatio: this.model.rawTokenIds ? (document.getElementById('training-text').value.length / this.model.rawTokenIds.length).toFixed(1) : '—',
      });
      this.updateTrainingPreview();
    }
    this.showCopyDetail();
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
        this.suggestVocabSize();
      };
      reader.readAsText(file);
    });

    // Text input
    document.getElementById('training-text').addEventListener('input', () => {
      this.updateTextStats();
      this.suggestVocabSize();
    });

    // Sample texts
    document.querySelectorAll('.sample-btn').forEach(btn => {
      btn.addEventListener('click', () => this.loadSample(btn.dataset.sample));
    });

    // Step header clicks (backward navigation)
    document.querySelectorAll('.lab-step-header').forEach(header => {
      header.addEventListener('click', () => {
        const stepNum = parseInt(header.parentElement.dataset.step);
        if (stepNum < this.currentStep) {
          this.goToStep(stepNum);
        }
      });
    });

    // Step navigation buttons
    document.getElementById('btn-next-1').addEventListener('click', () => {
      const text = document.getElementById('training-text').value.trim();
      if (text.length < 20) {
        alert(t('js.minChars'));
        return;
      }
      this.goToStep(2, true);
    });
    document.getElementById('btn-next-2').addEventListener('click', () => this.goToStep(3, true));
    document.getElementById('btn-pause').addEventListener('click', () => {
      if (this.training) {
        this.stopTraining();
      } else {
        this.startTraining();
      }
    });
    document.getElementById('btn-generate-step').addEventListener('click', () => {
      if (this.training) this.stopTraining();
      this.goToStep(4, true);
    });
    document.getElementById('btn-back-train').addEventListener('click', () => {
      this.goToStep(3, true);
    });

    // Tokenize button
    document.getElementById('btn-tokenize').addEventListener('click', () => this.tokenizeText());

    // Train button
    document.getElementById('btn-train').addEventListener('click', () => this.startTraining());
    document.getElementById('btn-reset').addEventListener('click', () => this.resetModel());

    // Generate
    document.getElementById('btn-generate').addEventListener('click', () => this.generateText());

    // Sliders
    document.getElementById('slider-embed').addEventListener('input', (e) => {
      document.getElementById('val-embed').textContent = e.target.value;
      this.updateTrainingPreview();
    });
    document.getElementById('slider-context').addEventListener('input', (e) => {
      document.getElementById('val-context').textContent = e.target.value;
      this.updateTrainingPreview();
    });
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
    document.getElementById('slider-vocab').addEventListener('input', (e) => {
      const val = parseInt(e.target.value);
      document.getElementById('val-vocab').textContent = val;
      if (this.model) this.model.maxVocabSize = val;
    });
    document.getElementById('slider-topk').addEventListener('input', (e) => {
      document.getElementById('val-topk').textContent = e.target.value;
    });
    document.getElementById('slider-topp').addEventListener('input', (e) => {
      document.getElementById('val-topp').textContent = parseFloat(e.target.value).toFixed(2);
    });
    document.getElementById('slider-reppen').addEventListener('input', (e) => {
      document.getElementById('val-reppen').textContent = parseFloat(e.target.value).toFixed(2);
    });
    document.getElementById('slider-length').addEventListener('input', (e) => {
      document.getElementById('val-length').textContent = e.target.value;
    });
  },

  // --- Learn page interactive demos ---
  initLearnDemos() {
    this.renderTokenizationDemo();
    this.renderPredictionDemo();
  },

  renderTokenizationDemo() {
    const container = document.getElementById('tokenization-demo');
    if (!container) return;

    // Show BPE merge process on example text. The merges are the real ones
    // the tokenizer makes on each sentence (pairs that appear at least twice).
    const demo = LANG === 'pt' ? {
      text: 'o gato e o pato no prato',
      steps: [
        { pair: ['a', 't'], result: 'at', tokens: ['o', '␣', 'g', 'at', 'o', '␣', 'e', '␣', 'o', '␣', 'p', 'at', 'o', '␣', 'n', 'o', '␣', 'p', 'r', 'at', 'o'] },
        { pair: ['at', 'o'], result: 'ato', tokens: ['o', '␣', 'g', 'ato', '␣', 'e', '␣', 'o', '␣', 'p', 'ato', '␣', 'n', 'o', '␣', 'p', 'r', 'ato'] },
      ],
    } : {
      text: 'the cat sat on the mat',
      steps: [
        { pair: ['a', 't'], result: 'at', tokens: ['t', 'h', 'e', '␣', 'c', 'at', '␣', 's', 'at', '␣', 'o', 'n', '␣', 't', 'h', 'e', '␣', 'm', 'at'] },
        { pair: ['t', 'h'], result: 'th', tokens: ['th', 'e', '␣', 'c', 'at', '␣', 's', 'at', '␣', 'o', 'n', '␣', 'th', 'e', '␣', 'm', 'at'] },
        { pair: ['th', 'e'], result: 'the', tokens: ['the', '␣', 'c', 'at', '␣', 's', 'at', '␣', 'o', 'n', '␣', 'the', '␣', 'm', 'at'] },
      ],
    };
    const text = demo.text;
    const charTokens = text.split('');

    let html = `<div class="demo-label">${t('demo.inputText')}</div>`;
    html += `<div class="demo-text-display">"${text}"</div>`;
    html += `<div class="demo-arrow">${t('demo.split')}</div>`;
    html += '<div class="token-grid">';
    for (const ch of charTokens) {
      const display = ch === ' ' ? '␣' : ch;
      html += `<div class="token-cell"><span class="token-char">${display}</span></div>`;
    }
    html += '</div>';

    for (const step of demo.steps) {
      html += `<div class="demo-arrow">${t('demo.merge', { a: step.pair[0], b: step.pair[1], r: step.result })}</div>`;
      html += '<div class="token-grid">';
      for (const tok of step.tokens) {
        const isMerged = tok === step.result;
        html += `<div class="token-cell ${isMerged ? 'token-merged' : ''}"><span class="token-char">${tok}</span></div>`;
      }
      html += '</div>';
    }

    const finalCount = demo.steps[demo.steps.length - 1].tokens.length;
    html += `<div class="merge-summary">${t('demo.summary', { chars: text.length, tokens: finalCount, ratio: (text.length / finalCount).toFixed(1) })}</div>`;
    container.innerHTML = html;
  },

  renderPredictionDemo() {
    const container = document.getElementById('prediction-demo');
    if (!container) return;

    const contextTokens = LANG === 'pt' ? ['o', ' ', 'ga'] : ['the', ' ', 'ca'];
    const predictions = LANG === 'pt' ? [
      { token: 'to', prob: 0.42 },
      { token: 't', prob: 0.18 },
      { token: 'l', prob: 0.15 },
      { token: 'r', prob: 0.10 },
      { token: 'nh', prob: 0.06 },
      { token: 'ma', prob: 0.05 },
      { token: 's', prob: 0.04 },
    ] : [
      { token: 't', prob: 0.42 },
      { token: 'ts', prob: 0.18 },
      { token: 'r', prob: 0.15 },
      { token: 'n', prob: 0.10 },
      { token: 'me', prob: 0.06 },
      { token: 'b', prob: 0.05 },
      { token: 'se', prob: 0.04 },
    ];

    let html = `<div class="demo-label">${t('demo.contextTokens')}</div>`;
    html += '<div class="context-display">';
    for (const tok of contextTokens) {
      const display = tok === ' ' ? '␣' : tok;
      html += `<span class="ctx-char">${display}</span>`;
    }
    html += '<span class="ctx-char ctx-next">?</span>';
    html += '</div>';
    html += `<div class="demo-arrow">${t('demo.nnPrediction')}</div>`;
    html += '<div class="prob-bars">';
    for (const p of predictions) {
      const pct = (p.prob * 100).toFixed(1);
      const display = p.token === ' ' ? '␣' : p.token;
      html += `<div class="prob-bar-row">
        <span class="prob-char">"${display}"</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width:${pct}%"></div>
        </div>
        <span class="prob-pct">${pct}%</span>
      </div>`;
    }
    html += '</div>';
    container.innerHTML = html;
  },

  // --- Step navigation ---

  goToStep(step, skipWarning = false) {
    if (step === this.currentStep) return;

    // Going backward
    if (step < this.currentStep && !skipWarning) {
      // No warning needed for 4→3 toggle
      if (!(this.currentStep === 4 && step === 3)) {
        if (!confirm(t('js.goBack'))) return;
      }
    }

    // Reset state for steps we're going back past
    if (step < this.currentStep) {
      if (step <= 2 && this.currentStep >= 3) {
        // Going back to step 1 or 2 from step 3+: stop training, reset model
        this.stopTraining();
        this.model = new MiniLM();
        this.clearTrainingUI();
      }
      if (step <= 2) {
        // Going back to step 1 or 2: also reset tokenization
        this.clearTokenizationUI();
      }
    }

    this.currentStep = step;
    this.updateStepStates();

    if (step === 3) this.updateTrainingPreview();

    // Scroll to the active step
    const stepEl = document.getElementById(`step-${step}`);
    if (stepEl) stepEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
  },

  updateStepStates() {
    for (let i = 1; i <= 4; i++) {
      const el = document.getElementById(`step-${i}`);
      if (!el) continue;
      el.classList.remove('active', 'completed', 'collapsed', 'locked');
      if (i < this.currentStep) {
        el.classList.add('collapsed', 'completed');
      } else if (i === this.currentStep) {
        el.classList.add('active');
      } else {
        el.classList.add('collapsed', 'locked');
      }
    }
  },

  clearTrainingUI() {
    this.lastResult = null;
    document.getElementById('training-viz').classList.remove('active');
    document.getElementById('model-info').innerHTML = '';
    document.getElementById('slider-embed').disabled = false;
    document.getElementById('slider-context').disabled = false;
    document.getElementById('btn-train').innerHTML = `<span class="btn-icon">&#9654;</span> ${t('js.trainModel')}`;
    document.getElementById('btn-train').style.display = 'inline-flex';
    document.getElementById('btn-pause').style.display = 'none';
    document.getElementById('btn-reset').style.display = 'none';
    document.getElementById('btn-generate-step').style.display = 'none';
    ['loss-canvas', 'embedding-canvas'].forEach(id => {
      const c = document.getElementById(id);
      if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height);
    });
    document.getElementById('live-predictions').innerHTML = '';
    document.getElementById('step-counter').textContent = t('js.step', { n: 0 });
    document.getElementById('loss-value').textContent = t('js.lossEmpty');
    document.getElementById('generation-output').innerHTML = '';
    document.getElementById('generation-steps-viz').innerHTML = '';
  },

  clearTokenizationUI() {
    document.getElementById('tokenization-viz').style.display = 'none';
    document.getElementById('btn-tokenize').disabled = false;
    document.getElementById('btn-tokenize').textContent = t('js.tokenize');
    document.getElementById('btn-next-2').style.display = 'none';
  },

  // --- Lab functionality ---

  updateTextStats() {
    const text = document.getElementById('training-text').value;
    const chars = text.length;
    const unique = new Set(text).size;
    const words = text.split(/\s+/).filter(w => w.length > 0).length;
    document.getElementById('text-stats').innerHTML =
      t('js.textStats', { chars: chars.toLocaleString(LANG === 'pt' ? 'pt-BR' : undefined), words: words.toLocaleString(LANG === 'pt' ? 'pt-BR' : undefined), unique });
  },

  loadSample(name) {
    const set = LANG === 'pt' && typeof SAMPLES_PT !== 'undefined' ? SAMPLES_PT : SAMPLES;
    const text = set[name] || SAMPLES[name] || '';
    document.getElementById('training-text').value = text;
    this.updateTextStats();
    this.suggestVocabSize();
  },

  // When the text changes, start the vocab slider at a size that fits it
  // (see BPETokenizer.suggestVocabSize). The slider can still be moved.
  suggestVocabSize() {
    const slider = document.getElementById('slider-vocab');
    const text = document.getElementById('training-text').value;
    const size = BPETokenizer.suggestVocabSize(text, 150, +slider.max);
    slider.value = size;
    document.getElementById('val-vocab').textContent = size;
  },

  // --- Tokenization ---

  tokenizeText() {
    const text = document.getElementById('training-text').value.trim();
    if (text.length < 20) {
      alert(t('js.minChars'));
      return;
    }

    // Re-initialize model to allow re-tokenization with new vocab size
    this.model = new MiniLM();
    const vocabSize = parseInt(document.getElementById('slider-vocab').value);
    const info = this.model.tokenize(text, vocabSize);
    // Remember the exact text the model learns from, to spot copied output later
    this.trainingText = text;
    this.copyIndex = null;
    this.showTokenInfo(info);
    this.showTokenizationViz(text);

    // Allow re-tokenizing, show Next
    document.getElementById('btn-tokenize').textContent = t('js.retokenize');
    document.getElementById('btn-next-2').style.display = 'inline-flex';
  },

  // --- Training ---

  startTraining() {
    if (!this.model.tokenizer) {
      alert(t('js.tokenizeFirst'));
      return;
    }

    if (!this.model.trained && this.model.step === 0) {
      this.model.embedDim = parseInt(document.getElementById('slider-embed').value);
      this.model.blockSize = parseInt(document.getElementById('slider-context').value);
      this.model.learningRate = parseFloat(document.getElementById('slider-lr').value);
      const result = this.model.buildTrainingData();
      if (!result) {
        alert(t('js.tooShort'));
        return;
      }
      this.showTrainingInfo(result);
      document.getElementById('slider-embed').disabled = true;
      document.getElementById('slider-context').disabled = true;
    }

    this.training = true;
    document.getElementById('btn-train').style.display = 'none';
    const btnPause = document.getElementById('btn-pause');
    btnPause.style.display = 'inline-flex';
    btnPause.innerHTML = `<span class="btn-icon">&#9632;</span> ${t('js.pause')}`;
    btnPause.classList.remove('btn-success');
    btnPause.classList.add('btn-danger');
    document.getElementById('btn-reset').style.display = 'inline-flex';
    document.getElementById('btn-generate-step').style.display = 'inline-flex';
    document.getElementById('training-viz').classList.add('active');

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
    const btnPause = document.getElementById('btn-pause');
    btnPause.innerHTML = `<span class="btn-icon">&#9654;</span> ${t('js.resume')}`;
    btnPause.classList.remove('btn-danger');
    btnPause.classList.add('btn-success');
  },

  resetModel() {
    this.lastResult = null;
    this.stopTraining();

    // Clear training viz
    document.getElementById('training-viz').classList.remove('active');
    document.getElementById('btn-train').innerHTML = `<span class="btn-icon">&#9654;</span> ${t('js.trainModel')}`;
    document.getElementById('btn-train').style.display = 'inline-flex';
    document.getElementById('btn-pause').style.display = 'none';
    document.getElementById('btn-reset').style.display = 'none';
    document.getElementById('btn-generate-step').style.display = 'none';
    ['loss-canvas', 'embedding-canvas'].forEach(id => {
      const c = document.getElementById(id);
      if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height);
    });
    document.getElementById('live-predictions').innerHTML = '';
    document.getElementById('step-counter').textContent = t('js.step', { n: 0 });
    document.getElementById('loss-value').textContent = t('js.lossEmpty');
    document.getElementById('generation-output').innerHTML = '';
    document.getElementById('generation-steps-viz').innerHTML = '';

    // Re-enable architecture sliders
    document.getElementById('slider-embed').disabled = false;
    document.getElementById('slider-context').disabled = false;

    // Re-init weights (buildTrainingData will be called again on next Train click)
    this.model.X = null;
    this.model.Y = null;
    this.model.step = 0;
    this.model.losses = [];
    this.model.trained = false;

    // Clear training info from model-info (keep token info)
    const tokenInfo = {
      vocabSize: this.model.vocabSize,
      numMerges: this.model.tokenizer ? this.model.tokenizer.merges.length : 0,
      compressionRatio: this.model.rawTokenIds ? (document.getElementById('training-text').value.length / this.model.rawTokenIds.length).toFixed(1) : '—',
    };
    this.showTokenInfo(tokenInfo);
    this.updateTrainingPreview();
  },

  showTokenInfo(info) {
    document.getElementById('model-info').innerHTML = `
      <div class="info-grid">
        <div class="info-item">
          <div class="info-label">${t('js.vocabulary')}</div>
          <div class="info-value">${t('js.vocabValue', { size: info.vocabSize, merges: info.numMerges })}</div>
        </div>
        <div class="info-item">
          <div class="info-label">${t('js.compression')}</div>
          <div class="info-value">${info.compressionRatio}x</div>
        </div>
      </div>
    `;
  },

  updateTrainingPreview() {
    if (!this.model || !this.model.rawTokenIds) return;
    const embedDim = parseInt(document.getElementById('slider-embed').value);
    const blockSize = parseInt(document.getElementById('slider-context').value);
    const vocabSize = this.model.vocabSize;
    const hiddenSize = this.model.hiddenSize;
    const flatSize = blockSize * embedDim;
    const numExamples = this.model.rawTokenIds.length + 1;
    const paramCount = vocabSize * embedDim + flatSize * hiddenSize + hiddenSize + hiddenSize * vocabSize + vocabSize;
    this.renderTrainingStats(numExamples, paramCount);
  },

  showTrainingInfo(info) {
    const paramCount = this.model.getParamCount();
    this.renderTrainingStats(info.numExamples, paramCount);
  },

  renderTrainingStats(numExamples, paramCount) {
    // Remove existing training stats if present
    const existing = document.getElementById('training-stats');
    if (existing) existing.remove();
    const el = document.getElementById('model-info');
    const div = document.createElement('div');
    div.id = 'training-stats';
    div.innerHTML = `
      <div class="info-grid" style="margin-top:8px;">
        <div class="info-item">
          <div class="info-label">${t('js.trainingExamples')}</div>
          <div class="info-value">${numExamples.toLocaleString(LANG === 'pt' ? 'pt-BR' : undefined)}</div>
        </div>
        <div class="info-item">
          <div class="info-label">${t('js.parameters')}</div>
          <div class="info-value">${paramCount.toLocaleString(LANG === 'pt' ? 'pt-BR' : undefined)}</div>
        </div>
      </div>
    `;
    el.appendChild(div);
  },

  showTokenizationViz(text) {
    const tok = this.model.tokenizer;
    if (!tok) return;

    document.getElementById('tokenization-viz').style.display = 'block';

    // Show top merge rules
    const mergesContainer = document.getElementById('token-viz-merges');
    const topMerges = tok.merges.slice(0, 20);
    let mergesHtml = '<div class="merge-rules-grid">';
    for (let i = 0; i < topMerges.length; i++) {
      const m = topMerges[i];
      const a = this.displayToken(tok.vocab[m.a]);
      const b = this.displayToken(tok.vocab[m.b]);
      const result = this.displayToken(tok.vocab[m.newId]);
      mergesHtml += `<div class="merge-rule">
        <span class="merge-num">#${i + 1}</span>
        <span class="merge-pair">"${this.escapeHtml(a)}" + "${this.escapeHtml(b)}"</span>
        <span class="merge-arrow-sm">→</span>
        <span class="merge-result">"${this.escapeHtml(result)}"</span>
      </div>`;
    }
    if (tok.merges.length > 20) {
      mergesHtml += `<div class="merge-rule" style="color:var(--text-muted)">${t('js.moreMerges', { n: tok.merges.length - 20 })}</div>`;
    }
    mergesHtml += '</div>';
    mergesContainer.innerHTML = mergesHtml;

    // Show tokenized text
    const textContainer = document.getElementById('token-viz-text');
    const tokenIds = tok.encode(text);
    const displayIds = tokenIds.slice(0, 300);

    let textHtml = '<div class="tokenized-text-display">';
    for (let i = 0; i < displayIds.length; i++) {
      const id = displayIds[i];
      const token = tok.vocab[id];
      const display = token
        .replace(/ /g, '␣')
        .replace(/\n/g, '↵')
        .replace(/\t/g, '⇥');
      const colorClass = token.length > 1 ? 'token-multi' : 'token-single';
      textHtml += `<span class="token-pill ${colorClass}" title="ID: ${id}, token: &quot;${this.escapeHtml(token)}&quot;">${this.escapeHtml(display)}</span>`;
    }
    if (tokenIds.length > 300) {
      textHtml += `<span class="token-pill" style="opacity:0.4">${t('js.moreTokens', { n: tokenIds.length - 300 })}</span>`;
    }
    textHtml += '</div>';
    textContainer.innerHTML = textHtml;
  },

  // --- Visualization ---

  updateTrainingViz(result) {
    this.lastResult = result;
    document.getElementById('step-counter').textContent = t('js.step', { n: result.step.toLocaleString(LANG === 'pt' ? 'pt-BR' : undefined) });
    document.getElementById('loss-value').textContent = t('js.loss', { loss: result.avgLoss.toFixed(4), lr: result.effectiveLR.toFixed(4) });

    this.drawLossChart();
    this.drawPredictions(result);

    if (result.step % 20 === 0) {
      this.drawEmbeddingViz();
    }
  },

  drawLossChart() {
    const canvas = document.getElementById('loss-canvas');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const parent = canvas.parentElement;
    const W = parent.clientWidth - 32;
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

    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH * i / 4);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(W - pad.right, y);
      ctx.stroke();
    }

    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '11px "IBM Plex Mono", monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const val = maxL - (maxL - minL) * i / 4;
      const y = pad.top + (plotH * i / 4);
      ctx.fillText(val.toFixed(2), pad.left - 8, y + 4);
    }
    ctx.textAlign = 'center';
    ctx.fillText(t('js.trainingSteps'), W / 2, H - 4);

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
    const ctxDisplay = result.contextTokens.map(t => this.displayToken(t)).join(' | ');
    const targetDisplay = this.displayToken(result.targetToken);
    let html = `<div class="pred-context">${t('js.context')}: <span class="mono">${this.escapeHtml(ctxDisplay)}</span> → ${t('js.target')}: <span class="mono">"${this.escapeHtml(targetDisplay)}"</span></div>`;
    html += '<div class="pred-bars">';
    for (const p of result.topPredictions.slice(0, 8)) {
      const pct = (p.prob * 100).toFixed(1);
      const isTarget = p.token === result.targetToken;
      const display = this.displayToken(p.token);
      html += `<div class="pred-bar-row ${isTarget ? 'pred-correct' : ''}">
        <span class="pred-char">${this.escapeHtml(display)}</span>
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

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of points) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;

    for (const p of points) {
      const x = pad + ((p.x - minX) / rangeX) * plotSize;
      const y = pad + ((p.y - minY) / rangeY) * plotSize;
      const label = this.displayToken(p.token).slice(0, 5);

      ctx.beginPath();
      ctx.fillStyle = this.tokenColor(p.token);
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = 'rgba(255,255,255,0.75)';
      ctx.font = '9px "IBM Plex Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(label, x, y - 6);
    }
  },

  tokenColor(token) {
    if (token.length === 1) {
      if (/\p{Ll}/u.test(token)) return '#4f8ff7';
      if (/\p{Lu}/u.test(token)) return '#7c5ff7';
      if (/[0-9]/.test(token)) return '#f7a84f';
      if (/\s/.test(token)) return '#4fc78f';
      return '#f75f8a';
    }
    // Multi-char tokens: color by length
    if (token.length <= 2) return '#6ba3ff';
    if (token.length <= 4) return '#a87cff';
    return '#ffb366';
  },

  displayToken(token) {
    if (!token) return '?';
    return token
      .replace(/ /g, '␣')
      .replace(/\n/g, '↵')
      .replace(/\t/g, '⇥');
  },

  // --- Generation ---

  generateText() {
    const seed = document.getElementById('prompt-input').value || '';
    const temp = parseFloat(document.getElementById('slider-temp').value);
    const topK = parseInt(document.getElementById('slider-topk').value);
    const topP = parseFloat(document.getElementById('slider-topp').value);
    const repPen = parseFloat(document.getElementById('slider-reppen').value);
    const length = parseInt(document.getElementById('slider-length').value);

    if (!this.model.C) {
      alert(t('js.trainFirst'));
      return;
    }

    const { steps, seed: usedSeed } = this.model.generate(seed, length, temp, topK, topP, repPen);

    const output = document.getElementById('generation-output');
    output.innerHTML = '';

    // Copy detection: index the training text once per tokenization
    if (!this.copyIndex && this.trainingText) {
      this.copyIndex = CopyCheck.buildIndex(this.trainingText);
    }
    let fullText = usedSeed;
    const tokenSpans = [];

    // Show seed
    const seedSpan = document.createElement('span');
    seedSpan.className = 'gen-seed';
    seedSpan.textContent = usedSeed;
    output.appendChild(seedSpan);

    // Animate token by token
    let tokenIdx = 0;

    const typeToken = () => {
      if (tokenIdx >= steps.length) {
        this.showGenerationSteps(steps);
        return;
      }
      const step = steps[tokenIdx];
      const span = document.createElement('span');
      span.className = 'gen-token';
      span.textContent = step.selected;
      span.style.opacity = 0.4 + step.selectedProb * 0.6;
      span.title = `"${step.selected}" (${t('js.confidence', { p: (step.selectedProb * 100).toFixed(1) })})`;
      output.appendChild(span);
      tokenSpans.push({ span, start: fullText.length, end: fullText.length + step.selected.length });
      fullText += step.selected;
      tokenIdx++;
      this.markCopied(fullText, usedSeed.length, tokenSpans, tokenIdx < steps.length);
      if (tokenIdx < steps.length) {
        setTimeout(typeToken, 40);
      } else {
        this.showGenerationSteps(steps);
      }
    };
    this.markCopied(fullText, usedSeed.length, tokenSpans, steps.length > 0);
    typeToken();
  },

  // Highlight generated tokens that belong to a run of 3+ words found in the
  // training text, and update the "% copied" meter. Runs after every token,
  // so it also works while the output is still being typed out.
  markCopied(fullText, genStart, tokenSpans, partial) {
    const meter = document.getElementById('copy-meter');
    if (!this.copyIndex) {
      meter.style.display = 'none';
      return;
    }
    const res = CopyCheck.analyze(this.copyIndex, fullText, genStart, partial);
    for (const ts of tokenSpans) {
      ts.span.classList.toggle('gen-copied', CopyCheck.overlaps(res.ranges, ts.start, ts.end));
    }
    meter.style.display = 'flex';
    document.getElementById('copy-meter-value').textContent = res.totalWords ? `${res.pct}%` : '—';
    document.getElementById('copy-meter-fill').style.width = `${res.pct}%`;
    this.lastCopy = res;
    this.showCopyDetail();
  },

  showCopyDetail() {
    if (!this.lastCopy) return;
    document.getElementById('copy-meter-detail').textContent =
      t('js.copyDetail', { copied: this.lastCopy.copiedWords, total: this.lastCopy.totalWords });
  },

  showGenerationSteps(steps) {
    const container = document.getElementById('generation-steps-viz');
    if (steps.length === 0) return;

    let html = `<h4>${t('js.generationSteps')}</h4>`;
    html += '<div class="gen-steps-scroll">';
    for (let i = 0; i < Math.min(steps.length, 50); i++) {
      const s = steps[i];
      const ctxDisplay = s.contextTokens.map(t => this.displayToken(t)).join('|');
      html += `<div class="gen-step" data-step="${i}">
        <div class="gen-step-num">#${i + 1}</div>
        <div class="gen-step-ctx">
          <span class="mono">${this.escapeHtml(ctxDisplay)}</span> →
          <span class="gen-step-selected">"${this.escapeHtml(this.displayToken(s.selected))}"</span>
          <span class="gen-step-conf">(${(s.selectedProb * 100).toFixed(1)}%)</span>
        </div>
        <div class="gen-step-probs">
          ${s.topPredictions.slice(0, 5).map(p => {
            const display = this.displayToken(p.token);
            return `<span class="gen-prob-pill ${p.token === s.selected ? 'selected' : ''}">${this.escapeHtml(display)} ${(p.prob * 100).toFixed(0)}%</span>`;
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
