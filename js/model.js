// ============================================================
// Mini Language Model — Sub-word Neural Network
// Architecture: Embedding → Concat → Linear → Tanh → Linear → Softmax
// Uses BPE tokenization for sub-word prediction.
// This is a Bengio-style neural language model (2003),
// the foundation that modern LLMs like GPT build upon.
// ============================================================

class MiniLM {
  constructor(config = {}) {
    this.blockSize = config.blockSize || 5;
    this.embedDim = config.embedDim || 8;
    this.hiddenSize = config.hiddenSize || 128;
    this.learningRate = config.learningRate || 0.05;
    this.batchSize = config.batchSize || 32;
    this.maxVocabSize = config.maxVocabSize || 150;

    this.tokenizer = null;
    this.vocabSize = 0;

    // Parameters
    this.C = null;   // Embedding table  [vocabSize, embedDim]
    this.W1 = null;  // Hidden weights    [flatSize, hiddenSize]
    this.b1 = null;  // Hidden biases     [hiddenSize]
    this.W2 = null;  // Output weights    [hiddenSize, vocabSize]
    this.b2 = null;  // Output biases     [vocabSize]

    // Training data
    this.X = null;
    this.Y = null;

    // State
    this.step = 0;
    this.losses = [];
    this.trained = false;
  }

  // --- Matrix utilities ---

  static randn(rows, cols, scale = 0.1) {
    const m = new Array(rows);
    for (let i = 0; i < rows; i++) {
      m[i] = new Float64Array(cols);
      for (let j = 0; j < cols; j++) {
        const u1 = Math.random() || 1e-10;
        const u2 = Math.random();
        m[i][j] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * scale;
      }
    }
    return m;
  }

  static zeros1D(n) {
    return new Float64Array(n);
  }

  static zeros2D(rows, cols) {
    const m = new Array(rows);
    for (let i = 0; i < rows; i++) m[i] = new Float64Array(cols);
    return m;
  }

  // --- Data preparation ---

  initParams() {
    const { vocabSize, embedDim, blockSize, hiddenSize } = this;
    const flatSize = blockSize * embedDim;

    this.C  = MiniLM.randn(vocabSize, embedDim, 0.2);
    this.W1 = MiniLM.randn(flatSize, hiddenSize, Math.sqrt(2.0 / flatSize));
    this.b1 = MiniLM.zeros1D(hiddenSize);
    this.W2 = MiniLM.randn(hiddenSize, vocabSize, 0.01);
    this.b2 = MiniLM.zeros1D(vocabSize);
  }

  // Step 2: Train BPE tokenizer and encode text (no weights or X/Y yet)
  tokenize(text, maxVocabSize) {
    this.tokenizer = new BPETokenizer();
    this.rawTokenIds = this.tokenizer.train(text, maxVocabSize || this.maxVocabSize);
    this.vocabSize = this.tokenizer.vocabSize;

    return {
      vocabSize: this.vocabSize,
      numMerges: this.tokenizer.merges.length,
      compressionRatio: (text.length / this.rawTokenIds.length).toFixed(1),
    };
  }

  // Step 3: Build X/Y training pairs and init weights (uses current blockSize/embedDim)
  buildTrainingData() {
    if (!this.rawTokenIds) return false;

    // Pad with blockSize BOS tokens so the model sees every BOS-padding pattern
    const tokenIds = [
      ...new Array(this.blockSize).fill(this.tokenizer.BOS_ID),
      ...this.rawTokenIds,
      this.tokenizer.EOS_ID,
    ];

    // Init neural network parameters with current embedDim
    this.initParams();

    // Create training examples from token sequence
    const X = [];
    const Y = [];

    for (let i = 0; i <= tokenIds.length - this.blockSize - 1; i++) {
      const ctx = new Int32Array(this.blockSize);
      for (let j = 0; j < this.blockSize; j++) {
        ctx[j] = tokenIds[i + j];
      }
      X.push(ctx);
      Y.push(tokenIds[i + this.blockSize]);
    }

    // Guard against zero training examples
    if (X.length === 0) return false;

    // Shuffle
    for (let i = X.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [X[i], X[j]] = [X[j], X[i]];
      [Y[i], Y[j]] = [Y[j], Y[i]];
    }

    this.X = X;
    this.Y = Y;
    this.step = 0;
    this.losses = [];
    this.trained = false;

    return { numExamples: X.length };
  }

  // --- Training step ---

  trainStep() {
    const { X, Y, blockSize, embedDim, hiddenSize, vocabSize } = this;
    const flatSize = blockSize * embedDim;
    const B = Math.min(this.batchSize, X.length);

    // Learning rate decay: halve every 2000 steps, floor at 0.001
    const decays = Math.floor(this.step / 3000);
    const effectiveLR = Math.max(this.learningRate * Math.pow(0.5, decays), 0.001);

    // Sample mini-batch
    const idx = new Int32Array(B);
    for (let i = 0; i < B; i++) {
      idx[i] = Math.floor(Math.random() * X.length);
    }

    // ---- FORWARD PASS ----

    // 1. Embedding lookup → flatten
    const flat = MiniLM.zeros2D(B, flatSize);
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < blockSize; t++) {
        const ci = X[idx[b]][t];
        const offset = t * embedDim;
        for (let e = 0; e < embedDim; e++) {
          flat[b][offset + e] = this.C[ci][e];
        }
      }
    }

    // 2. Hidden layer: h = tanh(flat @ W1 + b1)
    const hPre = MiniLM.zeros2D(B, hiddenSize);
    const h = MiniLM.zeros2D(B, hiddenSize);
    for (let b = 0; b < B; b++) {
      for (let j = 0; j < hiddenSize; j++) {
        let sum = this.b1[j];
        for (let k = 0; k < flatSize; k++) {
          sum += flat[b][k] * this.W1[k][j];
        }
        hPre[b][j] = sum;
        h[b][j] = Math.tanh(sum);
      }
    }

    // 3. Output: logits = h @ W2 + b2
    const logits = MiniLM.zeros2D(B, vocabSize);
    for (let b = 0; b < B; b++) {
      for (let j = 0; j < vocabSize; j++) {
        let sum = this.b2[j];
        for (let k = 0; k < hiddenSize; k++) {
          sum += h[b][k] * this.W2[k][j];
        }
        logits[b][j] = sum;
      }
    }

    // 4. Softmax + cross-entropy loss
    const probs = MiniLM.zeros2D(B, vocabSize);
    let loss = 0;
    for (let b = 0; b < B; b++) {
      let maxL = -Infinity;
      for (let j = 0; j < vocabSize; j++) {
        if (logits[b][j] > maxL) maxL = logits[b][j];
      }
      let sumExp = 0;
      for (let j = 0; j < vocabSize; j++) {
        probs[b][j] = Math.exp(logits[b][j] - maxL);
        sumExp += probs[b][j];
      }
      for (let j = 0; j < vocabSize; j++) {
        probs[b][j] /= sumExp;
      }
      loss -= Math.log(probs[b][Y[idx[b]]] + 1e-10);
    }
    loss /= B;

    // ---- BACKWARD PASS ----

    // dlogits = (probs - one_hot(targets)) / B
    const dlogits = MiniLM.zeros2D(B, vocabSize);
    for (let b = 0; b < B; b++) {
      for (let j = 0; j < vocabSize; j++) {
        dlogits[b][j] = probs[b][j] / B;
      }
      dlogits[b][Y[idx[b]]] -= 1.0 / B;
    }

    // dW2 = h^T @ dlogits, db2 = colsum(dlogits)
    const dW2 = MiniLM.zeros2D(hiddenSize, vocabSize);
    const db2 = MiniLM.zeros1D(vocabSize);
    for (let b = 0; b < B; b++) {
      for (let j = 0; j < vocabSize; j++) {
        db2[j] += dlogits[b][j];
        for (let i = 0; i < hiddenSize; i++) {
          dW2[i][j] += h[b][i] * dlogits[b][j];
        }
      }
    }

    // dh = dlogits @ W2^T
    const dh = MiniLM.zeros2D(B, hiddenSize);
    for (let b = 0; b < B; b++) {
      for (let i = 0; i < hiddenSize; i++) {
        let sum = 0;
        for (let j = 0; j < vocabSize; j++) {
          sum += dlogits[b][j] * this.W2[i][j];
        }
        dh[b][i] = sum;
      }
    }

    // dhPre = dh * (1 - h^2)   [tanh derivative]
    const dhPre = MiniLM.zeros2D(B, hiddenSize);
    for (let b = 0; b < B; b++) {
      for (let i = 0; i < hiddenSize; i++) {
        dhPre[b][i] = dh[b][i] * (1 - h[b][i] * h[b][i]);
      }
    }

    // dW1 = flat^T @ dhPre, db1 = colsum(dhPre)
    const dW1 = MiniLM.zeros2D(flatSize, hiddenSize);
    const db1 = MiniLM.zeros1D(hiddenSize);
    for (let b = 0; b < B; b++) {
      for (let j = 0; j < hiddenSize; j++) {
        db1[j] += dhPre[b][j];
        for (let i = 0; i < flatSize; i++) {
          dW1[i][j] += flat[b][i] * dhPre[b][j];
        }
      }
    }

    // dflat = dhPre @ W1^T
    const dflat = MiniLM.zeros2D(B, flatSize);
    for (let b = 0; b < B; b++) {
      for (let i = 0; i < flatSize; i++) {
        let sum = 0;
        for (let j = 0; j < hiddenSize; j++) {
          sum += dhPre[b][j] * this.W1[i][j];
        }
        dflat[b][i] = sum;
      }
    }

    // dC: accumulate embedding gradients
    const dC = MiniLM.zeros2D(vocabSize, embedDim);
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < blockSize; t++) {
        const ci = X[idx[b]][t];
        const offset = t * embedDim;
        for (let e = 0; e < embedDim; e++) {
          dC[ci][e] += dflat[b][offset + e];
        }
      }
    }

    // ---- GRADIENT CLIPPING ----
    const maxNorm = 5.0;
    let gradNorm = 0;
    for (let i = 0; i < vocabSize; i++)
      for (let j = 0; j < embedDim; j++)
        gradNorm += dC[i][j] * dC[i][j];
    for (let i = 0; i < flatSize; i++)
      for (let j = 0; j < hiddenSize; j++)
        gradNorm += dW1[i][j] * dW1[i][j];
    for (let j = 0; j < hiddenSize; j++)
      gradNorm += db1[j] * db1[j];
    for (let i = 0; i < hiddenSize; i++)
      for (let j = 0; j < vocabSize; j++)
        gradNorm += dW2[i][j] * dW2[i][j];
    for (let j = 0; j < vocabSize; j++)
      gradNorm += db2[j] * db2[j];
    gradNorm = Math.sqrt(gradNorm);

    if (gradNorm > maxNorm) {
      const scale = maxNorm / gradNorm;
      for (let i = 0; i < vocabSize; i++)
        for (let j = 0; j < embedDim; j++)
          dC[i][j] *= scale;
      for (let i = 0; i < flatSize; i++)
        for (let j = 0; j < hiddenSize; j++)
          dW1[i][j] *= scale;
      for (let j = 0; j < hiddenSize; j++)
        db1[j] *= scale;
      for (let i = 0; i < hiddenSize; i++)
        for (let j = 0; j < vocabSize; j++)
          dW2[i][j] *= scale;
      for (let j = 0; j < vocabSize; j++)
        db2[j] *= scale;
    }

    // ---- SGD UPDATE ----
    for (let i = 0; i < vocabSize; i++)
      for (let j = 0; j < embedDim; j++)
        this.C[i][j] -= effectiveLR * dC[i][j];

    for (let i = 0; i < flatSize; i++)
      for (let j = 0; j < hiddenSize; j++)
        this.W1[i][j] -= effectiveLR * dW1[i][j];

    for (let j = 0; j < hiddenSize; j++)
      this.b1[j] -= effectiveLR * db1[j];

    for (let i = 0; i < hiddenSize; i++)
      for (let j = 0; j < vocabSize; j++)
        this.W2[i][j] -= effectiveLR * dW2[i][j];

    for (let j = 0; j < vocabSize; j++)
      this.b2[j] -= effectiveLR * db2[j];

    this.step++;
    this.losses.push(loss);

    // Return info for visualization
    const topProbs = [];
    for (let j = this.tokenizer.NUM_SPECIAL; j < vocabSize; j++) {
      topProbs.push({ token: this.tokenizer.vocab[j], prob: probs[0][j] });
    }
    topProbs.sort((a, b) => b.prob - a.prob);

    return {
      loss,
      step: this.step,
      effectiveLR,
      avgLoss: this.losses.length > 50
        ? this.losses.slice(-50).reduce((a, b) => a + b, 0) / 50
        : this.losses.reduce((a, b) => a + b, 0) / this.losses.length,
      topPredictions: topProbs.slice(0, 10),
      targetToken: this.tokenizer.vocab[Y[idx[0]]],
      contextTokens: Array.from(X[idx[0]]).map(i => this.tokenizer.vocab[i]),
    };
  }

  // --- Text generation ---

  generate(seed, length, temperature = 0.8, topK = 0, topP = 0, repetitionPenalty = 1.0) {
    if (!this.C || !this.tokenizer) return { text: seed, steps: [], seed };

    const { blockSize, embedDim, hiddenSize, vocabSize } = this;
    const flatSize = blockSize * embedDim;

    // Words carry their leading space (" cat"), so a trailing space in the
    // seed would be a lone token the model never saw in training: drop it
    seed = seed.replace(/ +$/, '');

    // Tokenize the seed
    let seedTokens = seed ? this.tokenizer.encode(seed) : [];

    // Build context: pad or trim to blockSize
    let context = new Int32Array(blockSize);
    if (seedTokens.length >= blockSize) {
      for (let i = 0; i < blockSize; i++) {
        context[i] = seedTokens[seedTokens.length - blockSize + i];
      }
    } else {
      const pad = blockSize - seedTokens.length;
      for (let i = 0; i < pad; i++) context[i] = this.tokenizer.BOS_ID;
      for (let i = 0; i < seedTokens.length; i++) {
        context[pad + i] = seedTokens[i];
      }
    }

    let result = seed;
    const steps = [];
    const seenTokens = new Set();

    for (let n = 0; n < length; n++) {
      // Forward pass (single example)
      const flat = new Float64Array(flatSize);
      for (let t = 0; t < blockSize; t++) {
        const offset = t * embedDim;
        for (let e = 0; e < embedDim; e++) {
          flat[offset + e] = this.C[context[t]][e];
        }
      }

      const h = new Float64Array(hiddenSize);
      for (let j = 0; j < hiddenSize; j++) {
        let sum = this.b1[j];
        for (let k = 0; k < flatSize; k++) {
          sum += flat[k] * this.W1[k][j];
        }
        h[j] = Math.tanh(sum);
      }

      const logits = new Float64Array(vocabSize);
      for (let j = 0; j < vocabSize; j++) {
        let sum = this.b2[j];
        for (let k = 0; k < hiddenSize; k++) {
          sum += h[k] * this.W2[k][j];
        }
        logits[j] = sum;
      }

      // Suppress special tokens (allow EOS for early stopping)
      logits[this.tokenizer.PAD_ID] = -Infinity;
      logits[this.tokenizer.BOS_ID] = -Infinity;
      logits[this.tokenizer.UNK_ID] = -Infinity;

      // Repetition penalty (before temperature scaling)
      if (repetitionPenalty !== 1.0) {
        for (const prevIdx of seenTokens) {
          if (logits[prevIdx] > 0) {
            logits[prevIdx] /= repetitionPenalty;
          } else {
            logits[prevIdx] *= repetitionPenalty;
          }
        }
      }

      // Temperature scaling
      for (let j = 0; j < vocabSize; j++) {
        logits[j] /= temperature;
      }

      // Top-K filtering: set logits outside top K to -Infinity
      if (topK > 0 && topK < vocabSize) {
        const indexed = [];
        for (let j = 0; j < vocabSize; j++) indexed.push({ j, v: logits[j] });
        indexed.sort((a, b) => b.v - a.v);
        const cutoff = indexed[topK - 1].v;
        for (let j = 0; j < vocabSize; j++) {
          if (logits[j] < cutoff) logits[j] = -Infinity;
        }
      }

      // Softmax
      let maxL = -Infinity;
      for (let j = 0; j < vocabSize; j++) if (logits[j] > maxL) maxL = logits[j];
      const probs = new Float64Array(vocabSize);
      let sumExp = 0;
      for (let j = 0; j < vocabSize; j++) {
        probs[j] = Math.exp(logits[j] - maxL);
        sumExp += probs[j];
      }
      for (let j = 0; j < vocabSize; j++) probs[j] /= sumExp;

      // Top-P (nucleus) filtering: zero out tokens beyond cumulative threshold
      if (topP > 0 && topP < 1.0) {
        const indexed = [];
        for (let j = 0; j < vocabSize; j++) indexed.push({ j, p: probs[j] });
        indexed.sort((a, b) => b.p - a.p);
        let cumulative = 0;
        let cutoffReached = false;
        for (const item of indexed) {
          if (cutoffReached) {
            probs[item.j] = 0;
          } else {
            cumulative += item.p;
            if (cumulative > topP) cutoffReached = true;
          }
        }
        // Re-normalize
        let pSum = 0;
        for (let j = 0; j < vocabSize; j++) pSum += probs[j];
        for (let j = 0; j < vocabSize; j++) probs[j] /= pSum;
      }

      // Sample
      let r = Math.random();
      let nextIdx = vocabSize - 1;
      for (let j = 0; j < vocabSize; j++) {
        r -= probs[j];
        if (r <= 0) { nextIdx = j; break; }
      }

      // Stop on EOS
      if (nextIdx === this.tokenizer.EOS_ID) break;

      const nextToken = this.tokenizer.vocab[nextIdx];
      result += nextToken;
      seenTokens.add(nextIdx);

      // Top predictions for visualization
      const topProbs = [];
      for (let j = this.tokenizer.NUM_SPECIAL; j < vocabSize; j++) {
        topProbs.push({ token: this.tokenizer.vocab[j], prob: probs[j] });
      }
      topProbs.sort((a, b) => b.prob - a.prob);

      steps.push({
        contextTokens: Array.from(context).map(i => this.tokenizer.vocab[i]),
        topPredictions: topProbs.slice(0, 8),
        selected: nextToken,
        selectedProb: probs[nextIdx],
      });

      // Slide window
      const newCtx = new Int32Array(blockSize);
      for (let i = 0; i < blockSize - 1; i++) newCtx[i] = context[i + 1];
      newCtx[blockSize - 1] = nextIdx;
      context = newCtx;
    }

    return { text: result, steps, seed };
  }

  // Get 2D embedding projection for top tokens
  getEmbedding2D() {
    if (!this.C || !this.tokenizer) return [];
    const topTokens = this.tokenizer.getTopTokens(60);
    const points = [];
    for (const t of topTokens) {
      if (t.id < this.vocabSize) {
        points.push({
          token: t.token,
          freq: t.freq,
          x: this.C[t.id][0],
          y: this.C[t.id][1],
        });
      }
    }
    return points;
  }

  getParamCount() {
    if (!this.C) return 0;
    const flatSize = this.blockSize * this.embedDim;
    return (
      this.vocabSize * this.embedDim +       // C
      flatSize * this.hiddenSize +            // W1
      this.hiddenSize +                       // b1
      this.hiddenSize * this.vocabSize +      // W2
      this.vocabSize                          // b2
    );
  }
}
