// ============================================================
// Mini Language Model — Character-level Neural Network
// Architecture: Embedding → Concat → Linear → Tanh → Linear → Softmax
// This is a Bengio-style neural language model (2003),
// the foundation that modern LLMs like GPT build upon.
// ============================================================

class MiniLM {
  constructor(config = {}) {
    this.blockSize = config.blockSize || 6;
    this.embedDim = config.embedDim || 12;
    this.hiddenSize = config.hiddenSize || 100;
    this.learningRate = config.learningRate || 0.05;
    this.batchSize = config.batchSize || 64;

    this.vocab = null;
    this.charToIdx = {};
    this.idxToChar = {};
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

  buildVocab(text) {
    const chars = [...new Set(text)].sort();
    this.vocab = chars;
    this.vocabSize = chars.length;
    this.charToIdx = {};
    this.idxToChar = {};
    chars.forEach((ch, i) => {
      this.charToIdx[ch] = i;
      this.idxToChar[i] = ch;
    });
  }

  initParams() {
    const { vocabSize, embedDim, blockSize, hiddenSize } = this;
    const flatSize = blockSize * embedDim;

    this.C  = MiniLM.randn(vocabSize, embedDim, 0.2);
    this.W1 = MiniLM.randn(flatSize, hiddenSize, Math.sqrt(2.0 / flatSize));
    this.b1 = MiniLM.zeros1D(hiddenSize);
    this.W2 = MiniLM.randn(hiddenSize, vocabSize, 0.01);
    this.b2 = MiniLM.zeros1D(vocabSize);
  }

  prepareData(text) {
    this.buildVocab(text);
    this.initParams();

    const X = [];
    const Y = [];

    for (let i = 0; i <= text.length - this.blockSize - 1; i++) {
      const ctx = new Int32Array(this.blockSize);
      for (let j = 0; j < this.blockSize; j++) {
        ctx[j] = this.charToIdx[text[i + j]];
      }
      X.push(ctx);
      Y.push(this.charToIdx[text[i + this.blockSize]]);
    }

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

    return {
      vocabSize: this.vocabSize,
      vocab: this.vocab,
      numExamples: X.length,
    };
  }

  // --- Training step ---

  trainStep() {
    const { X, Y, blockSize, embedDim, hiddenSize, vocabSize, learningRate } = this;
    const flatSize = blockSize * embedDim;
    const B = Math.min(this.batchSize, X.length);

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

    // ---- SGD UPDATE ----
    for (let i = 0; i < vocabSize; i++)
      for (let j = 0; j < embedDim; j++)
        this.C[i][j] -= learningRate * dC[i][j];

    for (let i = 0; i < flatSize; i++)
      for (let j = 0; j < hiddenSize; j++)
        this.W1[i][j] -= learningRate * dW1[i][j];

    for (let j = 0; j < hiddenSize; j++)
      this.b1[j] -= learningRate * db1[j];

    for (let i = 0; i < hiddenSize; i++)
      for (let j = 0; j < vocabSize; j++)
        this.W2[i][j] -= learningRate * dW2[i][j];

    for (let j = 0; j < vocabSize; j++)
      this.b2[j] -= learningRate * db2[j];

    this.step++;
    this.losses.push(loss);

    // Return info for visualization
    const topProbs = [];
    for (let j = 0; j < vocabSize; j++) {
      topProbs.push({ char: this.idxToChar[j], prob: probs[0][j] });
    }
    topProbs.sort((a, b) => b.prob - a.prob);

    return {
      loss,
      step: this.step,
      avgLoss: this.losses.length > 50
        ? this.losses.slice(-50).reduce((a, b) => a + b, 0) / 50
        : this.losses.reduce((a, b) => a + b, 0) / this.losses.length,
      topPredictions: topProbs.slice(0, 10),
      targetChar: this.idxToChar[Y[idx[0]]],
      contextChars: Array.from(X[idx[0]]).map(i => this.idxToChar[i]).join(''),
    };
  }

  // --- Text generation ---

  generate(seed, length, temperature = 0.8) {
    if (!this.C) return { text: seed, steps: [] };

    const { blockSize, embedDim, hiddenSize, vocabSize } = this;
    const flatSize = blockSize * embedDim;

    // Build initial context from seed
    let context = new Int32Array(blockSize);
    const safeSeed = seed || this.idxToChar[0].repeat(blockSize);
    for (let i = 0; i < blockSize; i++) {
      const seedIdx = safeSeed.length - blockSize + i;
      if (seedIdx >= 0 && seedIdx < safeSeed.length && this.charToIdx[safeSeed[seedIdx]] !== undefined) {
        context[i] = this.charToIdx[safeSeed[seedIdx]];
      } else {
        context[i] = 0;
      }
    }

    let result = seed;
    const steps = [];

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
        logits[j] = sum / temperature;
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

      // Sample
      let r = Math.random();
      let nextIdx = vocabSize - 1;
      for (let j = 0; j < vocabSize; j++) {
        r -= probs[j];
        if (r <= 0) { nextIdx = j; break; }
      }

      const nextChar = this.idxToChar[nextIdx];
      result += nextChar;

      // Top predictions for visualization
      const topProbs = [];
      for (let j = 0; j < vocabSize; j++) {
        topProbs.push({ char: this.idxToChar[j], prob: probs[j] });
      }
      topProbs.sort((a, b) => b.prob - a.prob);

      steps.push({
        context: Array.from(context).map(i => this.idxToChar[i]).join(''),
        topPredictions: topProbs.slice(0, 8),
        selected: nextChar,
        selectedProb: probs[nextIdx],
      });

      // Slide window
      const newCtx = new Int32Array(blockSize);
      for (let i = 0; i < blockSize - 1; i++) newCtx[i] = context[i + 1];
      newCtx[blockSize - 1] = nextIdx;
      context = newCtx;
    }

    return { text: result, steps };
  }

  // Get 2D embedding projection (first 2 principal components via simple method)
  getEmbedding2D() {
    if (!this.C) return [];
    const points = [];
    // Use first two embedding dimensions (simple but effective for visualization)
    for (let i = 0; i < this.vocabSize; i++) {
      points.push({
        char: this.idxToChar[i],
        x: this.C[i][0],
        y: this.C[i][1],
      });
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
