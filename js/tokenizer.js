// ============================================================
// BPE (Byte Pair Encoding) Tokenizer
// Learns sub-word tokens from training text by iteratively
// merging the most frequent adjacent pairs.
// ============================================================

class BPETokenizer {
  constructor() {
    this.vocab = {};        // id → string
    this.tokenToId = {};    // string → id
    this.merges = [];       // ordered merge rules: [{a, b, newId}, ...]
    this.vocabSize = 0;
    this.tokenFreqs = {};   // id → frequency count (for visualization)

    // Special token IDs (reserved at the start of the vocab)
    this.PAD_ID = 0;
    this.BOS_ID = 1;
    this.EOS_ID = 2;
    this.UNK_ID = 3;
    this.NUM_SPECIAL = 4;
  }

  // --- Train BPE on text, returns the final token ID sequence ---
  train(text, maxVocabSize = 150) {
    // Step 1: Reserve special tokens, then build base character vocabulary
    const chars = [...new Set(text)].sort();
    this.vocab = {};
    this.tokenToId = {};

    const specialTokens = ['<PAD>', '<BOS>', '<EOS>', '<UNK>'];
    specialTokens.forEach((tok, i) => {
      this.vocab[i] = tok;
      this.tokenToId[tok] = i;
    });

    chars.forEach((ch, i) => {
      const id = i + this.NUM_SPECIAL;
      this.vocab[id] = ch;
      this.tokenToId[ch] = id;
    });

    let nextId = chars.length + this.NUM_SPECIAL;

    // Step 2: Convert text to character-level token IDs
    let tokens = [];
    for (let i = 0; i < text.length; i++) {
      tokens.push(this.tokenToId[text[i]]);
    }

    // Step 3: Iterative merging
    this.merges = [];

    while (nextId < maxVocabSize) {
      // Count all adjacent pairs
      const pairCounts = {};
      for (let i = 0; i < tokens.length - 1; i++) {
        const key = tokens[i] + ',' + tokens[i + 1];
        pairCounts[key] = (pairCounts[key] || 0) + 1;
      }

      // Find most frequent pair
      let bestPair = null;
      let bestCount = 1; // minimum count of 2 to merge
      for (const key in pairCounts) {
        if (pairCounts[key] > bestCount) {
          bestCount = pairCounts[key];
          bestPair = key;
        }
      }

      if (!bestPair) break; // No pair occurs more than once

      const [aStr, bStr] = bestPair.split(',');
      const a = parseInt(aStr);
      const b = parseInt(bStr);

      // Create new merged token
      const newTokenStr = this.vocab[a] + this.vocab[b];
      this.vocab[nextId] = newTokenStr;
      this.tokenToId[newTokenStr] = nextId;

      this.merges.push({ a, b, newId: nextId });

      // Replace all occurrences of (a, b) with newId in token sequence
      tokens = this._replacePair(tokens, a, b, nextId);

      nextId++;
    }

    this.vocabSize = nextId;

    // Count token frequencies for visualization
    this.tokenFreqs = {};
    for (const id of tokens) {
      this.tokenFreqs[id] = (this.tokenFreqs[id] || 0) + 1;
    }

    return tokens;
  }

  // Replace all adjacent (a, b) pairs with newId in token list
  _replacePair(tokens, a, b, newId) {
    const result = [];
    let i = 0;
    while (i < tokens.length) {
      if (i < tokens.length - 1 && tokens[i] === a && tokens[i + 1] === b) {
        result.push(newId);
        i += 2;
      } else {
        result.push(tokens[i]);
        i += 1;
      }
    }
    return result;
  }

  // --- Encode new text using learned merges ---
  encode(text) {
    // Convert to character-level IDs
    let tokens = [];
    for (let i = 0; i < text.length; i++) {
      const id = this.tokenToId[text[i]];
      tokens.push(id !== undefined ? id : this.UNK_ID);
    }

    // Replay merges in order
    for (const merge of this.merges) {
      tokens = this._replacePair(tokens, merge.a, merge.b, merge.newId);
    }

    return tokens;
  }

  // --- Decode token IDs back to text ---
  decode(ids) {
    let result = '';
    for (const id of ids) {
      if (id < this.NUM_SPECIAL) continue;
      if (this.vocab[id] !== undefined) {
        result += this.vocab[id];
      }
    }
    return result;
  }

  // --- Get vocab entries sorted by frequency (for visualization) ---
  getTopTokens(n = 60) {
    const entries = [];
    for (let id = this.NUM_SPECIAL; id < this.vocabSize; id++) {
      entries.push({
        id,
        token: this.vocab[id],
        freq: this.tokenFreqs[id] || 0,
        len: this.vocab[id].length,
      });
    }
    entries.sort((a, b) => b.freq - a.freq);
    return entries.slice(0, n);
  }
}
