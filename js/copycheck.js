// ============================================================
// Copy detection: which parts of a generated text were copied
// word for word from the training text?
//
// A word counts as "copied" when it belongs to a run of 3 or more
// consecutive words that also appears, in the same order, somewhere
// in the training text. Case and punctuation are ignored, so
// "The cat, sat" matches "the cat sat".
// ============================================================

const CopyCheck = {
  MIN_WORDS: 3,

  // Words with their character positions in the original string
  words(text) {
    const out = [];
    const re = /[\p{L}\p{N}]+/gu;
    let m;
    while ((m = re.exec(text)) !== null) {
      out.push({ w: m[0].toLowerCase(), start: m.index, end: m.index + m[0].length });
    }
    return out;
  },

  // Set of every run of `n` consecutive words in the training text
  buildIndex(trainingText, n = this.MIN_WORDS) {
    const ws = this.words(trainingText).map(x => x.w);
    const grams = new Set();
    for (let i = 0; i + n <= ws.length; i++) {
      grams.add(ws.slice(i, i + n).join(' '));
    }
    return { n, grams };
  },

  // Analyze `fullText` (seed + generated). Only words that start at or
  // after `genStart` count as generated; the seed still provides context,
  // so a run that starts in the seed and continues into the output counts.
  // The last word is skipped while `partial` is true (it may be cut off).
  analyze(index, fullText, genStart = 0, partial = false) {
    const ws = this.words(fullText);
    if (partial && ws.length && ws[ws.length - 1].end === fullText.length) ws.pop();
    const n = index.n;
    const copied = new Array(ws.length).fill(false);
    for (let i = 0; i + n <= ws.length; i++) {
      const key = ws.slice(i, i + n).map(x => x.w).join(' ');
      if (index.grams.has(key)) {
        for (let j = i; j < i + n; j++) copied[j] = true;
      }
    }

    // Merge consecutive copied words into character ranges (generated part only)
    const ranges = [];
    let total = 0, copiedCount = 0;
    for (let i = 0; i < ws.length; i++) {
      if (ws[i].start < genStart) continue;
      total++;
      if (!copied[i]) continue;
      copiedCount++;
      const last = ranges[ranges.length - 1];
      if (last && copied[i - 1] && ws[i - 1].start >= genStart) {
        last[1] = ws[i].end;
      } else {
        ranges.push([ws[i].start, ws[i].end]);
      }
    }

    return {
      ranges,
      totalWords: total,
      copiedWords: copiedCount,
      pct: total ? Math.round(100 * copiedCount / total) : 0,
    };
  },

  // Does the character range [a, b) contain a copied word character?
  overlaps(ranges, a, b) {
    for (const [s, e] of ranges) {
      if (s < b && a < e) return true;
    }
    return false;
  },
};

if (typeof module !== 'undefined') module.exports = CopyCheck;
