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

    // Tokenize button
    document.getElementById('btn-tokenize').addEventListener('click', () => this.tokenizeText());

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

    // Show BPE merge process on example text
    const text = "the cat sat on the mat";
    const charTokens = text.split('');

    let html = '<div class="demo-label">Input text:</div>';
    html += `<div class="demo-text-display">"${text}"</div>`;
    html += '<div class="demo-arrow">Step 1: Split into characters</div>';
    html += '<div class="token-grid">';
    for (const ch of charTokens) {
      const display = ch === ' ' ? '␣' : ch;
      html += `<div class="token-cell"><span class="token-char">${display}</span></div>`;
    }
    html += '</div>';

    // Show merge steps
    const mergeSteps = [
      { pair: ['a', 't'], result: 'at', tokens: ['t', 'h', 'e', '␣', 'c', 'at', '␣', 's', 'at', '␣', 'o', 'n', '␣', 't', 'h', 'e', '␣', 'm', 'at'] },
      { pair: ['t', 'h'], result: 'th', tokens: ['th', 'e', '␣', 'c', 'at', '␣', 's', 'at', '␣', 'o', 'n', '␣', 'th', 'e', '␣', 'm', 'at'] },
      { pair: ['th', 'e'], result: 'the', tokens: ['the', '␣', 'c', 'at', '␣', 's', 'at', '␣', 'o', 'n', '␣', 'the', '␣', 'm', 'at'] },
    ];

    for (let s = 0; s < mergeSteps.length; s++) {
      const step = mergeSteps[s];
      html += `<div class="demo-arrow">Merge "${step.pair[0]}" + "${step.pair[1]}" → "${step.result}" (most frequent pair)</div>`;
      html += '<div class="token-grid">';
      for (const tok of step.tokens) {
        const display = tok === '␣' ? '␣' : tok;
        const isMerged = tok === step.result;
        html += `<div class="token-cell ${isMerged ? 'token-merged' : ''}"><span class="token-char">${display}</span></div>`;
      }
      html += '</div>';
    }

    html += '<div class="merge-summary">22 characters → 15 tokens (1.5x compression)</div>';
    container.innerHTML = html;
  },

  renderPredictionDemo() {
    const container = document.getElementById('prediction-demo');
    if (!container) return;

    const contextTokens = ['the', ' ', 'ca'];
    const predictions = [
      { token: 't', prob: 0.42 },
      { token: 'ts', prob: 0.18 },
      { token: 'r', prob: 0.15 },
      { token: 'n', prob: 0.10 },
      { token: 'me', prob: 0.06 },
      { token: 'b', prob: 0.05 },
      { token: 'se', prob: 0.04 },
    ];

    let html = '<div class="demo-label">Context tokens:</div>';
    html += '<div class="context-display">';
    for (const tok of contextTokens) {
      const display = tok === ' ' ? '␣' : tok;
      html += `<span class="ctx-char">${display}</span>`;
    }
    html += '<span class="ctx-char ctx-next">?</span>';
    html += '</div>';
    html += '<div class="demo-arrow">Neural Network Prediction</div>';
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

  // --- Lab functionality ---

  updateTextStats() {
    const text = document.getElementById('training-text').value;
    const chars = text.length;
    const unique = new Set(text).size;
    const words = text.split(/\s+/).filter(w => w.length > 0).length;
    document.getElementById('text-stats').innerHTML =
      `${chars.toLocaleString()} characters &middot; ${words.toLocaleString()} words &middot; ${unique} unique chars`;
  },

  loadSample(name) {
    const samples = {
      1: `hello world.
hello there.
hello friend.
hi there.
good morning.
good evening.
good night.

the sun is warm.
the moon is pale.
the sky is blue.
the grass is green.
the rain is cold.
the wind is soft.
the fire is bright.

a cat sits on the mat.
a dog runs in the yard.
a bird sings in the tree.
a fish swims in the pond.
a child reads a book.
a baker makes bread.
a farmer grows rice.

the cat is small.
the dog is happy.
the bird is quick.
the fish is quiet.
the child is kind.
the baker is busy.
the farmer is strong.

i see the sun.
i see the moon.
i see the tree.
i see the river.
i see the road.
i see the house.
i see the light.

you see the stars.
you see the clouds.
you see the field.
you see the garden.
you see the gate.
you see the hill.
you see the town.

we walk to the river.
we walk to the market.
we walk to the house.
we walk to the garden.
we walk in the morning.
we walk in the evening.
we walk together.

they sit by the fire.
they sit by the window.
they sit near the door.
they sit under the tree.
they talk and laugh.
they sing and smile.
they rest and wait.

the red apple is sweet.
the green apple is sour.
the yellow lemon is sharp.
the brown bread is warm.
the white milk is fresh.
the black tea is hot.
the clear water is cool.

one little stone.
two little stones.
three little stones.
four little stones.
five little stones.

one small bird.
two small birds.
three small birds.
four small birds.
five small birds.

monday comes after sunday.
tuesday comes after monday.
wednesday comes after tuesday.
thursday comes after wednesday.
friday comes after thursday.
saturday comes after friday.
sunday comes after saturday.

spring brings flowers.
summer brings heat.
autumn brings wind.
winter brings frost.

the boy said, "hello."
the girl said, "how are you?"
the boy said, "i am fine."
the girl said, "let us walk."
they went down the road.
they saw a little shop.
they bought bread and tea.
then they went home.

in the morning, the town is quiet.
at noon, the town is bright.
in the evening, the town is calm.
at night, the town is dark.

a small lamp on a wooden desk.
a closed book near the lamp.
a thin pen beside the book.
a cup of tea by the pen.
a quiet room in the house.

this is a simple story.
this is a calm story.
this is a short story.
this is a clear story.
the story has a start.
the story has a middle.
the story has an end.

start:
the child wakes up early.
the child opens the door.
the child walks outside.

middle:
the child sees the sky.
the child hears a bird.
the child finds a flower.

end:
the child smiles softly.
the child walks back home.
the child sleeps at night.

question: where is the cat?
answer: the cat is on the mat.

question: where is the dog?
answer: the dog is in the yard.

question: what is in the cup?
answer: tea is in the cup.

question: who reads the book?
answer: the child reads the book.

yes.
no.
maybe.
please.
thank you.
you are welcome.

the river is long and slow.
the road is long and straight.
the hill is high and green.
the lake is still and clear.
the forest is deep and cool.

some words are short.
some words are long.
some lines are short.
some lines are long.
every line has meaning.

the model reads text.
the model learns patterns.
the model predicts words.
the model predicts letters.
the model improves with practice.

hello world.
good morning.
the sun is warm.
the cat is small.
the child reads a book.
we walk to the river.
they sit by the fire.
the story has an end.`,
      2: `the day begins.
the town wakes slowly.
the street is quiet.
the air is cool.
the light is soft.

a man opens a shop.
a woman sweeps the floor.
a child ties a shoe.
a dog waits by the gate.
a cat sleeps on a chair.

the baker lights the oven.
the teacher opens a book.
the farmer walks to the field.
the driver starts the bus.
the cook boils water.

bread is warm.
milk is fresh.
tea is hot.
water is clear.
fruit is sweet.

the apple is red.
the pear is green.
the plum is dark.
the peach is soft.
the lemon is bright.

one cup on the table.
two cups on the table.
three cups on the table.
four cups on the table.
five cups on the table.

one plate on the shelf.
two plates on the shelf.
three plates on the shelf.
four plates on the shelf.
five plates on the shelf.

the room is small.
the room is neat.
the room is clean.
the room is bright.
the room is still.

a window faces the road.
a lamp stands by the bed.
a coat hangs by the door.
a rug lies on the floor.
a clock ticks on the wall.

i open the window.
i close the gate.
i carry a basket.
i wash my hands.
i drink some water.
i sit in the chair.
i read a page.

you open the book.
you close the box.
you carry a bag.
you wash a cup.
you drink hot tea.
you sit by the fire.
you read a story.

we go to the market.
we go to the river.
we go to the station.
we go to the garden.
we go in the morning.
we go in the rain.
we go together.

they come from the hill.
they come from the town.
they come from the farm.
they come in silence.
they come with smiles.
they come with bread.
they come to help.

the market is busy.
the market is loud.
the market is full of color.
the market is full of people.
the market is full of sound.

a woman sells apples.
a man sells bread.
a child sells flowers.
a boy carries water.
a girl counts coins.

"good morning," said the baker.
"good morning," said the farmer.
"how is the day?" asked the teacher.
"the day is fine," said the cook.
"the bread is ready," said the baker.

the bell rings once.
the bell rings twice.
the bell rings three times.
people stop and listen.
then people move again.

at noon, the sun is high.
at noon, the road is bright.
at noon, the shops are full.
at noon, the square is warm.
at noon, the town is awake.

after noon, the wind rises.
after noon, the clouds gather.
after noon, the leaves shake.
after noon, the birds hide.
after noon, the rain begins.

rain falls on the roof.
rain falls on the road.
rain falls on the trees.
rain falls on the fields.
rain falls on the river.

the dog runs inside.
the cat hides under the chair.
the child shuts the window.
the mother folds the cloth.
the father adds wood to the fire.

inside the house, the room is calm.
inside the house, the lamp is warm.
inside the house, the soup smells good.
inside the house, the chairs are dry.
inside the house, the voices are low.

the family eats together.
the family talks softly.
the family shares bread.
the family drinks tea.
the family laughs a little.

after the rain, the sky clears.
after the rain, the grass shines.
after the rain, the road is dark.
after the rain, the air is fresh.
after the rain, the birds return.

evening comes slowly.
the shops begin to close.
the market grows quiet.
the lamps begin to glow.
the town becomes calm.

the teacher walks home.
the farmer washes his boots.
the baker counts the coins.
the cook covers the pot.
the child watches the stars.

one star in the sky.
two stars in the sky.
three stars in the sky.
four stars in the sky.
five stars in the sky.

night is cool.
night is dark.
night is gentle.
night is quiet.
night is deep.

question: where is the cat?
answer: the cat is under the chair.

question: where is the baker?
answer: the baker is in the shop.

question: what falls on the roof?
answer: rain falls on the roof.

question: who reads a story?
answer: you read a story.

this is a simple town.
this is a quiet town.
this is a working town.
this is a kind town.
this is a small town with many voices.

the day begins.
the town wakes slowly.
the street is quiet.
the air is cool.
the light is soft.`,
      3: `there is a house by the lake.
there is a path behind the house.
there is a fence by the path.
there is a tree by the fence.
there is a bench under the tree.

the lake is wide and still.
the water is blue in the day.
the water is silver at night.
the water moves in the wind.
the water reflects the moon.

a boy lives in the house.
a girl lives in the house.
their mother lives in the house.
their father lives in the house.
an old dog lives there too.

every morning, the boy wakes early.
every morning, the girl opens the curtains.
every morning, the mother makes tea.
every morning, the father checks the boat.
every morning, the dog waits by the door.

the boat is old but strong.
the rope is thick and dry.
the wood is dark and smooth.
the seat is narrow and hard.
the oars are long and straight.

the boy carries a small bag.
the girl carries a blue hat.
the father carries a net.
the mother carries bread.
the dog carries nothing.

they walk down the path.
they pass the tree.
they pass the fence.
they reach the shore.
they step into the boat.

the lake is calm in the morning.
the mist is low on the water.
the birds call from the reeds.
the fish move below the surface.
the boat drifts from the shore.

the father rows slowly.
the boy watches the water.
the girl points at the sky.
the mother smiles at the children.
the dog sits very still.

"look," said the girl.
"i see a bird," said the boy.
"i see two birds," said the mother.
"the day will be clear," said the father.
the dog gave no reply.

one white bird above the lake.
two white birds above the lake.
three white birds above the lake.
four white birds above the lake.
five white birds above the lake.

the sun climbs higher.
the mist fades away.
the water becomes bright.
the boat casts a shadow.
the shore grows small.

near the middle of the lake, the father stops.
near the middle of the lake, the boat turns.
near the middle of the lake, the water is deep.
near the middle of the lake, the fish are quick.
near the middle of the lake, the air is cool.

the father lowers the net.
the boy leans to see.
the girl holds the side.
the mother opens the basket.
the dog watches the net.

sometimes the net is empty.
sometimes the net is light.
sometimes the net is heavy.
sometimes the net shines with fish.
sometimes the net brings only weeds.

today the net is heavy.
today the fish are many.
today the father laughs.
today the boy claps.
today the girl sings.

they place the fish in a box.
they cover the box with cloth.
they wipe the floor of the boat.
they eat bread and cheese.
they drink cool water.

at noon, the lake is warm.
at noon, the birds are quiet.
at noon, the sky is pale.
at noon, the boat rocks gently.
at noon, the family rests.

after noon, clouds gather in the west.
after noon, the wind grows stronger.
after noon, the water becomes rough.
after noon, the father looks at the shore.
after noon, the mother closes the basket.

"it is time to go back," said the father.
"the wind is changing," said the mother.
"i hear the waves," said the boy.
"i will hold the box," said the girl.
the dog stood up at once.

the father rows hard.
the boat cuts through the water.
the wind pushes from behind.
the waves strike the side.
the box slides a little.

the boy holds the box.
the girl holds the bench.
the mother steadies the basket.
the father keeps the boat straight.
the dog stays close to the mother.

soon they see the tree.
soon they see the fence.
soon they see the path.
soon they see the shore.
soon they reach the house again.

rain begins as they land.
rain taps on the boat.
rain darkens the path.
rain bends the grass.
rain cools the air.

they run into the house.
they shut the door.
they dry the net.
they clean the fish.
they hang wet coats by the fire.

inside, the kitchen is warm.
inside, the fire burns bright.
inside, the kettle sings.
inside, the windows shine with rain.
inside, the table is ready.

the mother cooks the fish.
the father salts the soup.
the boy sets the plates.
the girl folds a cloth.
the dog lies near the fire.

the meal is simple.
the meal is hot.
the meal is good.
the meal is shared.
the meal ends with quiet smiles.

later, the rain stops.
later, the clouds break.
later, the lake grows still again.
later, the stars appear.
later, the moon rises over the water.

the boy stands at the window.
the girl stands beside him.
they look at the lake.
they look at the moon.
they look at the silent boat.

"it was a good day," said the boy.
"yes," said the girl.
"the lake was kind," said the mother.
"and the wind was strong," said the father.
the dog closed its eyes.

question: where is the house?
answer: the house is by the lake.

question: who rows the boat?
answer: the father rows the boat.

question: what is in the box?
answer: fish are in the box.

question: where does the dog lie?
answer: the dog lies near the fire.

this is a story about a lake.
this is a story about a family.
this is a story about work.
this is a story about weather.
this is a story about going out and coming home.

there is a house by the lake.
there is a path behind the house.
there is a fence by the path.
there is a tree by the fence.
there is a bench under the tree.`,
    };

    const text = samples[name] || '';
    document.getElementById('training-text').value = text;
    this.updateTextStats();
  },

  // --- Tokenization ---

  tokenizeText() {
    const text = document.getElementById('training-text').value.trim();
    if (text.length < 20) {
      alert('Please enter at least 20 characters of training text.');
      return;
    }

    const vocabSize = parseInt(document.getElementById('slider-vocab').value);
    this.model.maxVocabSize = vocabSize;
    const info = this.model.prepareData(text);
    this.showModelInfo(info);
    this.showTokenizationViz(text);

    // Show train section, disable tokenize button
    document.getElementById('train-section').style.display = 'block';
    document.getElementById('btn-tokenize').disabled = true;
    document.getElementById('btn-tokenize').textContent = 'Tokenized';
  },

  // --- Training ---

  startTraining() {
    if (!this.model.tokenizer) {
      alert('Please tokenize the text first!');
      return;
    }

    if (!this.model.trained && this.model.step === 0) {
      const lr = parseFloat(document.getElementById('slider-lr').value);
      this.model.learningRate = lr;
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
    document.getElementById('tokenization-viz').style.display = 'none';
    document.getElementById('train-section').style.display = 'none';
    document.getElementById('generation-section').style.display = 'none';
    document.getElementById('model-info').innerHTML = '';
    document.getElementById('btn-tokenize').disabled = false;
    document.getElementById('btn-tokenize').textContent = 'Tokenize';

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
          <div class="info-value">${info.vocabSize} tokens (${info.numMerges} merges)</div>
        </div>
        <div class="info-item">
          <div class="info-label">Compression</div>
          <div class="info-value">${info.compressionRatio}x</div>
        </div>
        <div class="info-item">
          <div class="info-label">Training examples</div>
          <div class="info-value">${info.numExamples.toLocaleString()}</div>
        </div>
        <div class="info-item">
          <div class="info-label">Parameters</div>
          <div class="info-value">${paramCount.toLocaleString()}</div>
        </div>
      </div>
    `;
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
      mergesHtml += `<div class="merge-rule" style="color:var(--text-muted)">... and ${tok.merges.length - 20} more merges</div>`;
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
      textHtml += `<span class="token-pill" style="opacity:0.4">... ${tokenIds.length - 300} more</span>`;
    }
    textHtml += '</div>';
    textContainer.innerHTML = textHtml;
  },

  // --- Visualization ---

  updateTrainingViz(result) {
    document.getElementById('step-counter').textContent = `Step: ${result.step.toLocaleString()}`;
    document.getElementById('loss-value').textContent = `Loss: ${result.avgLoss.toFixed(4)} · LR: ${result.effectiveLR.toFixed(4)}`;

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
    ctx.fillText('Training Steps', W / 2, H - 4);

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
    let html = `<div class="pred-context">Context: <span class="mono">${this.escapeHtml(ctxDisplay)}</span> → Target: <span class="mono">"${this.escapeHtml(targetDisplay)}"</span></div>`;
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
      if (/[a-z]/.test(token)) return '#4f8ff7';
      if (/[A-Z]/.test(token)) return '#7c5ff7';
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
      alert('Please train the model first!');
      return;
    }

    const { text, steps } = this.model.generate(seed, length, temp, topK, topP, repPen);

    const output = document.getElementById('generation-output');
    output.innerHTML = '';

    // Show seed
    const seedSpan = document.createElement('span');
    seedSpan.className = 'gen-seed';
    seedSpan.textContent = seed;
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
      span.title = `"${step.selected}" (${(step.selectedProb * 100).toFixed(1)}% confidence)`;
      output.appendChild(span);
      tokenIdx++;
      if (tokenIdx < steps.length) {
        setTimeout(typeToken, 40);
      } else {
        this.showGenerationSteps(steps);
      }
    };
    typeToken();
  },

  showGenerationSteps(steps) {
    const container = document.getElementById('generation-steps-viz');
    if (steps.length === 0) return;

    let html = '<h4>Generation Steps</h4>';
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
