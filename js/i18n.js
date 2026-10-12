// ============================================================
// i18n — English (default) and Brazilian Portuguese
//
// Elements with data-i18n keep their English HTML in index.html (so the page
// works without JavaScript and stays indexable). On load, the English HTML is
// captured from the DOM and the Portuguese version comes from I18N.pt.
// Strings built in JavaScript use t('key', { vars }).
// ============================================================

const I18N = {
  en: {
    // --- strings used in JavaScript ---
    'js.minChars': 'Please enter at least 20 characters of training text.',
    'js.goBack': 'Going back will reset steps after this one. Continue?',
    'js.tokenizeFirst': 'Please tokenize the text first!',
    'js.tooShort': 'Text is too short to create training examples. Please provide more text.',
    'js.trainFirst': 'Please train the model first!',
    'js.trainModel': 'Train Model',
    'js.pause': 'Pause',
    'js.resume': 'Resume',
    'js.tokenize': 'Tokenize',
    'js.retokenize': 'Re-tokenize',
    'js.step': 'Step: {n}',
    'js.loss': 'Loss: {loss} · LR: {lr}',
    'js.lossEmpty': 'Loss: —',
    'js.textStats': '{chars} characters &middot; {words} words &middot; {unique} unique chars',
    'js.vocabulary': 'Vocabulary',
    'js.vocabValue': '{size} tokens ({merges} merges)',
    'js.compression': 'Compression',
    'js.trainingExamples': 'Training examples',
    'js.parameters': 'Parameters',
    'js.moreMerges': '... and {n} more merges',
    'js.moreTokens': '... {n} more',
    'js.trainingSteps': 'Training Steps',
    'js.context': 'Context',
    'js.target': 'Target',
    'js.confidence': '{p}% confidence',
    'js.generationSteps': 'Generation Steps',
    'demo.inputText': 'Input text:',
    'demo.split': 'Step 1: Split into characters',
    'demo.merge': 'Merge "{a}" + "{b}" → "{r}" (most frequent pair)',
    'demo.summary': '{chars} characters → {tokens} tokens ({ratio}x compression)',
    'demo.contextTokens': 'Context tokens:',
    'demo.nnPrediction': 'Neural Network Prediction',
    'doc.title': 'Tiny Language Model — Learn How Language Models Work',
  },

  pt: {
    // --- page text (keys match data-i18n attributes in index.html) ---
    'nav.learn': 'Aprender',
    'nav.lab': 'Laboratório',
    'hero.title': 'Como funcionam os <span class="gradient-text">modelos de linguagem</span>',
    'hero.sub': 'Um guia prático das ideias centrais por trás dos modelos de linguagem, do zero.',

    's1.title': 'O que é um modelo de linguagem?',
    's1.p': 'Um modelo de linguagem é um programa que prevê o próximo token a partir do que veio antes. Só isso. Quando você digita "O gato sentou no", o modelo calcula que "tapete" é mais provável que "xilofone". Encadeando essas previsões, ele consegue gerar parágrafos inteiros de texto coerente.',
    's1.c1.t': 'Ideia central',
    's1.c1.p': 'Dado um trecho de texto, prever o que vem a seguir. Cada previsão é uma distribuição de probabilidades sobre todos os próximos tokens possíveis.',
    's1.c2.t': 'Por que funciona',
    's1.c2.p': 'A linguagem tem padrões. Depois de aprender milhões de padrões em textos, o modelo captura gramática, fatos e até raciocínio, tudo a partir da previsão do próximo token.',
    's1.c3.t': 'O tamanho importa',
    's1.c3.p': 'Os modelos mais avançados podem ter trilhões de parâmetros. O nosso modelo minúsculo tem cerca de 20 mil. É a mesma ideia, em escalas muito diferentes, mas os fundamentos são idênticos.',

    's2.title': 'Tokenização',
    's2.p': 'Antes de processar um texto, o modelo precisa quebrá-lo em <strong>tokens</strong>: pedaços com significado, cada um com um número de identificação. Nosso modelo usa o <strong>Byte Pair Encoding (BPE)</strong>, o mesmo algoritmo usado pelo GPT e pela maioria dos modelos de linguagem atuais. O BPE começa com as letras soltas e vai juntando os pares vizinhos mais frequentes em novos tokens, como "ato", "ão" ou "que".',
    's2.math': '<span class="comment">// Algoritmo BPE:</span><br>1. Comece com cada caractere como um token<br>2. Conte todos os pares de tokens vizinhos<br>3. Junte o par mais frequente em um novo token<br>4. Repita até atingir o vocab_size<br><br><span class="comment">// Resultado: padrões comuns viram um único token</span><br>"que" → [token_42], &nbsp; " de" → [token_87], &nbsp; "ão" → [token_63]<br>vocab_size = caracteres de base + número de fusões (em geral, de 100 a 300)',

    's3.title': 'Embeddings',
    's3.p': 'Depois da tokenização, cada token é só um número, um ID como 0, 1 ou 42. Mas um número sozinho não diz nada ao modelo sobre o que o token <em>significa</em>. O modelo precisa de uma representação mais rica: uma lista de números (um <strong>vetor</strong>) que capture as propriedades do token. É isso que é um <strong>embedding</strong>.',
    's3.p2': 'Pense em coordenadas. Uma cidade pode ser descrita por dois números (latitude e longitude) que dizem onde ela fica em relação às outras. Um embedding faz o mesmo para um token, mas com muito mais dimensões. Cada dimensão pode captar, de forma aproximada, algo como "quanto isto parece um substantivo?" ou "isto aparece no começo das frases?", mas é o próprio modelo que descobre essas dimensões durante o treino.',
    's3.c1.t': 'A tabela de consulta',
    's3.c1.p': 'O modelo guarda uma grande tabela (uma matriz) em que cada linha é o embedding de um token. Para pegar o embedding do token nº 42, basta ler a linha 42. O número de dimensões é um parâmetro ajustável: nosso modelo minúsculo usa 8 por padrão, enquanto modelos como o GPT usam milhares. Você pode mudar isso no Laboratório.',
    's3.c2.t': 'Tokens parecidos ficam perto',
    's3.c2.p': 'Conforme o modelo treina, tokens que se comportam de forma parecida, como "o" e "a", ou "corre" e "anda", acabam com embeddings parecidos (vetores próximos). Dá para ver isso no gráfico de embeddings do Laboratório.',
    's3.c3.t': 'Janela de contexto',
    's3.c3.p': 'O modelo olha vários tokens de uma vez (uma "janela de contexto" de 10 tokens, no nosso caso). Ele junta os embeddings deles num único vetor longo, o que dá à rede neural uma visão completa do contexto recente.',
    's3.math': '<span class="comment">// A tabela de embeddings: uma linha por token, cada linha com embed_dim números</span><br>C = matriz de formato [vocab_size, embed_dim]<br><br><span class="comment">// Exemplo: o token "que" tem ID 42</span><br><span class="comment">// O embedding dele é a linha 42 da tabela, um vetor de 8 números</span><br>C[42] = [0.23, -0.71, 0.45, 0.12, ...]<br><br><span class="comment">// Para entregar 10 tokens à rede, junte os embeddings deles</span><br>input = concat(C[tok_0], C[tok_1], C[tok_2], C[tok_3], C[tok_4], ...)<br><span class="comment">// Resultado: um único vetor de 10 × 8 = 80 números</span>',

    's4.title': 'A rede neural',
    's4.p': 'Os embeddings juntos passam por uma rede neural que aprende a ligar padrões da entrada a previsões sobre o próximo token. Nosso modelo usa uma camada oculta com a função de ativação <strong>tanh</strong>.',
    's4.a1': 'Entrada<br><small>10 tokens</small>',
    's4.a2': 'Embeddings<br><small>10 × 8 = 80</small>',
    's4.a3': 'Camada oculta<br><small>128 neurônios, tanh</small>',
    's4.a4': 'Previsões<br><small>softmax → probabilidades</small>',
    's4.math': '<span class="comment">// Passagem pela rede (forward pass)</span><br>hidden = tanh(input_vector × W1 + b1) &nbsp; <span class="comment">// 128 neurônios aprendem padrões</span><br>logits = hidden × W2 + b2 &nbsp; <span class="comment">// pontuação bruta de cada token</span><br>probs = softmax(logits) &nbsp; <span class="comment">// vira probabilidade (a soma dá 1)</span><br><br><span class="comment">// Fórmula do softmax: transforma quaisquer números em probabilidades</span><br>softmax(x_i) = exp(x_i) / soma(exp(x_j) para todo j)',

    's5.title': 'Treino',
    's5.p': 'O treino é como o modelo aprende. Mostramos exemplos tirados do texto: "dados estes 10 tokens, o próximo deveria ser X". O modelo faz uma previsão, medimos o quanto ela errou (a <strong>perda</strong>, ou <em>loss</em>) e então ajustamos um pouco todos os parâmetros para acertar mais na próxima vez. Isso se chama <strong>descida do gradiente</strong>.',
    's5.math': '<span class="comment">// 1. Perda: o quanto a previsão errou?</span><br>loss = -log(probabilidade dada ao token correto)<br><span class="comment">// Se o modelo deu 90% à resposta certa: loss = 0.105 (baixa = bom)</span><br><span class="comment">// Se o modelo deu 1% à resposta certa: loss = 4.605 (alta = ruim)</span><br><br><span class="comment">// 2. Gradientes: para que lado ajustar cada parâmetro?</span><br>gradients = d(loss) / d(parameters) &nbsp; <span class="comment">// calculados por retropropagação</span><br><br><span class="comment">// 3. Atualização: empurrar os parâmetros para reduzir a perda</span><br>parameters = parameters - learning_rate × gradients<br><span class="comment">// learning_rate controla o tamanho do passo (em geral, de 0.001 a 0.1)</span>',
    's5.c1.t': 'Treino em lotes',
    's5.c1.p': 'Em vez de atualizar depois de cada exemplo, tiramos a média dos gradientes de um "minilote" de exemplos (por exemplo, 64 de cada vez). Isso deixa o treino mais rápido e mais estável.',
    's5.c2.t': 'Sobreajuste (overfitting)',
    's5.c2.p': 'Com treino suficiente, o modelo decora o texto de treino. No nosso modelo minúsculo, isso é esperado! Os modelos de verdade treinam com trilhões de tokens justamente para evitar isso.',

    's6.title': 'Geração',
    's6.p': 'Para gerar texto, começamos com um trecho inicial e prevemos o próximo token várias vezes seguidas. Cada previsão é <strong>sorteada</strong> a partir da distribuição de probabilidades: nem sempre escolhemos o token mais provável. O parâmetro <strong>temperatura</strong> controla o quanto o resultado varia.',
    's6.math': '<span class="comment">// A temperatura controla a aleatoriedade</span><br>scaled_logits = logits / temperature<br><br>temperature = 0.1 → quase sempre escolhe a previsão mais provável (previsível)<br>temperature = 1.0 → sorteia na proporção da distribuição<br>temperature = 2.0 → mais aleatório, mais criativo e caótico',
    's6.cta': 'Experimente no Laboratório →',

    'lab.title': 'O Laboratório',
    'lab.sub': 'Treine o seu próprio modelo de linguagem minúsculo direto no navegador. Coloque um texto, aperte treinar e veja o modelo aprender.',
    'lab.s1': 'Dados de treino',
    'lab.s1.p': 'Cole um texto, carregue um exemplo ou envie um arquivo .txt. Quanto mais texto, mais padrões para aprender.',
    'lab.ex1': 'Exemplo 1',
    'lab.ex2': 'Exemplo 2',
    'lab.ex3': 'Exemplo 3',
    'lab.textarea': 'Cole ou digite aqui o texto de treino...',
    'lab.upload': 'Enviar arquivo .txt',
    'lab.next': 'Próximo',
    'lab.s2': 'Tokenizar',
    'lab.s2.p': 'Rode a tokenização BPE no seu texto para montar um vocabulário antes do treino.',
    'lab.vocab': 'Tamanho do vocabulário <span class="label-unit">tokens</span>',
    'lab.tokresult': 'Resultado da tokenização',
    'lab.tokresult.p': 'O BPE aprendeu estas regras de fusão com o seu texto. Passe o mouse sobre os tokens para ver os IDs.',
    'lab.tokentext': 'Texto tokenizado (primeiros 300 tokens)',
    'lab.s3': 'Treinar',
    'lab.embed': 'Dimensões do embedding',
    'lab.context': 'Janela de contexto <span class="label-unit">tokens</span>',
    'lab.lr': 'Taxa de aprendizado',
    'lab.speed': 'Velocidade do treino <span class="label-unit">passos/quadro</span>',
    'lab.progress': 'Progresso do treino',
    'lab.losschart': 'Perda ao longo do tempo',
    'lab.embchart': 'Embeddings dos tokens (projeção 2D)',
    'lab.live': 'Previsões ao vivo',
    'lab.reset': 'Recomeçar',
    'lab.genstep': 'Gerar →',
    'lab.s4': 'Gerar texto',
    'lab.s4.p': 'Digite um trecho inicial e veja o modelo gerar o texto token por token. A transparência mostra a confiança.',
    'lab.temp': 'Temperatura',
    'lab.topk': 'Top-K <span class="label-unit">(0 = desligado)</span>',
    'lab.topp': 'Top-P <span class="label-unit">(1.0 = desligado)</span>',
    'lab.reppen': 'Penalidade de repetição <span class="label-unit">(1.0 = desligado)</span>',
    'lab.length': 'Tamanho <span class="label-unit">tokens</span>',
    'lab.prompt': 'Digite um trecho inicial (por exemplo, "o ")...',
    'lab.generate': 'Gerar',
    'lab.back': 'Voltar ao treino',

    // --- strings used in JavaScript ---
    'js.minChars': 'Coloque pelo menos 20 caracteres de texto de treino.',
    'js.goBack': 'Voltar vai apagar as etapas seguintes. Continuar?',
    'js.tokenizeFirst': 'Tokenize o texto primeiro!',
    'js.tooShort': 'O texto é curto demais para criar exemplos de treino. Coloque mais texto.',
    'js.trainFirst': 'Treine o modelo primeiro!',
    'js.trainModel': 'Treinar o modelo',
    'js.pause': 'Pausar',
    'js.resume': 'Continuar',
    'js.tokenize': 'Tokenizar',
    'js.retokenize': 'Tokenizar de novo',
    'js.step': 'Passo: {n}',
    'js.loss': 'Perda: {loss} · Taxa: {lr}',
    'js.lossEmpty': 'Perda: —',
    'js.textStats': '{chars} caracteres &middot; {words} palavras &middot; {unique} caracteres diferentes',
    'js.vocabulary': 'Vocabulário',
    'js.vocabValue': '{size} tokens ({merges} fusões)',
    'js.compression': 'Compressão',
    'js.trainingExamples': 'Exemplos de treino',
    'js.parameters': 'Parâmetros',
    'js.moreMerges': '... e mais {n} fusões',
    'js.moreTokens': '... mais {n}',
    'js.trainingSteps': 'Passos de treino',
    'js.context': 'Contexto',
    'js.target': 'Alvo',
    'js.confidence': '{p}% de confiança',
    'js.generationSteps': 'Passos da geração',
    'demo.inputText': 'Texto de entrada:',
    'demo.split': 'Passo 1: separar em caracteres',
    'demo.merge': 'Juntar "{a}" + "{b}" → "{r}" (par mais frequente)',
    'demo.summary': '{chars} caracteres → {tokens} tokens (compressão de {ratio}x)',
    'demo.contextTokens': 'Tokens de contexto:',
    'demo.nnPrediction': 'Previsão da rede neural',
    'doc.title': 'Tiny Language Model: aprenda como funcionam os modelos de linguagem',
  },
};

// Language: ?lang=pt|en, then the saved choice, then the browser language.
function detectLang() {
  const fromUrl = new URLSearchParams(location.search).get('lang');
  if (fromUrl === 'pt' || fromUrl === 'en') return fromUrl;
  try {
    const saved = localStorage.getItem('tinylm-lang');
    if (saved === 'pt' || saved === 'en') return saved;
  } catch (e) { /* storage unavailable */ }
  return (navigator.language || 'en').toLowerCase().startsWith('pt') ? 'pt' : 'en';
}

let LANG = detectLang();

function t(key, vars) {
  let s = (I18N[LANG] && I18N[LANG][key]) || I18N.en[key] || key;
  if (vars) for (const k in vars) s = s.split(`{${k}}`).join(vars[k]);
  return s;
}

function applyI18n() {
  document.documentElement.lang = LANG === 'pt' ? 'pt-BR' : 'en';
  document.title = t('doc.title');
  document.querySelectorAll('[data-i18n]').forEach(el => {
    if (el.dataset.en === undefined) el.dataset.en = el.innerHTML;
    const pt = I18N.pt[el.dataset.i18n];
    el.innerHTML = LANG === 'pt' && pt ? pt : el.dataset.en;
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    if (el.dataset.enPlaceholder === undefined) el.dataset.enPlaceholder = el.placeholder;
    const pt = I18N.pt[el.dataset.i18nPlaceholder];
    el.placeholder = LANG === 'pt' && pt ? pt : el.dataset.enPlaceholder;
  });
  document.querySelectorAll('.lang-btn').forEach(b => b.classList.toggle('active', b.dataset.lang === LANG));
}

function setLang(lang) {
  LANG = lang;
  try { localStorage.setItem('tinylm-lang', lang); } catch (e) { /* storage unavailable */ }
  applyI18n();
  if (typeof app !== 'undefined' && app.onLangChange) app.onLangChange();
}
