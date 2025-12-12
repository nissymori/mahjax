const agentSelect = document.getElementById('agentSelect');
const modeSelect = document.getElementById('modeSelect');
const seatSelect = document.getElementById('seatSelect');
const seedInput = document.getElementById('seedInput');
const humanNameInput = document.getElementById('humanName');
const aiNameInput = document.getElementById('aiName');
const aiDelayInput = document.getElementById('aiDelay');
const hideOpponentsInput = document.getElementById('hideOpponents');
const startBtn = document.getElementById('startBtn');
const endBtn = document.getElementById('endBtn');

const boardEl = document.getElementById('board');
const handTilesEl = document.getElementById('handTiles');
const actionButtonsEl = document.getElementById('actionButtons');
const callButtonsEl = document.getElementById('callButtons');
const kanButtonsEl = document.getElementById('kanButtons');
const advanceButtonsEl = document.getElementById('advanceButtons');
const scoreRowsEl = document.getElementById('scoreRows');
const eventListEl = document.getElementById('eventList');
const statusBarEl = document.getElementById('statusBar');
const summaryOverlay = document.getElementById('roundSummary');
const summaryTitle = document.getElementById('summaryTitle');
const summaryBody = document.getElementById('summaryBody');
const summaryContinueBtn = document.getElementById('summaryContinue');

const handTitleEl = document.getElementById('handTitle');
const actionTitleEl = document.getElementById('actionTitle');
const scoreTitleEl = document.getElementById('scoreTitle');
const eventsTitleEl = document.getElementById('eventsTitle');
const scoreHeaderRow = document.getElementById('scoreHeaderRow');
const controlTextRefs = {
  agent: document.querySelector('[data-i18n="controls.agent"]'),
  mode: document.querySelector('[data-i18n="controls.mode"]'),
  humanSeat: document.querySelector('[data-i18n="controls.humanSeat"]'),
  seed: document.querySelector('[data-i18n="controls.seed"]'),
  humanName: document.querySelector('[data-i18n="controls.humanName"]'),
  aiName: document.querySelector('[data-i18n="controls.aiName"]'),
  aiDelay: document.querySelector('[data-i18n="controls.aiDelay"]'),
  hideOpponents: document.querySelector('[data-i18n="controls.hideOpponents"]'),
  start: document.querySelector('[data-i18n="controls.start"]'),
  end: document.querySelector('[data-i18n="controls.end"]'),
};
const modeOptionRefs = {
  hanchan: document.querySelector('[data-i18n-mode="hanchan"]'),
  one_round: document.querySelector('[data-i18n-mode="one_round"]'),
};
const seatOptionRefs = {
  auto: document.querySelector('[data-i18n-seat="auto"]'),
  east: document.querySelector('[data-i18n-seat="east"]'),
  south: document.querySelector('[data-i18n-seat="south"]'),
  west: document.querySelector('[data-i18n-seat="west"]'),
  north: document.querySelector('[data-i18n-seat="north"]'),
};
const LANGUAGE_BUTTON_SELECTOR = '#languageToggle .language-btn[data-lang]';

const Languages = { JA: 'ja', EN: 'en' };
const I18N = {
  ja: {
    code: 'ja',
    you: 'あなた',
    relativeSeats: ['あなた', '下家', '対面', '上家'],
    honors: ['東', '南', '西', '北', '白', '發', '中'],
    winds: {
      東: '東',
      南: '南',
      西: '西',
      北: '北',
      白: '白',
      發: '發',
      中: '中',
    },
    sections: {
      hand: '手牌',
      actions: 'アクション',
      score: 'スコア',
      events: 'ログ',
    },
    controls: {
      agent: 'Agent',
      mode: 'Mode',
      humanSeat: 'Human Seat',
      seed: 'Seed',
      humanName: 'Human',
      aiName: 'Agent Base Name',
      aiDelay: 'Agent Delay(ms)',
      hideOpponents: '相手の手牌を隠す',
      start: 'Start Game',
      end: 'End Game',
      modes: {
        hanchan: '半荘戦',
        one_round: '一局戦',
      },
      seats: {
        auto: 'ランダム',
        east: '東',
        south: '南',
        west: '西',
        north: '北',
      },
    },
    scoreboardHeaders: ['席', '名前', '点数', '直近'],
    actions: {
      tsumogiri: 'ツモ切り',
      riichi: '立直',
      tsumo: '自摸',
      ron: 'ロン',
      pass: 'パス',
      pon: 'ポン',
      chi: 'チー',
      openKan: '明槓',
      closedKan: '暗槓',
      addedKan: '加槓',
      advanceFinal: '終局',
      advanceNext: '次の局へ',
    },
    statuses: {
      idle: 'Choose settings and start a game.',
      sending: '送信中…',
      gameStarted: 'Game started.',
      noGame: 'No active game.',
      gameEnded: 'Game ended.',
      awaitingHuman: 'あなたの手番です。',
      awaitingAI: 'Agent 思考中…',
      roundSummaryPending: '結果を確認して「次の局へ」を押してください。',
      roundSummaryPrompt: (label) => `「${label}」を押して結果を表示してください。`,
      finished: 'Game finished.',
    },
    summaryReasons: {
      tsumo: '自摸',
      ron: 'ロン',
      abortive_draw_normal: '流局',
    },
    summary: {
      defaultTitle: (reason) => `局結果 (${reason})`,
      finalTitle: '終局',
      winnersHeader: '和了詳細',
      yakuLabel: '役',
      yakuman: (count) => `${count}倍役満`,
      fanFu: (fan, fu) => `${fan}翻 ${fu}符`,
      winningTile: '和了牌',
      winningTileFrom: (tile, rel, name) => `和了牌: ${tile} ← ${rel}(${name})`,
      meta: (honba, kyotaku) => `本場: ${honba}　供託: ${kyotaku}`,
      tableHeaders: ['順位', '席', '名前', '点数', '増減'],
      continue: '次の局へ',
      endCta: '終局',
    },
    advance: {
      next: '次の局へ',
      final: '終局',
    },
    kanKinds: {
      加槓: '加槓',
      暗槓: '暗槓',
    },
  },
  en: {
    code: 'en',
    you: 'You',
    relativeSeats: ['You', 'Right Player', 'Across', 'Left Player'],
    honors: ['East', 'South', 'West', 'North', 'White', 'Green', 'Red'],
    winds: {
      東: 'East',
      南: 'South',
      西: 'West',
      北: 'North',
      白: 'White',
      發: 'Green',
      中: 'Red',
    },
    sections: {
      hand: 'Hand',
      actions: 'Actions',
      score: 'Score',
      events: 'Log',
    },
    controls: {
      agent: 'Agent',
      mode: 'Mode',
      humanSeat: 'Seat',
      seed: 'Seed',
      humanName: 'Human',
      aiName: 'Agent Base Name',
      aiDelay: 'Agent Delay (ms)',
      hideOpponents: 'Hide opponent hands',
      start: 'Start Game',
      end: 'End Game',
      modes: {
        hanchan: 'Hanchan match',
        one_round: 'Single round',
      },
      seats: {
        auto: 'Random',
        east: 'East',
        south: 'South',
        west: 'West',
        north: 'North',
      },
    },
    scoreboardHeaders: ['Seat', 'Name', 'Points', 'Delta'],
    actions: {
      tsumogiri: 'Tsumogiri',
      riichi: 'Riichi',
      tsumo: 'Tsumo',
      ron: 'Ron',
      pass: 'Pass',
      pon: 'Pon',
      chi: 'Chi',
      openKan: 'Open Kan',
      closedKan: 'Concealed Kan',
      addedKan: 'Added Kan',
      advanceFinal: 'End Game',
      advanceNext: 'Next Round',
    },
    statuses: {
      idle: 'Choose settings and start a game.',
      sending: 'Sending...',
      gameStarted: 'Game started.',
      noGame: 'No active game.',
      gameEnded: 'Game ended.',
      awaitingHuman: 'Your turn.',
      awaitingAI: 'Agent is thinking...',
      roundSummaryPending: 'Review the result and press "Next Round".',
      roundSummaryPrompt: (label) => `Press "${label}" to view the result.`,
      finished: 'Game finished.',
    },
    summaryReasons: {
      tsumo: 'Tsumo',
      ron: 'Ron',
      abortive_draw_normal: 'Draw',
    },
    summary: {
      defaultTitle: (reason) => `Round Result (${reason})`,
      finalTitle: 'Game End',
      winnersHeader: 'Winning Details',
      yakuLabel: 'Yaku',
      yakuman: (count) => `${count}x Yakuman`,
      fanFu: (fan, fu) => `${fan} han ${fu} fu`,
      winningTile: 'Winning Tile',
      winningTileFrom: (tile, rel, name) => `Winning Tile: ${tile} ← ${rel} (${name})`,
      meta: (honba, kyotaku) => `Honba: ${honba}    Riichi Sticks: ${kyotaku}`,
      tableHeaders: ['Rank', 'Seat', 'Name', 'Points', 'Delta'],
      continue: 'Next Round',
      endCta: 'End Game',
    },
    advance: {
      next: 'Next Round',
      final: 'End Game',
    },
    kanKinds: {
      加槓: 'Added Kan',
      暗槓: 'Concealed Kan',
    },
  },
};

let currentLanguage = Languages.JA;
let currentGameId = null;
let eventHistory = [];
let aiTimer = null;
let latestState = null;
let isPendingAction = false;
let isUpdatingHideOpponents = false;
const pendingButtons = new Set();
let pendingDiscardVisual = null;
let pendingSummaryData = null;
let summaryRevealRequested = false;
let agentCache = [];

humanNameInput.dataset.autoFilled = 'false';
aiNameInput.dataset.autoFilled = 'false';

applyLocaleToStaticElements();
setStatus('idle');

document.addEventListener('click', (event) => {
  const target = event.target;
  if (!(target instanceof Element)) {
    return;
  }
  const btn = target.closest(LANGUAGE_BUTTON_SELECTOR);
  if (!btn) return;
  event.preventDefault();
  const { lang } = btn.dataset;
  if (lang) {
    setLanguage(lang);
  }
});

if (hideOpponentsInput) {
  hideOpponentsInput.addEventListener('change', () => {
    if (!currentGameId) {
      return;
    }
    const desired = Boolean(hideOpponentsInput.checked);
    updateHideOpponentsSetting(desired);
  });
}

function getLocale(lang = currentLanguage) {
  return I18N[lang] || I18N.ja;
}

function updateLanguageButtons() {
  const buttons = document.querySelectorAll(LANGUAGE_BUTTON_SELECTOR);
  buttons.forEach((btn) => {
    const lang = btn.dataset.lang;
    if (!lang) return;
    if (lang === currentLanguage) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
    btn.disabled = lang === currentLanguage;
  });
}

function syncHideOpponentsControl(state) {
  if (!hideOpponentsInput) return;
  if (state && typeof state.hideOpponentHands === 'boolean') {
    hideOpponentsInput.checked = Boolean(state.hideOpponentHands);
  }
  hideOpponentsInput.disabled = isUpdatingHideOpponents;
}

function applyLocaleToStaticElements() {
  const locale = getLocale();
  if (handTitleEl) handTitleEl.textContent = locale.sections.hand;
  if (actionTitleEl) actionTitleEl.textContent = locale.sections.actions;
  if (scoreTitleEl) scoreTitleEl.textContent = locale.sections.score;
  if (eventsTitleEl) eventsTitleEl.textContent = locale.sections.events;
  Object.entries(controlTextRefs).forEach(([key, el]) => {
    setTextContent(el, locale.controls?.[key]);
  });
  Object.entries(modeOptionRefs).forEach(([key, el]) => {
    setTextContent(el, locale.controls?.modes?.[key]);
  });
  Object.entries(seatOptionRefs).forEach(([key, el]) => {
    setTextContent(el, locale.controls?.seats?.[key]);
  });
  if (scoreHeaderRow) {
    scoreHeaderRow.innerHTML = '';
    locale.scoreboardHeaders.forEach((header) => {
      const th = document.createElement('th');
      th.textContent = header;
      scoreHeaderRow.appendChild(th);
    });
  }
  if (summaryContinueBtn) {
    summaryContinueBtn.textContent = locale.summary.continue;
  }
  if (!statusBarEl.textContent) {
    setStatus('idle');
  }
  document.documentElement.lang = locale.code;
  updateLanguageButtons();
  renderEventList();
}

function setTextContent(element, text) {
  if (!element || typeof text !== 'string') return;
  element.textContent = text;
}

function setLanguage(lang) {
  if (!Object.values(Languages).includes(lang)) return;
  if (lang === currentLanguage) return;
  currentLanguage = lang;
  applyLocaleToStaticElements();
  if (latestState) {
    renderState(latestState, {
      preservePending: true,
      skipEvents: true,
      skipAutoTimerReset: true,
    });
    updateBoardLanguageOnly();
  }
  if (pendingSummaryData && summaryOverlay.classList.contains('active')) {
    updateSummaryOverlay(pendingSummaryData);
  }
}

function setStatus(key, params = {}) {
  if (!statusBarEl) return;
  if (!key) {
    statusBarEl.textContent = params.message || params.fallback || '';
    return;
  }
  const locale = getLocale();
  const template = locale.statuses[key];
  if (typeof template === 'function') {
    statusBarEl.textContent = template(params);
  } else if (typeof template === 'string') {
    statusBarEl.textContent = template;
  } else if (params.message) {
    statusBarEl.textContent = params.message;
  }
}

function markPendingButton(button, actionType) {
  if (!button) return;
  pendingButtons.add(button);
  button.disabled = true;
  button.classList.add('pending');
  if (actionType === 'discard') {
    pendingDiscardVisual = {
      button,
      display: button.style.display,
    };
    button.style.display = 'none';
  }
}

function restorePendingVisuals() {
  pendingButtons.forEach((btn) => {
    btn.disabled = false;
    btn.classList.remove('pending');
  });
  pendingButtons.clear();
  if (pendingDiscardVisual) {
    const { button, display } = pendingDiscardVisual;
    if (button) {
      button.style.display = display || '';
      button.disabled = false;
      button.classList.remove('pending');
    }
  }
  pendingDiscardVisual = null;
}

function attachSendAction(button, action, actionType = 'generic') {
  if (!button) return;
  button.addEventListener('click', () => sendAction(action, { button, actionType }));
}

function playerOrder(state) {
  const total = state?.playerNames?.length || 0;
  const start = Number.isInteger(state?.humanSeat) ? state.humanSeat : 0;
  return Array.from({ length: total }, (_, i) => (start + i) % total);
}

function relativeIndex(state, absoluteSeat) {
  const total = state?.playerNames?.length || 0;
  if (!total) return 0;
  const base = Number.isInteger(state?.humanSeat) ? state.humanSeat : 0;
  return (absoluteSeat - base + total) % total;
}

function relativeSeatLabel(idx) {
  const locale = getLocale();
  return locale.relativeSeats[idx] || `Player ${idx}`;
}

async function fetchAgents() {
  const res = await fetch('/api/agents');
  if (!res.ok) {
    throw new Error('Failed to fetch agents');
  }
  return res.json();
}

function populateAgents(agents) {
  agentCache = [...agents];
  if (!agentSelect) return;
  agentSelect.innerHTML = '';
  agents.forEach((agent) => {
    const option = document.createElement('option');
    option.value = agent.id;
    option.textContent = agent.name;
    agentSelect.appendChild(option);
  });
  if (agentSelect.options.length > 0) {
    agentSelect.value = agentSelect.options[0].value;
    if (!aiNameInput.value) {
      const firstAgent = agents.find((agent) => agent.id === agentSelect.value) || agents[0];
      aiNameInput.value = firstAgent?.name ?? 'Agent';
      aiNameInput.dataset.autoFilled = 'true';
    }
  }
  if (!humanNameInput.value) {
    humanNameInput.value = 'You';
    humanNameInput.dataset.autoFilled = 'true';
  }
}

function collectGameRequest() {
  const agentId = agentSelect?.value;
  if (!agentId) {
    throw new Error('Agent is required');
  }
  const mode = modeSelect.value;
  const seatValue = seatSelect.value;
  const body = {
    agent_id: agentId,
    mode,
    random_seat: seatValue === 'auto',
    human_seat: seatValue === 'auto' ? null : Number(seatValue),
    human_name: humanNameInput.value.trim() || undefined,
    ai_name: aiNameInput.value.trim() || undefined,
    ai_delay_ms: Number(aiDelayInput.value) || 800,
    hide_opponent_hands: hideOpponentsInput ? Boolean(hideOpponentsInput.checked) : false,
  };
  if (seedInput.value) {
    body.seed = Number(seedInput.value);
  }
  return body;
}

async function startGame() {
  try {
    const body = collectGameRequest();
    const res = await fetch('/api/game', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'Failed to start game');
    }
    const state = await res.json();
    currentGameId = state.gameId;
    eventHistory = [];
    renderState(state);
    setStatus('gameStarted');
  } catch (err) {
    setStatus(null, { message: `Error: ${err.message}` });
  }
}

async function endGame() {
  if (!currentGameId) {
    setStatus('noGame');
    return;
  }
  await fetch(`/api/game/${currentGameId}`, { method: 'DELETE' });
  currentGameId = null;
  clearBoard();
  setStatus('gameEnded');
}

function clearBoard() {
  boardEl.innerHTML = '';
  handTilesEl.innerHTML = '';
  actionButtonsEl.innerHTML = '';
  callButtonsEl.innerHTML = '';
  kanButtonsEl.innerHTML = '';
  advanceButtonsEl.innerHTML = '';
  scoreRowsEl.innerHTML = '';
  eventListEl.innerHTML = '';
  summaryOverlay.classList.remove('active');
  if (aiTimer) {
    clearTimeout(aiTimer);
    aiTimer = null;
  }
  latestState = null;
  isPendingAction = false;
  pendingButtons.clear();
  pendingDiscardVisual = null;
  pendingSummaryData = null;
  summaryRevealRequested = false;
  renderEventList();
  updateLanguageButtons();
  isUpdatingHideOpponents = false;
  if (hideOpponentsInput) {
    hideOpponentsInput.disabled = false;
  }
}

async function sendAction(action, options = {}) {
  if (!currentGameId || isPendingAction) return;
  const { button = null, actionType = 'generic' } = options;
  isPendingAction = true;
  updateLanguageButtons();
  markPendingButton(button, actionType);
  setStatus('sending');
  try {
    const res = await fetch(`/api/game/${currentGameId}/action`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      setStatus(null, { message: data.detail || 'Failed to submit action' });
      restorePendingVisuals();
      if (actionType === 'advance') {
        summaryRevealRequested = false;
      }
      isPendingAction = false;
      updateLanguageButtons();
      return;
    }
    const state = await res.json();
    renderState(state);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    setStatus(null, { message: `Error: ${message}` });
    restorePendingVisuals();
    if (actionType === 'advance') {
      summaryRevealRequested = false;
    }
    isPendingAction = false;
    updateLanguageButtons();
  }
}

async function requestAutoStep() {
  if (!currentGameId || isPendingAction) return;
  const res = await fetch(`/api/game/${currentGameId}/auto`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ steps: 1 }),
  });
  if (!res.ok) {
    return;
  }
  const state = await res.json();
  renderState(state);
}

async function continueRound() {
  if (!currentGameId) return;
  const res = await fetch(`/api/game/${currentGameId}/continue`, { method: 'POST' });
  if (!res.ok) return;
  const state = await res.json();
  summaryOverlay.classList.remove('active');
  renderState(state);
}

async function updateHideOpponentsSetting(hide) {
  if (!currentGameId || isUpdatingHideOpponents) return;
  const previousValue = latestState?.hideOpponentHands ?? !hide;
  isUpdatingHideOpponents = true;
  if (hideOpponentsInput) {
    hideOpponentsInput.disabled = true;
  }
  try {
    const res = await fetch(`/api/game/${currentGameId}/visibility`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hide_opponent_hands: hide }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'Failed to update visibility');
    }
    const state = await res.json();
    renderState(state, {
      preservePending: true,
      skipEvents: true,
      skipAutoTimerReset: true,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    setStatus(null, { message });
    if (hideOpponentsInput) {
      hideOpponentsInput.checked = previousValue;
    }
  } finally {
    isUpdatingHideOpponents = false;
    if (hideOpponentsInput) {
      hideOpponentsInput.disabled = false;
    }
  }
}

function renderState(state, options = {}) {
  if (!state) return;
  const { preservePending = false, skipEvents = false, skipAutoTimerReset = false } = options;
  latestState = state;
  if (!preservePending) {
    isPendingAction = false;
    pendingButtons.clear();
    pendingDiscardVisual = null;
  }
  if (!skipAutoTimerReset && aiTimer) {
    clearTimeout(aiTimer);
    aiTimer = null;
  }
  boardEl.innerHTML = getBoardSvg(state);
  normalizeBoardSvg();
  renderHand(state);
  renderActions(state);
  renderRoundSummary(state);
  renderAdvance(state);
  renderScoreboard(state);
  syncHideOpponentsControl(state);
  if (skipEvents) {
    renderEventList();
  } else {
    appendEvents(state, state.events || []);
  }
  updateStatus(state);
  updateLanguageButtons();
  if (!skipAutoTimerReset && state.phase === 'awaiting_ai' && !state.roundSummary && !state.terminated) {
    aiTimer = setTimeout(requestAutoStep, state.aiDelayMs ?? 800);
  }
}

function getBoardSvg(state) {
  if (!state) return '';
  if (currentLanguage === Languages.EN) {
    return state.svgEnglish || state.svg || '';
  }
  return state.svgJapanese || state.svg || '';
}

function updateBoardLanguageOnly() {
  if (!latestState) return;
  boardEl.innerHTML = getBoardSvg(latestState);
  normalizeBoardSvg();
}

function normalizeBoardSvg() {
  const svg = boardEl.querySelector('svg');
  if (!svg) return;
  const viewBox = svg.getAttribute('viewBox');
  const widthAttr = svg.getAttribute('width');
  const heightAttr = svg.getAttribute('height');
  if (!viewBox && widthAttr && heightAttr) {
    const width = parseFloat(widthAttr);
    const height = parseFloat(heightAttr);
    if (!Number.isNaN(width) && !Number.isNaN(height)) {
      svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    }
  }
  if (!svg.getAttribute('preserveAspectRatio')) {
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  }
  svg.removeAttribute('width');
  svg.removeAttribute('height');
  svg.style.width = '100%';
  svg.style.height = 'auto';
  svg.style.display = 'block';
}

function renderHand(state) {
  handTilesEl.innerHTML = '';
  if (!state.hand || !state.hand.sequence) return;
  const discardMap = new Map();
  if (state.legalActions && state.legalActions.discardTiles) {
    state.legalActions.discardTiles.forEach((item) => {
      discardMap.set(item.tile, item.enabled);
    });
  }
  const baseSequence = state.hand.sequence;
  const separateLastDraw = Boolean(state.hand.separateLastDraw);
  const drawTile = separateLastDraw ? state.hand.drawTile : null;
  const lastDrawValue = state.hand.lastDraw;
  const lastIndex = separateLastDraw ? -1 : baseSequence.lastIndexOf(lastDrawValue);
  baseSequence.forEach((tile, idx) => {
    const btn = document.createElement('button');
    btn.className = 'tile-btn';
    btn.textContent = `${tileLabel(tile)}`;
    const enabled = discardMap.get(tile);
    if (!enabled) {
      btn.classList.add('disabled');
      btn.disabled = true;
    } else {
      attachSendAction(btn, tile, 'discard');
    }
    if (!separateLastDraw && lastIndex >= 0 && tile === lastDrawValue && idx === lastIndex) {
      btn.classList.add('last-draw');
    }
    handTilesEl.appendChild(btn);
  });
  if (separateLastDraw && drawTile !== null && drawTile !== undefined) {
    const btn = document.createElement('button');
    btn.className = 'tile-btn draw-separated last-draw';
    btn.textContent = `${tileLabel(drawTile)}`;
    const enabled = discardMap.get(drawTile);
    if (!enabled) {
      btn.classList.add('disabled');
      btn.disabled = true;
    } else {
      attachSendAction(btn, drawTile, 'discard');
    }
    handTilesEl.appendChild(btn);
  }
}

function renderActions(state) {
  actionButtonsEl.innerHTML = '';
  callButtonsEl.innerHTML = '';
  kanButtonsEl.innerHTML = '';
  if (!state.legalActions) return;

  const locale = getLocale();
  const { riichi, tsumogiri, tsumo, ron, pass: passAct } = state.legalActions;

  if (tsumogiri) {
    const btn = createActionButton(locale.actions.tsumogiri, tsumogiri.enabled);
    if (tsumogiri.enabled) attachSendAction(btn, tsumogiri.action, 'special');
    actionButtonsEl.appendChild(btn);
  }
  if (riichi) {
    const btn = createActionButton(locale.actions.riichi, riichi.enabled);
    if (riichi.enabled) attachSendAction(btn, riichi.action, 'special');
    actionButtonsEl.appendChild(btn);
  }
  if (tsumo) {
    const btn = createActionButton(locale.actions.tsumo, tsumo.enabled);
    if (tsumo.enabled) attachSendAction(btn, tsumo.action, 'special');
    actionButtonsEl.appendChild(btn);
  }
  if (ron) {
    const tileText = typeof ron.target === 'number' ? tileLabel(ron.target) : ron.targetLabel;
    const label = tileText ? `${locale.actions.ron} ${tileText}` : locale.actions.ron;
    const btn = createActionButton(label, ron.enabled);
    if (ron.enabled) attachSendAction(btn, ron.action, 'special');
    actionButtonsEl.appendChild(btn);
  }
  if (passAct) {
    const btn = createActionButton(locale.actions.pass, passAct.enabled);
    if (passAct.enabled) attachSendAction(btn, passAct.action, 'special');
    actionButtonsEl.appendChild(btn);
  }

  if (state.legalActions.kan && state.legalActions.kan.length) {
    state.legalActions.kan.forEach((item) => {
      const kindLabel = translateKanKind(item.kind);
      const label = `${kindLabel} ${tileLabel(item.tile)}`;
      const btn = createActionButton(label, true);
      attachSendAction(btn, item.action, 'special');
      kanButtonsEl.appendChild(btn);
    });
  }

  const call = state.legalActions.call || {};
  if (call.pon) {
    const label = `${locale.actions.pon} ${formatTileSequence(call.pon.tiles || [])}`;
    const btn = createActionButton(label || locale.actions.pon, true);
    attachSendAction(btn, call.pon.action, 'special');
    callButtonsEl.appendChild(btn);
  }
  if (call.open_kan) {
    const label = `${locale.actions.openKan} ${formatTileSequence(call.open_kan.tiles || [])}`;
    const btn = createActionButton(label || locale.actions.openKan, true);
    attachSendAction(btn, call.open_kan.action, 'special');
    callButtonsEl.appendChild(btn);
  }
  if (Array.isArray(call.chi)) {
    call.chi.forEach((item) => {
      const label = `${locale.actions.chi} ${formatTileSequence(item.tiles || [])}`;
      const btn = createActionButton(label || locale.actions.chi, true);
      attachSendAction(btn, item.action, 'special');
      callButtonsEl.appendChild(btn);
    });
  }
}

function renderAdvance(state) {
  advanceButtonsEl.innerHTML = '';
  if (!pendingSummaryData || summaryOverlay.classList.contains('active')) return;
  const advanceInfo = state.advanceAction || state.legalActions?.advance;
  if (!advanceInfo || advanceInfo.enabled === false) return;
  const label = localizeAdvanceLabel(advanceInfo.label, advanceInfo.isFinal);
  const btn = createActionButton(label, true);
  btn.addEventListener('click', () => {
    if (typeof advanceInfo.action === 'number') {
      summaryRevealRequested = true;
      sendAction(advanceInfo.action, { button: btn, actionType: 'advance' });
    } else {
      showRoundSummary();
    }
  });
  advanceButtonsEl.appendChild(btn);
}

function createActionButton(label, enabled) {
  const btn = document.createElement('button');
  btn.textContent = label;
  if (!enabled) {
    btn.classList.add('secondary');
    btn.disabled = true;
  }
  return btn;
}

function renderScoreboard(state) {
  scoreRowsEl.innerHTML = '';
  if (!state.scores) return;
  const locale = getLocale();
  const order = playerOrder(state);
  order.forEach((seatIdx, relativeIdx) => {
    const score = state.scores[seatIdx];
    const tr = document.createElement('tr');
    if (relativeIdx === 0) {
      tr.style.fontWeight = '700';
    }
    const seatCell = document.createElement('td');
    const relLabel = relativeSeatLabel(relativeIdx);
    const wind = translateWindName(state.winds[seatIdx]);
    seatCell.textContent = `${relLabel} (${wind})`;
    const nameCell = document.createElement('td');
    nameCell.textContent = state.playerNames[seatIdx];
    const scoreCell = document.createElement('td');
    scoreCell.textContent = Number(score).toLocaleString();
    const deltaCell = document.createElement('td');
    const delta = state.rewards[seatIdx];
    deltaCell.textContent = delta === 0 ? '-' : (delta > 0 ? `+${delta}` : `${delta}`);
    tr.appendChild(seatCell);
    tr.appendChild(nameCell);
    tr.appendChild(scoreCell);
    tr.appendChild(deltaCell);
    scoreRowsEl.appendChild(tr);
  });
}

function appendEvents(state, events) {
  if (events && events.length) {
    events.forEach((evt) => {
      const relIdx = relativeIndex(state, evt.player);
      eventHistory.unshift({
        ...evt,
        relativeIndex: relIdx,
      });
    });
    eventHistory = eventHistory.slice(0, 50);
  }
  renderEventList();
}

function renderEventList() {
  eventListEl.innerHTML = '';
  const locale = getLocale();
  eventHistory.forEach((evt) => {
    const li = document.createElement('li');
    const time = new Date(evt.timestamp * 1000).toLocaleTimeString();
    const relLabel = relativeSeatLabel(evt.relativeIndex);
    const name = evt.relativeIndex === 0 ? locale.you : evt.playerName;
    const description = translateEventDescription(evt.description);
    li.textContent = `[${time}] ${relLabel}(${name}) ${description}`;
    eventListEl.appendChild(li);
  });
}

function renderRoundSummary(state) {
  const summary = state.roundSummary;
  if (!summary) {
    summaryOverlay.classList.remove('active');
    pendingSummaryData = null;
    summaryRevealRequested = false;
    return;
  }
  pendingSummaryData = {
    summary,
    playerNames: Array.isArray(state.playerNames) ? [...state.playerNames] : [],
    winds: Array.isArray(state.winds) ? [...state.winds] : [],
    scores: Array.isArray(state.scores) ? [...state.scores] : [],
    rewards: Array.isArray(state.rewards) ? [...state.rewards] : [],
    rankOrder: Array.isArray(state.rankOrder) ? [...state.rankOrder] : [],
    humanSeat: Number.isInteger(state.humanSeat) ? state.humanSeat : 0,
  };
  if (summaryOverlay.classList.contains('active')) {
    updateSummaryOverlay(pendingSummaryData);
  }
}

function showRoundSummary() {
  if (!pendingSummaryData) return;
  updateSummaryOverlay(pendingSummaryData);
  summaryOverlay.classList.add('active');
}

function updateSummaryOverlay(data) {
  const { summary, playerNames, winds, scores, rewards, rankOrder, humanSeat } = data;
  const locale = getLocale();
  const reasonLabel = summaryReasonLabel(summary.reason);
  summaryTitle.textContent = summary.isGameEnd
    ? locale.summary.finalTitle
    : locale.summary.defaultTitle(reasonLabel);

  const body = document.createElement('div');
  const order = Array.isArray(rankOrder) && rankOrder.length
    ? rankOrder
    : playerNames.map((_, idx) => idx);
  if (order.length) {
    const table = document.createElement('table');
    table.className = 'summary-table';
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    locale.summary.tableHeaders.forEach((label) => {
      const th = document.createElement('th');
      th.textContent = label;
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    order.forEach((seatIdx, rankIdx) => {
      const tr = document.createElement('tr');
      if (seatIdx === humanSeat) {
        tr.style.fontWeight = '700';
      }
      const rankCell = document.createElement('td');
      rankCell.textContent = `${rankIdx + 1}`;
      const seatCell = document.createElement('td');
      const wind = translateWindName(winds[seatIdx]);
      seatCell.textContent = wind ? `${wind}` : `Seat ${seatIdx + 1}`;
      const nameCell = document.createElement('td');
      nameCell.textContent = playerNames[seatIdx] || `Player ${seatIdx + 1}`;
      const scoreCell = document.createElement('td');
      const scoreVal = scores[seatIdx] ?? 0;
      scoreCell.textContent = Number(scoreVal).toLocaleString();
      const deltaCell = document.createElement('td');
      const delta = rewards[seatIdx] ?? 0;
      deltaCell.textContent = delta === 0 ? '-' : (delta > 0 ? `+${delta}` : `${delta}`);
      [rankCell, seatCell, nameCell, scoreCell, deltaCell].forEach((cell) => {
        tr.appendChild(cell);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    body.appendChild(table);
  }

  if (Array.isArray(summary.winners) && summary.winners.length) {
    const winnersHeader = document.createElement('h3');
    winnersHeader.textContent = locale.summary.winnersHeader;
    body.appendChild(winnersHeader);
    summary.winners.forEach((winner) => {
      const section = document.createElement('div');
      section.className = 'winner-section';
      const name = document.createElement('div');
      name.innerHTML = `<strong>${winner.name}</strong> (+${winner.pointsDelta})`;
      section.appendChild(name);
      if (winner.yaku && winner.yaku.length) {
        const yaku = document.createElement('div');
        yaku.textContent = `${locale.summary.yakuLabel}: ${winner.yaku.join(', ')}`;
        section.appendChild(yaku);
      }
      const detail = document.createElement('div');
      if (winner.yakuman > 0) {
        detail.textContent = locale.summary.yakuman(winner.yakuman);
      } else {
        detail.textContent = locale.summary.fanFu(winner.fan, winner.fu);
      }
      section.appendChild(detail);
      if (winner.winningTile !== undefined && winner.winningTile !== null) {
        const tileInfo = document.createElement('div');
        const tileText = tileLabel(winner.winningTile);
        if (typeof winner.fromPlayer === 'number') {
          const relIdx = relativeIndexFromSeat(humanSeat, winner.fromPlayer);
          const relLabel = relativeSeatLabel(relIdx);
          const fromName = playerNames[winner.fromPlayer] || `Player ${winner.fromPlayer + 1}`;
          tileInfo.textContent = locale.summary.winningTileFrom(tileText, relLabel, fromName);
        } else {
          tileInfo.textContent = `${locale.summary.winningTile}: ${tileText}`;
        }
        section.appendChild(tileInfo);
      }
      body.appendChild(section);
    });
  }

  const meta = document.createElement('div');
  meta.textContent = locale.summary.meta(summary.honba, summary.kyotaku);
  body.appendChild(meta);

  summaryBody.innerHTML = '';
  summaryBody.appendChild(body);
  summaryContinueBtn.textContent = summary.isGameEnd
    ? locale.summary.endCta
    : locale.summary.continue;
}

function summaryReasonLabel(reason) {
  const locale = getLocale();
  return locale.summaryReasons[reason] || reason;
}

function relativeIndexFromSeat(humanSeat, absoluteSeat) {
  const total = 4;
  return (absoluteSeat - humanSeat + total) % total;
}

function updateStatus(state) {
  if (state.terminated) {
    setStatus('finished');
    return;
  }
  const locale = getLocale();
  switch (state.phase) {
    case 'awaiting_human':
      setStatus('awaitingHuman');
      break;
    case 'awaiting_ai':
      setStatus('awaitingAI');
      break;
    case 'round_end':
      if (summaryOverlay.classList.contains('active')) {
        setStatus('roundSummaryPending');
      } else {
        const label = pendingSummaryData?.summary?.isGameEnd
          ? locale.advance.final
          : locale.advance.next;
        setStatus('roundSummaryPrompt', { label });
      }
      break;
    default:
      break;
  }
}

function tileLabel(tile, lang = currentLanguage) {
  const locale = getLocale(lang);
  if (tile < 0) return '';
  if (tile < 9) return `${tile + 1}m`;
  if (tile < 18) return `${tile - 8}p`;
  if (tile < 27) return `${tile - 17}s`;
  return locale.honors[tile - 27] || `${tile}`;
}

function formatTileSequence(tiles) {
  if (!Array.isArray(tiles) || !tiles.length) return '';
  const joiner = currentLanguage === Languages.JA ? '' : ' ';
  return tiles.map((tile) => tileLabel(tile)).join(joiner);
}

function translateKanKind(kind) {
  const locale = getLocale();
  return locale.kanKinds[kind] || kind || locale.actions.openKan;
}

function translateWindName(wind) {
  const locale = getLocale();
  return locale.winds[wind] || wind;
}

function translateEventDescription(description) {
  if (currentLanguage === Languages.JA) return description;
  if (!description) return '';
  if (description.startsWith('打 ')) {
    return `Discard ${description.slice(2)}`;
  }
  if (description.startsWith('カン ')) {
    return `Kan ${description.slice(3)}`;
  }
  if (description === 'ツモ切り') return 'Tsumogiri';
  if (description === '立直宣言') return 'Riichi declaration';
  if (description === '自摸') return 'Tsumo';
  if (description.startsWith('ロン')) return description.replace('ロン', 'Ron');
  if (description.startsWith('ポン')) return description.replace('ポン', 'Pon');
  if (description.startsWith('明槓')) return description.replace('明槓', 'Open Kan');
  if (description.startsWith('チー')) return description.replace('チー', 'Chi');
  if (description === 'パス') return 'Pass';
  if (description === '進行') return 'Advance';
  if (description.startsWith('アクション')) return description.replace('アクション', 'Action');
  return description;
}

function localizeAdvanceLabel(label, isFinal) {
  const locale = getLocale();
  if (currentLanguage === Languages.JA) {
    return label || (isFinal ? locale.advance.final : locale.advance.next);
  }
  if (!label) {
    return isFinal ? locale.advance.final : locale.advance.next;
  }
  if (label === '終局') return locale.advance.final;
  if (label === '次の局へ') return locale.advance.next;
  return label;
}

summaryContinueBtn.addEventListener('click', continueRound);
startBtn.addEventListener('click', startGame);
endBtn.addEventListener('click', endGame);
aiNameInput.addEventListener('input', () => {
  aiNameInput.dataset.autoFilled = 'false';
});
humanNameInput.addEventListener('input', () => {
  humanNameInput.dataset.autoFilled = 'false';
});
if (agentSelect) {
  agentSelect.addEventListener('change', () => {
    if (aiNameInput.dataset.autoFilled === 'false') return;
    const selected = agentCache.find((agent) => agent.id === agentSelect.value);
    if (selected) {
      aiNameInput.value = selected.name;
      aiNameInput.dataset.autoFilled = 'true';
    }
  });
}
(async function init() {
  try {
    const agents = await fetchAgents();
    populateAgents(agents);
  } catch (err) {
    setStatus(null, { message: `Failed to load agents: ${err.message}` });
  }
})();
