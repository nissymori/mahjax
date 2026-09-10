// Play and replay share one board. The server sends a list of positions and the
// browser plays through them, so nothing here decides what is legal -- it only
// draws what arrived and sends back the action the player picked.

const $ = (id) => document.getElementById(id);
const el = {
  stage: $("stage"),
  table: $("table"),
  seats: $("seats"),
  center: $("center"),
  actions: $("actions"),
  empty: $("empty"),
  status: $("status"),
  drawer: $("drawer"),
  scrim: $("scrim"),
  overlay: $("overlay"),
  resultTitle: $("resultTitle"),
  resultBody: $("resultBody"),
  resultNext: $("resultNext"),
  resultPanel: $("resultPanel"),
  finalPanel: $("finalPanel"),
  finalBody: $("finalBody"),
  replayBar: $("replayBar"),
  replayLabel: $("replayLabel"),
  replayTools: $("replayTools"),
  viewpoint: $("viewpoint"),
  showAll: $("showAll"),
  recordList: $("recordList"),
};

const WINDS = ["東", "南", "西", "北"];
const HONOURS = ["east", "south", "west", "north", "white", "gd", "rd"];
const ACTION_LABEL = {
  ron: "ロン",
  tsumo: "ツモ",
  kan: "カン",
  pon: "ポン",
  chi: "チー",
  riichi: "リーチ",
  kyuushu: "九種九牌",
  pass: "パス",
};
const ACTION_ORDER = ["ron", "tsumo", "kan", "pon", "chi", "riichi", "kyuushu"];
const RESULT_TITLE = {
  tsumo: "ツモ",
  ron: "ロン",
  draw: "流局",
  abortive: "途中流局",
};
const ABORTIVE_REASON = {
  kyuushu: "九種九牌",
  four_winds: "四風連打",
  four_riichi: "四家立直",
  four_kans: "四槓散了",
  triple_ron: "三家和",
};

const S = {
  mode: "idle",
  gameId: null,
  replay: null,
  frames: [],
  index: 0,
  current: null,
  queue: [],
  timer: null,
  viewpoint: 0,
  showAll: true,
  lang: "ja",
  delay: 700,
  busy: false,
  recordId: null,
  lastGameRequest: null,
  autoOn: false,
  autoTimer: null,
};

// ------------------------------------------------------------------- network

async function call(method, url, body) {
  const init = { method, headers: {} };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch (_) {
      /* the body was not JSON; the status line will have to do */
    }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

const api = {
  agents: (env) => call("GET", `/api/agents?env=${encodeURIComponent(env)}`),
  createGame: (body) => call("POST", "/api/games", body),
  getGame: (id) => call("GET", `/api/games/${id}`),
  act: (id, action) => call("POST", `/api/games/${id}/act`, { action }),
  nextRound: (id) => call("POST", `/api/games/${id}/next`, {}),
  options: (id, body) => call("PATCH", `/api/games/${id}`, body),
  endGame: (id) => call("DELETE", `/api/games/${id}`),
  records: () => call("GET", "/api/records"),
  openReplay: (recordId) => call("POST", "/api/replays", { record_id: recordId }),
  frames: (id, from, to, viewpoint, showAll) =>
    call(
      "GET",
      `/api/replays/${id}/frames?from=${from}&to=${to}` +
        (viewpoint === null ? "" : `&viewpoint=${viewpoint}`) +
        `&showAll=${showAll ? "true" : "false"}`
    ),
};

// -------------------------------------------------------------------- pieces

function div(cls, text) {
  const node = document.createElement("div");
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function tileFile(tile) {
  if (tile === 34) return "5mr.svg";
  if (tile === 35) return "5pr.svg";
  if (tile === 36) return "5sr.svg";
  if (tile < 9) return `${tile + 1}m.svg`;
  if (tile < 18) return `${tile - 8}p.svg`;
  if (tile < 27) return `${tile - 17}s.svg`;
  return `${HONOURS[tile - 27]}.svg`;
}

function tileImg(tile, cls) {
  const img = document.createElement("img");
  img.className = cls ? `tile ${cls}` : "tile";
  const file = tile === null || tile === undefined ? "back.svg" : tileFile(tile);
  img.src = `/tiles/${S.lang}/${file}`;
  img.alt = "";
  img.draggable = false;
  return img;
}

function sideways(tile) {
  const slot = div("sideways");
  slot.append(tileImg(tile));
  return slot;
}

function roundLabel(round) {
  return `${WINDS[Math.floor(round.index / 4)] || "東"}${(round.index % 4) + 1}局`;
}

// ------------------------------------------------------------------ the board

function viewpointSeat(v) {
  if (S.mode === "replay") return S.viewpoint;
  const human = v.seats.findIndex((s) => s.kind === "human");
  return human < 0 ? 0 : human;
}

function fit() {
  const box = el.stage.getBoundingClientRect();
  const side = Math.max(260, Math.floor(Math.min(box.width, box.height)) - 8);
  el.table.style.setProperty("--t", `${side}px`);
  el.table.style.setProperty("--u", `${side / 100}px`);
}

function render(v, opts = {}) {
  S.current = v;
  el.table.hidden = false;
  el.empty.hidden = true;
  fit();
  const vp = viewpointSeat(v);
  renderSeats(v, vp);
  renderCenter(v, vp);
  renderActions(v);
  renderStatus(v);
  if (!v.result) clearCallouts();
  if (opts.overlay === false) el.overlay.hidden = true;
  else renderResult(v);
  if (S.mode === "replay") renderReplayBar(v);
}

/** A win is announced beside the player who made it, the way it is called out
 *  at a table, before any hand is turned over. */
function showCallouts(v, calls) {
  clearCallouts();
  const vp = viewpointSeat(v);
  for (const call of calls) {
    const node = div("callout", call.text);
    node.dataset.rel =
      call.seat === null || call.seat === undefined
        ? "center"
        : String((call.seat - vp + 4) % 4);
    el.table.append(node);
  }
}

function clearCallouts() {
  for (const node of Array.from(el.table.querySelectorAll(".callout"))) node.remove();
}

function renderSeats(v, vp) {
  el.seats.textContent = "";
  v.seats.forEach((seat, abs) => {
    const rel = (abs - vp + 4) % 4;
    const layer = div(rel === 0 ? "seat" : "seat other");
    layer.style.transform = `rotate(${-90 * rel}deg)`;
    const metrics = handMetrics(seat, rel === 0);
    layer.append(
      riverEl(v, seat, abs),
      nameplateEl(seat, metrics),
      handEl(v, seat, abs, rel === 0, metrics)
    );
    const melds = meldsEl(seat, abs, metrics);
    if (melds) layer.append(melds);
    el.seats.append(layer);
  });
}

/** Sits just above the left end of the hand, so it follows the hand's width
 *  instead of drifting off to the corner of the table. */
function nameplateEl(seat, metrics) {
  const box = div("nameplate");
  box.style.left = `${metrics.left}%`;
  box.append(div("badge", seat.kind === "human" ? "人" : "AI"));
  box.append(div("pname", seat.name));
  return box;
}

function riverEl(v, seat, abs) {
  const box = div("river");
  seat.river.forEach((d, i) => {
    const classes = ["cell"];
    if (d.tsumogiri) classes.push("tsumogiri");
    if (d.called) classes.push("taken");
    if (v.lastDiscard && v.lastDiscard.seat === abs && v.lastDiscard.index === i) {
      classes.push("last");
    }
    const cell = div(classes.join(" "));
    cell.append(d.riichi ? sideways(d.tile) : tileImg(d.tile));
    box.append(cell);
  });
  return box;
}

/** Lay a meld out the way it is set down: the called tile turned sideways, and
 *  placed on the side of the player it was taken from.
 *
 *  Returns one entry per *slot*, which is not the same as per tile: an added kan
 *  puts its fourth tile on top of the one already lying sideways, so three slots
 *  hold four tiles. */
function meldTiles(meld, ownerSeat) {
  const tiles = meld.tiles.map((t) => ({ tile: t, turned: false, hidden: false, stacked: false }));
  if (meld.kind === "kan_closed") {
    tiles[0].hidden = true;
    tiles[tiles.length - 1].hidden = true;
    return tiles;
  }
  if (meld.called === null || meld.called === undefined) return tiles;

  const rel = meld.from === null ? 2 : (meld.from - ownerSeat + 4) % 4;
  const position = rel === 3 ? 0 : rel === 2 ? 1 : 2;

  if (meld.kind === "kan_added") {
    // The pon stays as it was and the added tile rides on its sideways tile.
    const slots = [
      { tile: meld.tiles[0], turned: false, hidden: false, stacked: false },
      { tile: meld.tiles[1], turned: false, hidden: false, stacked: false },
    ];
    slots.splice(Math.min(position, slots.length), 0, {
      tile: meld.tiles[2],
      second: meld.tiles[3],
      turned: false,
      hidden: false,
      stacked: true,
    });
    return slots;
  }

  const called = tiles[meld.called];
  called.turned = true;
  if (meld.kind === "chi") {
    return [called, ...tiles.filter((_, i) => i !== meld.called)];
  }
  const rest = tiles.filter((_, i) => i !== meld.called);
  rest.splice(Math.min(position, rest.length), 0, called);
  return rest;
}

/** Melds are laid down at the edge of your side of the table: the first one
 *  called ends up furthest out, later ones stack back towards the hand. */
function meldsEl(seat, ownerSeat, metrics) {
  if (!seat.melds.length) return null;
  const box = div("melds");
  box.style.setProperty("--meld-scale", metrics.meldScale);
  for (const meld of [...seat.melds].reverse()) box.append(meldEl(meld, ownerSeat));
  return box;
}

function meldEl(meld, ownerSeat) {
  const box = div("meld");
  for (const piece of meldTiles(meld, ownerSeat)) {
    if (piece.stacked) {
      const slot = div("stacked");
      slot.append(tileImg(piece.second), tileImg(piece.tile));
      box.append(slot);
      continue;
    }
    const tile = piece.hidden ? null : piece.tile;
    box.append(piece.turned ? sideways(tile) : tileImg(tile));
  }
  return box;
}

function handEl(v, seat, abs, isSelf, metrics) {
  const anchor = div("hand-anchor");
  const row = div("hand-row");
  const canAct = isSelf && S.mode === "play" && v.prompt && !S.busy;
  const discardable = new Set(v.prompt ? v.prompt.discardable : []);
  const tsumogiri = v.prompt ? v.prompt.tsumogiri : null;

  const concealed = seat.hand ? seat.hand.slice() : null;
  let drawn = null;
  if (concealed && seat.drawn !== null && seat.drawn !== undefined) {
    const at = concealed.lastIndexOf(seat.drawn);
    if (at >= 0) {
      concealed.splice(at, 1);
      drawn = seat.drawn;
    }
  }

  const attach = (img, action) => {
    if (!canAct) return img;
    if (action === null) {
      img.classList.add("muted");
      return img;
    }
    img.classList.add("playable");
    img.addEventListener("click", () => sendAction(action, img));
    return img;
  };

  if (concealed) {
    for (const tile of concealed) {
      row.append(attach(tileImg(tile), discardable.has(tile) ? tile : null));
    }
    if (drawn !== null) {
      const action = tsumogiri !== null ? tsumogiri : discardable.has(drawn) ? drawn : null;
      row.append(attach(tileImg(drawn, "drawn"), action));
    }
  } else {
    // A hidden hand still has to show that its owner is holding a fresh tile.
    const total = seat.handCount;
    const separate = abs === v.current && total % 3 === 2;
    const backs = separate ? total - 1 : total;
    for (let i = 0; i < backs; i += 1) row.append(tileImg(null));
    if (separate) row.append(tileImg(null, "drawn"));
  }
  anchor.append(row);

  anchor.style.setProperty("--hand-scale", metrics.scale);
  anchor.style.right = `${metrics.right}%`;
  return anchor;
}

/** Where the hand and the melds sit on one side of the table.
 *
 *  The melds are pinned to the edge, and the hand is centred in whatever room
 *  is left of them. Space for the drawn tile is reserved whether or not one is
 *  held, so picking a tile up never nudges the tiles you were about to click.
 */
function handMetrics(seat, isSelf) {
  const ratio = 29 / 41.5;
  const tileW = (isSelf ? 7.2 : 5.6) * ratio;
  const meldTileW = 5.2 * ratio;
  const hasDrawn =
    Boolean(seat.hand) && seat.drawn !== null && seat.drawn !== undefined
      ? seat.hand.lastIndexOf(seat.drawn) >= 0
      : false;
  const concealed = (seat.hand ? seat.hand.length : seat.handCount) - (hasDrawn ? 1 : 0);

  let meldWidth = 0;
  if (seat.melds.length) {
    meldWidth = (seat.melds.length - 1) * 1.1;
    for (const meld of seat.melds) {
      for (const piece of meldTiles(meld, 0)) {
        meldWidth += piece.turned || piece.stacked ? 5.2 : meldTileW;
      }
    }
  }
  const meldScale = Math.min(1, 56 / Math.max(meldWidth, 1));
  const meldSpan = seat.melds.length ? meldWidth * meldScale + 5 : 0;

  const room = 100 - meldSpan;
  const width = concealed * tileW;
  const reserved = width + 1.7 + tileW; // the drawn tile hangs off the right
  const scale = Math.min(1, (room - 3) / Math.max(reserved, 1));
  return {
    scale,
    meldScale,
    right: meldSpan,
    left: (room - reserved * scale) / 2,
  };
}

function renderCenter(v, vp) {
  el.center.textContent = "";
  const info = div("info");
  info.append(div("round", roundLabel(v.round)));
  info.append(div("sticks", `${v.round.honba}本場 ・ 供託 ${v.round.kyotaku}`));
  info.append(div("sticks", `残り ${v.round.remaining}`));

  const dora = div("dora");
  for (const marker of v.round.dora) {
    dora.append(marker === null ? tileImg(null, "face-down") : tileImg(marker));
  }
  info.append(dora);
  el.center.append(info);

  v.seats.forEach((seat, abs) => {
    const rel = (abs - vp + 4) % 4;
    const layer = div("plate-layer");
    layer.style.transform = `rotate(${-90 * rel}deg)`;
    const windBadge = div(
      abs === v.dealer ? "seat-wind dealer" : "seat-wind",
      WINDS[seat.wind]
    );
    layer.append(windBadge);
    const plate = div("plate");
    if (abs === v.current) plate.classList.add("turn");
    plate.append(div("score", seat.score.toLocaleString()));
    if (seat.riichi !== "none") plate.append(div("stick"));
    layer.append(plate);
    el.center.append(layer);
  });
}

// ------------------------------------------------------------------- actions

function renderActions(v) {
  el.actions.textContent = "";
  if (S.mode !== "play" || !v.prompt || S.busy) return;
  const options = v.prompt.options;
  const byKind = new Map();
  for (const option of options) {
    if (!byKind.has(option.kind)) byKind.set(option.kind, []);
    byKind.get(option.kind).push(option);
  }
  for (const kind of ACTION_ORDER) {
    const list = byKind.get(kind);
    if (!list) continue;
    el.actions.append(
      list.length === 1 ? simpleButton(kind, list[0]) : choiceButton(kind, list)
    );
  }
  const pass = byKind.get("pass");
  if (pass) el.actions.append(simpleButton("pass", pass[0]));
}

function buttonFor(kind, label) {
  const button = document.createElement("button");
  button.textContent = label;
  if (kind === "ron" || kind === "tsumo") button.classList.add("win");
  if (kind === "pass") button.classList.add("pass");
  return button;
}

function simpleButton(kind, option) {
  const button = buttonFor(kind, option.label || ACTION_LABEL[kind] || kind);
  button.addEventListener("click", () => sendAction(option.action, button));
  return button;
}

/** Several ways to make the same call: show the tile groups and let the player
 *  pick the one they meant. */
function choiceButton(kind, list) {
  const wrap = div("choices");
  const button = buttonFor(kind, ACTION_LABEL[kind] || kind);
  wrap.append(button);
  let open = false;
  button.addEventListener("click", () => {
    open = !open;
    for (const node of Array.from(wrap.querySelectorAll(".choice"))) node.remove();
    if (!open) return;
    for (const option of list) {
      const choice = div("choice");
      choice.setAttribute("role", "button");
      choice.tabIndex = 0;
      for (const tile of option.tiles || []) choice.append(tileImg(tile));
      const pick = () => sendAction(option.action, choice);
      choice.addEventListener("click", pick);
      choice.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          pick();
        }
      });
      wrap.append(choice);
    }
  });
  return wrap;
}

function renderStatus(v) {
  if (S.mode === "replay") {
    el.status.textContent = "";
    return;
  }
  if (v.gameOver) el.status.textContent = "終局";
  else if (v.result) el.status.textContent = "";
  else if (v.prompt) el.status.textContent = v.prompt.kind === "claim" ? "鳴きますか" : "あなたの番";
  else el.status.textContent = "";
}

// ------------------------------------------------------------------ playback

function play(frames) {
  if (!frames || !frames.length) return;
  S.queue.push(...frames);
  if (S.timer === null) pump();
}

function pump() {
  const frame = S.queue.shift();
  if (!frame) {
    S.timer = null;
    return;
  }
  if (frame.result && S.mode === "play") {
    // A result is always the last frame, and it gets its own little sequence.
    S.queue.length = 0;
    S.timer = null;
    presentResult(frame);
    return;
  }
  render(frame);
  if (!S.queue.length) {
    S.timer = null;
    return;
  }
  S.timer = window.setTimeout(pump, S.delay);
}

function flush() {
  if (!S.queue.length) return;
  if (S.timer !== null) window.clearTimeout(S.timer);
  S.timer = null;
  const last = S.queue[S.queue.length - 1];
  S.queue.length = 0;
  render(last);
}

function wait(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

/** What gets announced before the hands turn over. A win belongs to the player
 *  who made it; an abortive draw ends the round for everyone, so it is called
 *  in the middle of the table whatever brought it about. Nine terminals still
 *  turns the declarer's hand over -- that is what names them. */
function calloutsFor(r) {
  if (r.type === "tsumo" || r.type === "ron") {
    return r.winners.map((w) => ({
      seat: w.seat,
      text: r.type === "tsumo" ? "ツモ" : "ロン",
    }));
  }
  if (r.type === "abortive") {
    return [{ seat: null, text: ABORTIVE_REASON[r.reason] || "途中流局" }];
  }
  return [];
}

/** Call it, turn the hands over, then offer the next round -- in that order,
 *  so the player sees what happened before a dialog covers the table. */
async function presentResult(frame) {
  const beat = Math.max(350, S.delay);
  const r = frame.result;
  const calls = calloutsFor(r);
  if (calls.length) {
    showCallouts(S.current || frame, calls);
    await wait(beat);
  }
  render(frame, { overlay: false });
  await wait(beat + 400);
  clearCallouts();
  renderResult(frame);
}

async function sendAction(action, node) {
  if (S.busy || !S.gameId) return;
  S.busy = true;
  if (node) node.classList.add("pending");
  el.actions.textContent = "";
  try {
    const data = await api.act(S.gameId, action);
    S.busy = false;
    if (data.recordId) S.recordId = data.recordId;
    play(data.frames);
  } catch (err) {
    S.busy = false;
    el.status.textContent = err.message;
    if (S.current) render(S.current);
  }
}

// -------------------------------------------------------------------- result

function renderResult(v) {
  if (!v.result) {
    el.overlay.hidden = true;
    return;
  }
  el.resultPanel.hidden = false;
  el.finalPanel.hidden = true;
  const r = v.result;
  let title = RESULT_TITLE[r.type] || r.type;
  if (r.type === "abortive" && r.reason) title = ABORTIVE_REASON[r.reason] || title;
  if (r.gameOver) title += " ・ 終局";
  el.resultTitle.textContent = title;

  const body = document.createElement("div");
  body.append(standingsTable(v, r));
  for (const winner of r.winners) body.append(winnerBlock(v, winner));
  if (r.type === "draw" && r.tenpai) {
    const line = r.tenpai
      .map((t, i) => `${v.seats[i].name}: ${t ? "聴牌" : "不聴"}`)
      .join(" ・ ");
    body.append(div("meta", line));
  }
  body.append(div("meta", `${r.round.honba}本場 ・ 供託 ${r.round.kyotaku}`));
  el.resultBody.textContent = "";
  el.resultBody.append(body);

  el.resultNext.textContent = r.gameOver ? "終局へ" : "次の局へ";
  el.overlay.hidden = false;
}

function standingsTable(v, r) {
  const table = document.createElement("table");
  const head = document.createElement("tr");
  for (const label of ["", "点数", "増減"]) {
    const th = document.createElement("th");
    th.textContent = label;
    head.append(th);
  }
  table.append(head);
  const order = r.final
    ? r.final.map((entry) => entry.seat)
    : v.seats.map((_, i) => i);
  for (const seat of order) {
    const row = document.createElement("tr");
    const who = document.createElement("td");
    who.textContent = `${WINDS[v.seats[seat].wind]} ${v.seats[seat].name}`;
    const score = document.createElement("td");
    score.textContent = r.scores[seat].toLocaleString();
    const delta = document.createElement("td");
    const value = r.deltas[seat];
    delta.textContent = (value > 0 ? "+" : "") + value.toLocaleString();
    delta.className = value > 0 ? "plus" : value < 0 ? "minus" : "";
    row.append(who, score, delta);
    table.append(row);
  }
  return table;
}

function winnerBlock(v, w) {
  const box = div("winner");
  const from = w.from === null ? "ツモ" : `${v.seats[w.from].name} から`;
  box.append(div(null, `${v.seats[w.seat].name} ・ ${from} ・ ${w.points.toLocaleString()}点`));

  const tiles = div("tiles");
  for (const tile of w.hand) tiles.append(tileImg(tile));
  if (w.winningTile !== null && w.winningTile !== undefined) {
    tiles.append(tileImg(w.winningTile, "winning-tile"));
  }
  // Melds are drawn by the same code as on the table, so a concealed kan shows
  // its two face-down tiles and an added kan its stacked fourth here too.
  for (const meld of w.melds) tiles.append(meldEl(meld, w.seat));
  // The indicators ride on the same row: a hand with many yaku is tall enough
  // already without giving them a line of their own.
  if (w.dora.length || w.uraDora.length) {
    const markers = div("markers");
    if (w.dora.length) {
      markers.append(div("marker-label", "ドラ"));
      for (const tile of w.dora) markers.append(tileImg(tile));
    }
    if (w.uraDora.length) {
      markers.append(div("marker-label", "裏"));
      for (const tile of w.uraDora) markers.append(tileImg(tile));
    }
    tiles.append(markers);
  }
  box.append(tiles);

  const yaku = div("yaku");
  for (const entry of w.yaku) {
    yaku.append(div(null, entry.name), div("han", `${entry.han}翻`));
  }
  const extras = [
    ["ドラ", w.doraHan],
    ["赤ドラ", w.akaHan],
    ["裏ドラ", w.uraHan],
  ];
  for (const [name, han] of extras) {
    if (han) yaku.append(div(null, name), div("han", `${han}翻`));
  }
  box.append(yaku);
  box.append(
    div("total", w.yakuman ? `役満 ${w.yakuman}倍` : `${w.han}翻 ${w.fu}符`)
  );

  return box;
}

/** The standings, once the last hand has been read. Points and rank bonus are
 *  shown side by side rather than added: that is how they are read. */
function showFinal(v) {
  const r = v.result;
  const rows = [...(r.final || [])].sort((a, b) => a.rank - b.rank);
  const hasUma = rows.some((entry) => entry.uma !== 0);

  const table = document.createElement("table");
  const head = document.createElement("tr");
  for (const label of hasUma ? ["順位", "", "点数", "順位点"] : ["順位", "", "点数"]) {
    const th = document.createElement("th");
    th.textContent = label;
    head.append(th);
  }
  table.append(head);
  for (const entry of rows) {
    const row = document.createElement("tr");
    const cells = [`${entry.rank}位`, v.seats[entry.seat].name, entry.score.toLocaleString()];
    if (hasUma) cells.push((entry.uma > 0 ? "+" : "") + entry.uma);
    cells.forEach((text, col) => {
      const td = document.createElement("td");
      td.textContent = text;
      if (hasUma && col === 3) {
        td.className = entry.uma > 0 ? "plus" : entry.uma < 0 ? "minus" : "";
      }
      row.append(td);
    });
    table.append(row);
  }

  el.finalBody.textContent = "";
  el.finalBody.append(table);
  const replayBtn = $("finalReplay");
  replayBtn.disabled = !S.recordId;
  replayBtn.title = S.recordId ? "" : "この対局は保存されていません";
  el.resultPanel.hidden = true;
  el.finalPanel.hidden = false;
  el.overlay.hidden = false;
}

$("finalQuit").addEventListener("click", async () => {
  el.overlay.hidden = true;
  if (S.gameId) await api.endGame(S.gameId).catch(() => {});
  S.gameId = null;
  S.mode = "idle";
  S.queue.length = 0;
  remember(null);
  el.table.hidden = true;
  el.empty.hidden = false;
  refreshRecords();
});

$("finalAgain").addEventListener("click", async () => {
  const body = { ...(S.lastGameRequest || {}) };
  delete body.seed; // a rematch should deal a new game, not repeat this one
  el.overlay.hidden = true;
  if (S.gameId) await api.endGame(S.gameId).catch(() => {});
  await startGame(body);
});

$("finalReplay").addEventListener("click", async () => {
  if (!S.recordId) {
    el.status.textContent = "この対局は保存されていません";
    return;
  }
  el.overlay.hidden = true;
  if (S.gameId) await api.endGame(S.gameId).catch(() => {});
  S.gameId = null;
  remember(null);
  await openReplay({ id: S.recordId });
});

el.resultNext.addEventListener("click", async () => {
  if (S.mode === "replay") {
    el.overlay.hidden = true;
    if (S.index < S.replay.total - 1) await goto(S.index + 1);
    return;
  }
  if (S.current && S.current.gameOver) {
    showFinal(S.current);
    return;
  }
  el.overlay.hidden = true;
  try {
    const data = await api.nextRound(S.gameId);
    if (data.recordId) S.recordId = data.recordId;
    play(data.frames);
  } catch (err) {
    el.status.textContent = err.message;
  }
});

// --------------------------------------------------------------------- replay

async function ensureLoaded(index) {
  if (S.frames[index]) return;
  const span = S.replay.rounds.find((r) => index >= r.start && index <= r.end) || {
    start: index,
    end: index,
  };
  const data = await api.frames(
    S.replay.replayId,
    span.start,
    span.end + 1,
    S.viewpoint,
    S.showAll
  );
  data.frames.forEach((frame, offset) => {
    S.frames[span.start + offset] = frame;
  });
}

async function goto(index) {
  if (!S.replay) return;
  const clamped = Math.max(0, Math.min(S.replay.total - 1, index));
  await ensureLoaded(clamped);
  S.index = clamped;
  render(S.frames[clamped]);
}

function roundOf(index) {
  return (
    S.replay.rounds.find((r) => index >= r.start && index <= r.end) || S.replay.rounds[0]
  );
}

function renderReplayBar(v) {
  const span = roundOf(S.index);
  const within = S.index - span.start + 1;
  const total = span.end - span.start + 1;
  const event = v.step && v.step.event ? describeEvent(v, v.step.event) : "配牌";
  el.replayLabel.textContent = `${span.label}  ${within}/${total}  ${event}`;
}

function describeEvent(v, event) {
  const who = v.seats[event.seat] ? `${WINDS[v.seats[event.seat].wind]}家` : "";
  const tiles = (event.tiles || []).length ? ` ${event.tiles.map(tileName).join("")}` : "";
  return `${who} ${event.kind}${tiles}`.trim();
}

function tileName(tile) {
  if (tile >= 34) return ["赤5m", "赤5p", "赤5s"][tile - 34];
  if (tile < 9) return `${tile + 1}m`;
  if (tile < 18) return `${tile - 8}p`;
  if (tile < 27) return `${tile - 17}s`;
  return ["東", "南", "西", "北", "白", "發", "中"][tile - 27];
}

/** Auto-play walks the record one step at a time at the same pace a live game
 *  is drawn, and stops on a round result so the overlay can be read. */
function updateAutoButton() {
  const button = $("replayAuto");
  button.textContent = S.autoOn ? "停止" : "再生";
  button.classList.toggle("playing", S.autoOn);
}

function stopAuto() {
  S.autoOn = false;
  if (S.autoTimer !== null) {
    window.clearTimeout(S.autoTimer);
    S.autoTimer = null;
  }
  updateAutoButton();
}

function scheduleAuto() {
  S.autoTimer = window.setTimeout(autoTick, Math.max(150, S.delay));
}

async function autoTick() {
  S.autoTimer = null;
  if (!S.autoOn || !S.replay) return;
  if (S.index >= S.replay.total - 1) {
    stopAuto();
    return;
  }
  await goto(S.index + 1);
  if (!S.autoOn) return; // stopped while the frame was loading
  const frame = S.frames[S.index];
  if (frame && frame.result) {
    stopAuto();
    return;
  }
  scheduleAuto();
}

$("replayAuto").addEventListener("click", () => {
  if (!S.replay) return;
  if (S.autoOn) {
    stopAuto();
    return;
  }
  S.autoOn = true;
  updateAutoButton();
  scheduleAuto();
});

el.replayBar.addEventListener("click", async (e) => {
  const button = e.target.closest("[data-jump]");
  if (!button || !S.replay) return;
  stopAuto(); // a jump means the viewer has taken over
  const span = roundOf(S.index);
  const at = S.replay.rounds.indexOf(span);
  switch (button.dataset.jump) {
    case "first":
      await goto(0);
      break;
    case "prev":
      await goto(S.index - 1);
      break;
    case "next":
      await goto(S.index + 1);
      break;
    case "last":
      await goto(S.replay.total - 1);
      break;
    case "prevRound":
      await goto(S.index === span.start && at > 0 ? S.replay.rounds[at - 1].start : span.start);
      break;
    case "nextRound":
      await goto(at + 1 < S.replay.rounds.length ? S.replay.rounds[at + 1].start : S.replay.total - 1);
      break;
    default:
      break;
  }
});

document.addEventListener("keydown", async (e) => {
  if (S.mode !== "replay") return;
  if (e.key === "ArrowRight" || e.key === "ArrowLeft") stopAuto();
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
  const span = roundOf(S.index);
  const at = S.replay.rounds.indexOf(span);
  if (e.key === "ArrowRight") {
    e.preventDefault();
    if (e.shiftKey) {
      await goto(at + 1 < S.replay.rounds.length ? S.replay.rounds[at + 1].start : S.replay.total - 1);
    } else {
      await goto(S.index + 1);
    }
  } else if (e.key === "ArrowLeft") {
    e.preventDefault();
    if (e.shiftKey) {
      await goto(S.index === span.start && at > 0 ? S.replay.rounds[at - 1].start : span.start);
    } else {
      await goto(S.index - 1);
    }
  }
});

el.viewpoint.addEventListener("change", async () => {
  S.viewpoint = Number(el.viewpoint.value);
  S.frames = [];
  await goto(S.index);
});

el.showAll.addEventListener("change", async () => {
  S.showAll = el.showAll.value === "1";
  S.frames = [];
  await goto(S.index);
});

// --------------------------------------------------------------------- drawer

function openDrawer(open) {
  el.drawer.hidden = !open;
  el.scrim.hidden = !open;
  $("menuBtn").setAttribute("aria-expanded", String(open));
  if (open) refreshRecords();
}

$("menuBtn").addEventListener("click", () => openDrawer(el.drawer.hidden));
el.scrim.addEventListener("click", () => openDrawer(false));
$("emptyStart").addEventListener("click", () => openDrawer(true));

async function refreshAgents() {
  const env = $("cfgEnv").value;
  try {
    const agents = await api.agents(env);
    const select = $("cfgAgent");
    select.textContent = "";
    for (const agent of agents) {
      const option = document.createElement("option");
      option.value = agent.id;
      option.textContent = agent.name;
      select.append(option);
    }
  } catch (err) {
    el.status.textContent = err.message;
  }
}

$("cfgEnv").addEventListener("change", refreshAgents);

async function refreshRecords() {
  try {
    const records = await api.records();
    el.recordList.textContent = "";
    if (!records.length) {
      el.recordList.append(div("hint", "保存された牌譜はありません。"));
      return;
    }
    for (const record of records) {
      const button = document.createElement("button");
      button.className = "record";
      const when = new Date(record.createdAt).toLocaleString();
      const mode = { half: "半荘", east: "東風", single: "一局" }[record.roundMode] || record.roundMode;
      const rule = record.env === "red_mahjong" ? "赤あり" : "赤なし";
      button.append(div("when", `${when} ・ ${rule} ${mode}${record.complete ? "" : " ・ 中断"}`));
      button.append(div("who", record.players.map((p) => p.name).join(" / ")));
      button.addEventListener("click", () => openReplay(record));
      el.recordList.append(button);
    }
  } catch (err) {
    el.recordList.textContent = err.message;
  }
}

$("refreshRecords").addEventListener("click", refreshRecords);

// ----------------------------------------------------------------- lifecycle

function collectGameRequest() {
  const seatValue = $("cfgSeat").value;
  const seed = $("cfgSeed").value;
  const body = {
    env_id: $("cfgEnv").value,
    round_mode: $("cfgMode").value,
    agent_id: $("cfgAgent").value || null,
    human_seat: seatValue === "none" ? null : seatValue === "random" ? 0 : Number(seatValue),
    random_seat: seatValue === "random",
    human_name: $("cfgName").value || "You",
    hide_hands: $("optHide").checked,
    no_calls: $("optNoCalls").checked,
    save_record: $("cfgSave").checked,
  };
  if (seed !== "") body.seed = Number(seed);
  return body;
}

async function startGame(body) {
  stopAuto();
  S.lastGameRequest = body;
  S.recordId = null;
  clearCallouts();
  el.status.textContent = "準備中…";
  try {
    const data = await api.createGame(body);
    if (data.gameId === null) {
      el.status.textContent = "";
      if (data.recordId) await openReplay({ id: data.recordId });
      return;
    }
    S.mode = "play";
    S.gameId = data.gameId;
    remember(data.gameId);
    S.replay = null;
    S.frames = [];
    S.queue.length = 0;
    el.replayBar.hidden = true;
    el.replayTools.hidden = true;
    el.status.textContent = "";
    play(data.frames);
  } catch (err) {
    el.status.textContent = err.message;
  }
}

$("startBtn").addEventListener("click", async () => {
  openDrawer(false);
  await startGame(collectGameRequest());
});

$("endBtn").addEventListener("click", async () => {
  if (!S.gameId) return;
  await api.endGame(S.gameId).catch(() => {});
  S.gameId = null;
  remember(null);
  S.mode = "idle";
  S.queue.length = 0;
  el.table.hidden = true;
  el.empty.hidden = false;
  el.overlay.hidden = true;
  openDrawer(false);
  refreshRecords();
});

async function openReplay(record) {
  openDrawer(false);
  stopAuto();
  el.status.textContent = "牌譜を読み込み中…";
  try {
    const replay = await api.openReplay(record.id);
    S.mode = "replay";
    S.replay = replay;
    S.frames = [];
    S.gameId = null;
    S.queue.length = 0;
    S.viewpoint = replay.humanSeat === null || replay.humanSeat === undefined ? 0 : replay.humanSeat;
    S.showAll = true;
    el.viewpoint.textContent = "";
    replay.seats.forEach((seat, i) => {
      const option = document.createElement("option");
      option.value = String(i);
      option.textContent = `${i + 1}. ${seat.name}`;
      el.viewpoint.append(option);
    });
    el.viewpoint.value = String(S.viewpoint);
    el.showAll.value = "1";
    updateAutoButton();
    el.replayTools.hidden = false;
    el.replayBar.hidden = false;
    el.status.textContent = "";
    await goto(0);
  } catch (err) {
    el.status.textContent = err.message;
  }
}

// ------------------------------------------------------------------ settings

$("optDelay").addEventListener("input", (e) => {
  S.delay = Number(e.target.value);
  $("optDelayValue").textContent = `${S.delay}ms`;
});

$("optLang").addEventListener("change", (e) => {
  S.lang = e.target.value;
  if (S.current) render(S.current);
});

for (const [id, key] of [["optHide", "hide_hands"], ["optNoCalls", "no_calls"]]) {
  $(id).addEventListener("change", async (e) => {
    if (!S.gameId) return;
    try {
      const data = await api.options(S.gameId, { [key]: e.target.checked });
      if (!S.queue.length) render(data.frames[0]);
    } catch (err) {
      el.status.textContent = err.message;
    }
  });
}

el.table.addEventListener("click", (e) => {
  if (e.target.closest(".playable, .choice, button")) return;
  flush();
});

window.addEventListener("resize", () => {
  if (S.current) fit();
});

/** Reloading the page should land you back in the game you were playing. */
function remember(gameId) {
  try {
    if (gameId) window.localStorage.setItem("mahjax.game", gameId);
    else window.localStorage.removeItem("mahjax.game");
  } catch (_) {
    /* private windows and blocked site data are fine; the link still works */
  }
}

function remembered() {
  try {
    return window.localStorage.getItem("mahjax.game");
  } catch (_) {
    return null;
  }
}

async function restore() {
  const params = new URLSearchParams(window.location.search);
  const replayId = params.get("replay");
  if (replayId) {
    await openReplay({ id: replayId });
    const at = Number(params.get("frame"));
    if (Number.isInteger(at) && at > 0) await goto(at);
    return;
  }
  const gameId = params.get("game") || remembered();
  if (!gameId) return;
  let data;
  try {
    data = await api.getGame(gameId);
  } catch (_) {
    remember(null); // the game is gone; fall back to the start screen
    return;
  }
  // Deliberately outside the catch: a failure to draw is a bug, not a missing
  // game, and swallowing it here would look like "nothing was saved".
  S.mode = "play";
  S.gameId = gameId;
  S.recordId = data.recordId || null;
  remember(gameId);
  render(data.frames[0]);
}

refreshAgents();
refreshRecords();
restore();
