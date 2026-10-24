// Renders the Altair-generated Vega-Lite specs in site/charts/ and wires up
// the interactive pieces (word trends, speech reader, table views, theme).

const cache = new Map();
const loadJSON = (url) => {
  if (!cache.has(url)) cache.set(url, fetch(url).then((r) => {
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return r.json();
  }));
  return cache.get(url);
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children.filter((c) => c != null));
  return node;
};
const fmtInt = new Intl.NumberFormat("en-US");
const fmtDate = (iso) => new Date(`${iso}T12:00:00`).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
const pct = (x) => `${(100 * x).toFixed(0)}%`;
const signed = (x) => {
  const r = Math.round(x * 100) / 100;
  return (r > 0 ? "+" : r < 0 ? "−" : "") + Math.abs(r).toFixed(2);
};

// ---------------------------------------------------------------- theme
const storage = {
  get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};
const prefersDark = matchMedia("(prefers-color-scheme: dark)");
const isDark = () => {
  const t = document.documentElement.dataset.theme;
  return t ? t === "dark" : prefersDark.matches;
};
const saved = storage.get("theme");
if (saved) document.documentElement.dataset.theme = saved;

$(".theme-toggle").addEventListener("click", () => {
  const next = isDark() ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  storage.set("theme", next);
  rerenderAll();
});
prefersDark.addEventListener("change", () => { if (!document.documentElement.dataset.theme) rerenderAll(); });

// ---------------------------------------------------------------- charts
const rendered = []; // {container, name, options}
let palette = {};

function themed(spec) {
  if (!isDark()) return spec;
  const json = JSON.stringify(spec).replace(/#[0-9a-f]{6}\b/gi, (hex) => palette[hex.toLowerCase()] ?? hex);
  return JSON.parse(json);
}

async function render(container, name, { datasets, bind, onClick, adjust } = {}) {
  let entry = rendered.find((r) => r.container === container);
  if (!entry) {
    entry = { container, name };
    rendered.push(entry);
  }
  Object.assign(entry, { datasets, bind, onClick, adjust });
  const spec = structuredClone(await loadJSON(`charts/${name}.json`));
  if (datasets) spec.datasets = { ...(spec.datasets || {}), ...datasets };
  if (adjust) adjust(spec, container);
  if (container.clientWidth < 480) {
    // Let legends wrap instead of running off the edge on phones.
    spec.config = { ...spec.config, legend: { ...spec.config?.legend, columns: 2, symbolLimit: 0 } };
  }
  entry.view?.finalize();
  container.classList.add("loading");
  if (bind) bind.replaceChildren();
  const result = await vegaEmbed(container, themed(spec), {
    actions: false,
    renderer: "svg",
    bind: bind || undefined,
    tooltip: { theme: isDark() ? "dark" : "light" },
  });
  container.classList.remove("loading");
  entry.view = result.view;
  if (onClick) {
    result.view.addEventListener("click", (_, item) => item?.datum?.id && onClick(item.datum.id));
  }
  return result.view;
}

function rerenderAll() {
  for (const r of rendered) render(r.container, r.name, r);
}

// Emotions is a 3-column facet; size its panels to the card width.
function fitFacet(spec, container) {
  const width = container.clientWidth || 600;
  const columns = width < 520 ? 2 : 3;
  spec.columns = columns;
  spec.spec.width = Math.max(110, Math.floor(width / columns) - 56);
}

// ---------------------------------------------------------------- table views
function addTableView(figure) {
  const [file, colList] = figure.dataset.table.split(":");
  const cols = colList.split(",");
  const details = el("details", { className: "table-view" }, el("summary", { textContent: "Show data table" }));
  figure.append(details);
  details.addEventListener("toggle", async () => {
    if (!details.open || details.querySelector("table")) return;
    const rows = await loadJSON(`data/${file}.json`);
    const head = el("tr", {}, ...cols.map((c) => el("th", { textContent: c.replaceAll("_", " ") })));
    const body = rows.map((row) => el("tr", {}, ...cols.map((c) => {
      const v = row[c];
      return el("td", { textContent: typeof v === "number" && !Number.isInteger(v) ? v.toFixed(3) : v ?? "" });
    })));
    details.append(el("div", { className: "table-scroll" }, el("table", {}, el("thead", {}, head), el("tbody", {}, ...body))));
  }, { once: false });
}

// ---------------------------------------------------------------- summary
async function fillSummary() {
  const [s, speeches] = await Promise.all([loadJSON("data/summary.json"), loadJSON("data/speeches.json")]);
  const set = (key, value) => $$(`[data-summary="${key}"]`).forEach((n) => { n.textContent = value; });
  set("speeches", fmtInt.format(s.speeches));
  set("presidents", fmtInt.format(s.presidents));
  set("words", `${(s.words / 1e6).toFixed(1)}M`);
  set("last", fmtDate(s.last));
  set("last-year", s.last.slice(0, 4));
  set("topic-count", s.topics.length);
  set("sentences", fmtInt.format(speeches.reduce((n, d) => n + d.sentences, 0)));
  const latest = $('[data-summary="latest"]');
  latest.replaceChildren(
    el("a", { href: `#speech=${s.latest.id}`, textContent: s.latest.date.slice(0, 4) }),
    el("small", { textContent: s.latest.president }),
  );
}

// ---------------------------------------------------------------- word trends
const MAX_TERMS = 5;

async function plotTrend(raw) {
  const index = await loadJSON("data/word_index.json");
  const note = $(".trend-note");
  const terms = [...new Set(raw.toLowerCase().split(",").map((t) => t.trim().replace(/\s+/g, " ")).filter(Boolean))].slice(0, MAX_TERMS);
  const missing = terms.filter((t) => !index.terms[t]);
  const found = terms.filter((t) => index.terms[t]);

  // Words per calendar year, so each year's rate is counts / words spoken that year.
  const years = [...new Set(index.when.map(Math.floor))].sort((a, b) => a - b);
  const wordsByYear = new Map();
  index.when.forEach((w, i) => {
    const y = Math.floor(w);
    wordsByYear.set(y, (wordsByYear.get(y) || 0) + index.words[i]);
  });

  const rows = [];
  for (const term of found) {
    const counts = new Map();
    for (const pair of index.terms[term].split(" ")) {
      const [i, n] = pair.split(":").map(Number);
      const y = Math.floor(index.when[i]);
      counts.set(y, (counts.get(y) || 0) + n);
    }
    // 5-year centered average of the per-year rate.
    years.forEach((y) => {
      let c = 0, w = 0;
      for (let d = -2; d <= 2; d++) {
        if (wordsByYear.has(y + d)) { c += counts.get(y + d) || 0; w += wordsByYear.get(y + d); }
      }
      rows.push({ term, when: y, rate: (1e6 * c) / w });
    });
  }

  const msgs = [];
  if (missing.length) msgs.push(`Not in the index: ${missing.join(", ")} (terms must appear at least 8 times across 3+ messages).`);
  if (raw.split(",").filter((t) => t.trim()).length > MAX_TERMS) msgs.push(`Showing the first ${MAX_TERMS} terms.`);
  note.textContent = msgs.join(" ");
  if (!found.length) return;
  await render($("#trend-chart"), "trend", { datasets: { trend: rows } });
}

function initTrends() {
  const input = $("#trend-input");
  $(".trend-form").addEventListener("submit", (e) => { e.preventDefault(); plotTrend(input.value); });
  $$(".presets button").forEach((b) => b.addEventListener("click", () => {
    input.value = b.dataset.terms;
    plotTrend(input.value);
  }));
  plotTrend(input.value);
}

// ---------------------------------------------------------------- speech reader
let speechesById = new Map();

async function initExplorer() {
  const speeches = await loadJSON("data/speeches.json");
  speechesById = new Map(speeches.map((s) => [s.id, s]));
  const select = $("#speech-select");
  const groups = new Map();
  [...speeches].reverse().forEach((s) => {
    if (!groups.has(s.president)) groups.set(s.president, el("optgroup", { label: s.president }));
    groups.get(s.president).append(el("option", { value: s.id, textContent: `${fmtDate(s.date)}${s.formal_sotu ? "" : " (joint session)"}` }));
  });
  select.append(...groups.values());
  select.addEventListener("change", () => openSpeech(select.value, false));
  $("#tone-toggle").addEventListener("change", (e) => {
    $("#sp-text").classList.toggle("toned", e.target.checked);
    $(".tone-legend").hidden = !e.target.checked;
  });

  const fromHash = location.hash.match(/^#speech=(.+)$/)?.[1];
  openSpeech(speechesById.has(fromHash) ? fromHash : speeches.at(-1).id, Boolean(fromHash));
}

window.addEventListener("hashchange", () => {
  const id = location.hash.match(/^#speech=(.+)$/)?.[1];
  if (id && speechesById.has(id)) openSpeech(id, true);
});

async function openSpeech(id, scroll = true) {
  const s = speechesById.get(id);
  if (!s) return;
  $("#speech-select").value = id;
  if (location.hash !== `#speech=${id}`) history.replaceState(null, "", `#speech=${id}`);
  if (scroll) $("#explore").scrollIntoView();

  $("#sp-meta").textContent = `${s.president} · ${s.party} · ${fmtDate(s.date)}`;
  $("#sp-title").textContent = s.title;
  $("#sp-source").replaceChildren(
    `${s.delivery === "spoken" ? "Delivered in person" : "Sent as a written message"}${s.formal_sotu ? "" : " · technically not a State of the Union"} · `,
    el("a", { href: s.url, textContent: "Source: American Presidency Project", target: "_blank", rel: "noopener" }),
  );
  const tile = (label, value, sub) => el("div", { className: "tile" }, el("dt", { textContent: label }), el("dd", {}, value, sub ? el("small", { textContent: sub }) : null));
  $("#sp-tiles").replaceChildren(
    tile("Words", fmtInt.format(s.words)),
    tile("Grade level", s.grade_level.toFixed(1), `${s.words_per_sentence.toFixed(0)} words / sentence`),
    tile("Net tone", signed(s.tone), `${pct(s.positive_share)} upbeat · ${pct(s.negative_share)} downbeat`),
    tile("Lexical diversity", s.lexical_diversity.toFixed(0), "MTLD"),
  );

  const emotions = ["joy", "fear", "anger", "sadness", "surprise", "disgust"]
    .map((e) => [e, s[`emotion_${e}`]]).sort((a, b) => b[1] - a[1]).slice(0, 4);
  $("#sp-emotions").replaceChildren(...emotions.map(([e, v]) => el("li", {}, el("b", { textContent: e }), ` ${pct(v)} of sentences`)));
  $("#sp-places").replaceChildren(...(s.places.length ? s.places : ["none detected"]).map((p) => el("li", { textContent: p })));
  const speechLinks = (ids) => ids.map((sid) => {
    const o = speechesById.get(sid);
    return el("li", {}, el("a", { href: `#speech=${sid}`, textContent: `${o.president.split(" ").at(-1)} ${o.date.slice(0, 4)}` }));
  });
  $("#sp-similar").replaceChildren(...speechLinks(s.similar));
  $("#sp-kindred").replaceChildren(...speechLinks(s.kindred));

  const arcs = await loadJSON("data/arcs.json");
  const arc = arcs[id].map((tone, i, a) => ({ position: i / (a.length - 1), tone }));
  const topics = Object.entries(s.topics).filter(([t]) => t !== "Other").slice(0, 8).map(([topic, share]) => ({ topic, share }));
  render($("#arc-chart"), "arc", { datasets: { arc } });
  render($("#speech-topics-chart"), "speech_topics", { datasets: { topics } });

  const reader = $("#sp-text");
  reader.replaceChildren(el("p", { textContent: "Loading…" }));
  const paragraphs = await loadJSON(`data/text/${id}.json`);
  if ($("#speech-select").value !== id) return; // a newer selection won
  reader.replaceChildren(...paragraphs.map((sentences) => el("p", {}, ...sentences.flatMap(([text, t]) => {
    const span = el("span", { textContent: text });
    if (Math.abs(t) >= 0.25) {
      span.dataset.t = t > 0 ? "p" : "n";
      span.style.background = `rgba(var(--${t > 0 ? "pos" : "neg"}), ${(Math.abs(t) * 0.4).toFixed(2)})`;
    }
    return [span, " "];
  }))));
  reader.scrollTop = 0;
}

// Background only shows while highlighting is on.
const style = document.createElement("style");
style.textContent = ".reader:not(.toned) span[data-t] { background: none !important; }";
document.head.append(style);

// ---------------------------------------------------------------- boot
async function main() {
  palette = await loadJSON("charts/palette.json");
  fillSummary();
  const clickable = new Set(["length", "grade", "diversity", "tone", "map"]);
  for (const figure of $$("figure[data-chart]")) {
    const name = figure.dataset.chart;
    render($(".chart", figure), name, {
      bind: figure.hasAttribute("data-bind") ? $(".bind", figure) : undefined,
      onClick: clickable.has(name) ? (id) => openSpeech(id) : undefined,
      adjust: name === "emotions" ? fitFacet : undefined,
    });
    if (figure.dataset.table) addTableView(figure);
  }
  initTrends();
  initExplorer();
}

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    const facet = rendered.find((r) => r.name === "emotions");
    if (facet) render(facet.container, facet.name, facet);
  }, 250);
});

main();
