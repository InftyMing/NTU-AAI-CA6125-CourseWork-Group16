const TARGETS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"];
const SAMPLES = [
  {
    label: "Strong essay",
    text:
      "Public libraries still play a central role in modern education even though most resources are available online. They give students a quiet space to think, compare credible sources, and ask a librarian when an idea feels uncertain. A well-managed library, with thoughtful catalogues and reading clubs, supports independent research and rewards careful, patient writing. Online tools certainly accelerate the search for information, but they rarely teach the same habits of focus and revision.",
  },
  {
    label: "Average essay",
    text:
      "Technology can help learning when teachers use it carefully. Students can find videos and articles online and practice with apps. However, screens make it easy to be distracted and not every student reads what teachers send. So I think the best classes mix online materials with normal lessons, and teachers should explain how to study with the new tools.",
  },
  {
    label: "Beginner essay",
    text:
      "School lunch need change because many student no like it. The food sometime cold and not have much vegetable. If school ask students what they want, lunch will be more good and student happy. Also, teacher should join because they care about us.",
  },
];

const COMPONENT_FAMILIES = {
  mean_baseline: "Sanity baseline",
  ridge_tfidf: "Linear · TF-IDF",
  ridge_tfidf_per_target: "Linear · per-target",
  ridge_tfidf_fused: "Linear · fused features",
  svr_tfidf: "Kernel · SVR",
  lgbm_text_features: "Tree · text statistics",
  lgbm_svd_fused: "Tree · SVD + statistics",
  stacked_ensemble: "Stacked ensemble",
  deberta_v3_base: "Transformer (skipped)",
};

const COMPONENT_DESCRIPTIONS = {
  mean_baseline: "Predicts the global mean of every target. Establishes the worst-case lower bound.",
  ridge_tfidf: "Single-alpha Ridge on word + character TF-IDF features.",
  ridge_tfidf_per_target: "Six Ridges, each with its own alpha selected on an inner 90/10 split.",
  ridge_tfidf_fused: "TF-IDF concatenated with ten standardized text statistics.",
  svr_tfidf: "MultiOutputRegressor wrapping an RBF SVR per target on TF-IDF features.",
  lgbm_text_features: "LightGBM trained on the ten interpretable text statistics only.",
  lgbm_svd_fused: "LightGBM on a 128-dim TruncatedSVD projection of TF-IDF plus the text statistics.",
  stacked_ensemble: "Per-target convex blend of the three component models, weights fitted on OOF predictions.",
  deberta_v3_base: "DeBERTa-v3-base fine-tuning pipeline; not executed in the demo environment.",
};

let metricsState = { items: [], final: {} };
let auditState = {};
let errorState = {};
let activeModelName = null;

const el = (id) => document.getElementById(id);

function fmt(value, digits = 5) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function formatBytes(value) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(2)} MB`;
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} -> ${response.status}`);
  return response.json();
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return response.json();
}

function renderAuditCards(audit) {
  const cards = el("auditCards");
  const cells = [
    ["Train essays", audit.train_rows ?? "—", "ELLIPSE corpus"],
    ["Test essays", audit.test_rows ?? "—", "Kaggle code-competition sample"],
    ["Avg word count", audit.train_text_summary?.word_count?.mean?.toFixed(0) ?? "—", "training set"],
    ["Avg sentence count", audit.train_text_summary?.sentence_count?.mean?.toFixed(1) ?? "—", "training set"],
    ["Targets", TARGETS.length, "cohesion → conventions"],
    ["Score range", "1.0 – 5.0", "step 0.5"],
  ];
  cards.innerHTML = cells
    .map(
      ([label, value, foot]) =>
        `<div class="card"><strong>${value}</strong><span>${label}</span><div class="kpi-foot">${foot}</div></div>`
    )
    .join("");

  el("kpiTrain").textContent = audit.train_rows ?? "—";
}

function renderMeanChart(audit) {
  const svg = el("targetMeanChart");
  if (!svg || !audit.target_summary) return;
  const w = 480;
  const h = 220;
  const margin = { top: 16, right: 16, bottom: 32, left: 36 };
  const chartW = w - margin.left - margin.right;
  const chartH = h - margin.top - margin.bottom;
  const means = TARGETS.map((t) => audit.target_summary[t]?.mean ?? 0);
  const stds = TARGETS.map((t) => audit.target_summary[t]?.std ?? 0);
  const max = 5.0;
  const barWidth = chartW / TARGETS.length - 12;
  const yScale = (v) => chartH - (v / max) * chartH;

  const yTicks = [1, 2, 3, 4, 5];
  const guides = yTicks
    .map(
      (tick) =>
        `<line x1="${margin.left}" x2="${margin.left + chartW}" y1="${margin.top + yScale(tick)}" y2="${
          margin.top + yScale(tick)
        }" stroke="#d6cdb8" stroke-width="1" stroke-dasharray="2 4" />` +
        `<text x="${margin.left - 6}" y="${margin.top + yScale(tick) + 3}" text-anchor="end" fill="#6f685b" font-size="10">${tick}</text>`
    )
    .join("");

  const bars = means
    .map((mean, i) => {
      const x = margin.left + i * (barWidth + 12) + 6;
      const y = margin.top + yScale(mean);
      const stdHigh = margin.top + yScale(Math.min(mean + stds[i], max));
      const stdLow = margin.top + yScale(Math.max(mean - stds[i], 0));
      return `
        <rect x="${x}" y="${y}" width="${barWidth}" height="${chartH - yScale(mean)}" fill="#161310" rx="6"/>
        <line x1="${x + barWidth / 2}" x2="${x + barWidth / 2}" y1="${stdLow}" y2="${stdHigh}" stroke="#8a5a1c" stroke-width="2" />
        <line x1="${x + barWidth / 2 - 4}" x2="${x + barWidth / 2 + 4}" y1="${stdHigh}" y2="${stdHigh}" stroke="#8a5a1c" stroke-width="2" />
        <line x1="${x + barWidth / 2 - 4}" x2="${x + barWidth / 2 + 4}" y1="${stdLow}" y2="${stdLow}" stroke="#8a5a1c" stroke-width="2" />
        <text x="${x + barWidth / 2}" y="${margin.top + chartH + 18}" text-anchor="middle" font-size="11" fill="#14110d">${TARGETS[i].slice(0, 6)}</text>
        <text x="${x + barWidth / 2}" y="${y - 6}" text-anchor="middle" font-size="11" fill="#14110d" font-weight="600">${mean.toFixed(2)}</text>
      `;
    })
    .join("");
  svg.innerHTML = `${guides}${bars}`;
}

function renderCorrHeatmap(audit) {
  const wrap = el("corrHeatmap");
  if (!wrap || !audit.target_correlations) return;
  const cells = [];
  cells.push('<div class="cell label"></div>');
  TARGETS.forEach((t) => cells.push(`<div class="cell col-label">${t.slice(0, 4)}</div>`));
  TARGETS.forEach((row) => {
    cells.push(`<div class="cell label">${row}</div>`);
    TARGETS.forEach((col) => {
      const value = audit.target_correlations[row]?.[col] ?? 0;
      const intensity = Math.min(1, Math.max(0, (value - 0.6) / 0.4));
      const lightness = 96 - intensity * 60;
      const fontColor = lightness < 60 ? "#fffaee" : "#14110d";
      cells.push(
        `<div class="cell" style="background: hsl(35, 45%, ${lightness}%); color: ${fontColor}">${value.toFixed(2)}</div>`
      );
    });
  });
  wrap.innerHTML = cells.join("");
}

function renderLengthChart(audit) {
  const svg = el("lengthChart");
  if (!svg) return;
  const w = 720;
  const h = 180;
  const margin = { top: 18, right: 24, bottom: 36, left: 48 };
  const chartW = w - margin.left - margin.right;
  const chartH = h - margin.top - margin.bottom;
  const stats = audit.train_text_summary?.word_count;
  if (!stats) {
    svg.innerHTML = "";
    return;
  }
  const min = stats.min ?? 0;
  const q1 = stats["25%"] ?? 0;
  const median = stats["50%"] ?? 0;
  const q3 = stats["75%"] ?? 0;
  const max = stats.max ?? 0;
  const range = max - min || 1;
  const x = (v) => margin.left + ((v - min) / range) * chartW;

  const yMid = margin.top + chartH / 2;
  const ticks = [min, q1, median, q3, max];
  const tickLines = ticks
    .map(
      (t) =>
        `<line x1="${x(t)}" x2="${x(t)}" y1="${yMid - chartH / 2 + 8}" y2="${yMid + chartH / 2 - 8}" stroke="#d6cdb8" stroke-dasharray="2 4" />` +
        `<text x="${x(t)}" y="${margin.top + chartH + 24}" text-anchor="middle" font-size="11" fill="#6f685b">${Math.round(t)}</text>`
    )
    .join("");

  const box = `
    <rect x="${x(q1)}" y="${yMid - 22}" width="${x(q3) - x(q1)}" height="44" fill="#fffaee" stroke="#161310" stroke-width="1" rx="6" />
    <line x1="${x(median)}" x2="${x(median)}" y1="${yMid - 22}" y2="${yMid + 22}" stroke="#161310" stroke-width="2" />
    <line x1="${x(min)}" x2="${x(q1)}" y1="${yMid}" y2="${yMid}" stroke="#161310" stroke-width="1" />
    <line x1="${x(q3)}" x2="${x(max)}" y1="${yMid}" y2="${yMid}" stroke="#161310" stroke-width="1" />
    <line x1="${x(min)}" x2="${x(min)}" y1="${yMid - 8}" y2="${yMid + 8}" stroke="#161310" />
    <line x1="${x(max)}" x2="${x(max)}" y1="${yMid - 8}" y2="${yMid + 8}" stroke="#161310" />
    <text x="${x(median)}" y="${yMid - 30}" text-anchor="middle" font-size="11" fill="#14110d" font-weight="600">median ${Math.round(median)} words</text>
  `;
  svg.innerHTML = `${tickLines}${box}`;
}

function renderMetrics(payload) {
  metricsState = payload;
  const items = (payload.items || []).filter((item) => item.cv_mcrmse !== null);
  items.sort((a, b) => (a.cv_mcrmse ?? 9) - (b.cv_mcrmse ?? 9));
  const best = items[0];
  if (best) {
    el("kpiBest").textContent = best.cv_mcrmse.toFixed(5);
    el("kpiBestModel").textContent = `${best.name} · ${COMPONENT_FAMILIES[best.name] ?? "model"}`;
  }
  const body = el("metricsBody");
  body.innerHTML = items
    .map((item) => {
      const family = COMPONENT_FAMILIES[item.name] ?? "—";
      return `<tr data-name="${item.name}">
        <td><strong>${item.name}</strong><div class="kpi-foot">${COMPONENT_DESCRIPTIONS[item.name] ?? ""}</div></td>
        <td>${family}</td>
        <td class="score">${fmt(item.cv_mcrmse)}</td>
        <td class="file">${item.submission_path?.replace(/\\/g, "/") ?? ""}</td>
      </tr>`;
    })
    .join("");
  body.querySelectorAll("tr").forEach((row) => {
    row.addEventListener("click", () => {
      activeModelName = row.dataset.name;
      highlightModel(activeModelName);
    });
  });
  if (!activeModelName && best) activeModelName = best.name;
  if (activeModelName) highlightModel(activeModelName);
}

function highlightModel(name) {
  document.querySelectorAll(".leaderboard tr").forEach((row) => {
    row.classList.toggle("active", row.dataset.name === name);
  });
  const item = (metricsState.items || []).find((m) => m.name === name);
  if (!item) return;
  el("perTargetSubtitle").textContent = `${item.name} · CV ${fmt(item.cv_mcrmse)}`;
  renderPerTargetChart(item);
  renderFoldChart(item);
}

function renderPerTargetChart(item) {
  const svg = el("perTargetChart");
  if (!svg) return;
  const w = 460;
  const h = 260;
  const margin = { top: 12, right: 24, bottom: 24, left: 110 };
  const data = TARGETS.map((t) => ({ name: t, value: item.column_rmse?.[t] ?? 0 }));
  const max = Math.max(0.7, ...data.map((d) => d.value));
  const chartW = w - margin.left - margin.right;
  const rowH = (h - margin.top - margin.bottom) / data.length;

  const bars = data
    .map((d, i) => {
      const y = margin.top + i * rowH + rowH * 0.15;
      const barH = rowH * 0.6;
      const value = d.value;
      const width = (value / max) * chartW;
      return `
        <text x="${margin.left - 10}" y="${y + barH / 2 + 4}" text-anchor="end" font-size="12" fill="#14110d">${d.name}</text>
        <rect x="${margin.left}" y="${y}" width="${chartW}" height="${barH}" fill="#ece6d4" rx="6" />
        <rect x="${margin.left}" y="${y}" width="${width}" height="${barH}" fill="#161310" rx="6" />
        <text x="${margin.left + width + 6}" y="${y + barH / 2 + 4}" font-size="12" fill="#14110d" font-weight="600">${value.toFixed(4)}</text>
      `;
    })
    .join("");
  svg.innerHTML = bars;
}

function renderFoldChart(item) {
  const svg = el("foldChart");
  if (!svg) return;
  const folds = item.fold_scores || [];
  const w = 720;
  const h = 200;
  const margin = { top: 24, right: 16, bottom: 32, left: 48 };
  const chartW = w - margin.left - margin.right;
  const chartH = h - margin.top - margin.bottom;
  if (!folds.length) {
    svg.innerHTML = `<text x="${w / 2}" y="${h / 2}" text-anchor="middle" fill="#6f685b" font-size="13">Stacked ensemble inherits its fold split from component models.</text>`;
    return;
  }
  const xs = folds.map((_, i) => margin.left + (i / Math.max(1, folds.length - 1)) * chartW);
  const values = folds.map((f) => f.mcrmse);
  const min = Math.min(...values, item.cv_mcrmse) - 0.01;
  const max = Math.max(...values, item.cv_mcrmse) + 0.01;
  const y = (v) => margin.top + chartH - ((v - min) / (max - min)) * chartH;

  const yTicks = 4;
  const yLines = Array.from({ length: yTicks + 1 }, (_, i) => {
    const value = min + ((max - min) * i) / yTicks;
    return `
      <line x1="${margin.left}" x2="${margin.left + chartW}" y1="${y(value)}" y2="${y(value)}" stroke="#d6cdb8" stroke-dasharray="2 4" />
      <text x="${margin.left - 6}" y="${y(value) + 3}" text-anchor="end" font-size="10" fill="#6f685b">${value.toFixed(3)}</text>
    `;
  }).join("");

  const path = values
    .map((v, i) => `${i === 0 ? "M" : "L"}${xs[i].toFixed(1)},${y(v).toFixed(1)}`)
    .join(" ");

  const points = values
    .map(
      (v, i) =>
        `<circle cx="${xs[i]}" cy="${y(v)}" r="4" fill="#161310" />` +
        `<text x="${xs[i]}" y="${y(v) - 10}" text-anchor="middle" font-size="11" fill="#14110d">${v.toFixed(4)}</text>` +
        `<text x="${xs[i]}" y="${margin.top + chartH + 18}" text-anchor="middle" font-size="11" fill="#6f685b">fold ${folds[i].fold}</text>`
    )
    .join("");

  const meanLine = `<line x1="${margin.left}" x2="${margin.left + chartW}" y1="${y(item.cv_mcrmse)}" y2="${y(item.cv_mcrmse)}" stroke="#92382b" stroke-dasharray="4 4" />` +
    `<text x="${margin.left + chartW}" y="${y(item.cv_mcrmse) - 6}" text-anchor="end" font-size="11" fill="#92382b">overall ${item.cv_mcrmse.toFixed(4)}</text>`;

  svg.innerHTML = `${yLines}<path d="${path}" fill="none" stroke="#161310" stroke-width="2" />${points}${meanLine}`;
}

function renderErrorBuckets(payload) {
  errorState = payload;
  if (!payload || payload.status === "missing") return;
  drawBucketChart("lengthBucketChart", payload.length_buckets, "bucket", "mcrmse");
  drawBucketChart("scoreBucketChart", payload.score_buckets, "bucket", "mcrmse");

  const overall = payload.overall || {};
  const rows = TARGETS.map((target) => {
    return `<tr>
      <td>${target}</td>
      <td class="numeric">${(overall.mean?.[target] ?? 0).toFixed(4)}</td>
      <td class="numeric">${(overall.mae?.[target] ?? 0).toFixed(4)}</td>
      <td class="numeric">${(overall.std?.[target] ?? 0).toFixed(4)}</td>
      <td class="numeric">${(overall.max_abs?.[target] ?? 0).toFixed(4)}</td>
    </tr>`;
  }).join("");
  const table = el("errorTable");
  table.innerHTML = `
    <thead>
      <tr><th>Target</th><th class="numeric">Mean residual</th><th class="numeric">MAE</th><th class="numeric">Std</th><th class="numeric">Max |error|</th></tr>
    </thead>
    <tbody>${rows}</tbody>
  `;
}

function drawBucketChart(svgId, buckets, labelKey, valueKey) {
  const svg = el(svgId);
  if (!svg || !buckets || !buckets.length) return;
  const w = 460;
  const h = 220;
  const margin = { top: 24, right: 16, bottom: 60, left: 40 };
  const chartW = w - margin.left - margin.right;
  const chartH = h - margin.top - margin.bottom;
  const max = Math.max(...buckets.map((b) => b[valueKey])) * 1.15;
  const barWidth = chartW / buckets.length - 16;
  const yScale = (v) => margin.top + chartH - (v / max) * chartH;

  const bars = buckets
    .map((b, i) => {
      const x = margin.left + i * (barWidth + 16) + 8;
      const y = yScale(b[valueKey]);
      return `
        <rect x="${x}" y="${y}" width="${barWidth}" height="${margin.top + chartH - y}" fill="#161310" rx="6" />
        <text x="${x + barWidth / 2}" y="${y - 6}" text-anchor="middle" font-size="11" fill="#14110d" font-weight="600">${b[valueKey].toFixed(4)}</text>
        <text x="${x + barWidth / 2}" y="${margin.top + chartH + 18}" text-anchor="middle" font-size="11" fill="#14110d">${b[labelKey]}</text>
        <text x="${x + barWidth / 2}" y="${margin.top + chartH + 34}" text-anchor="middle" font-size="10" fill="#6f685b">n = ${b.count}</text>
      `;
    })
    .join("");

  svg.innerHTML = bars;
}

function renderTeam(payload) {
  const grid = el("teamGrid");
  grid.innerHTML = (payload.members || [])
    .map((member) => {
      const initials = member.name
        .split(/\s+/)
        .filter(Boolean)
        .slice(0, 2)
        .map((part) => part[0])
        .join("");
      return `<article class="member-card">
        <div class="avatar">${initials}</div>
        <div>
          <h3>${member.name}</h3>
          <strong>${member.role}</strong>
          <p>${member.contribution}</p>
        </div>
      </article>`;
    })
    .join("");
}

function renderSubmission(payload) {
  const finalBox = el("finalBox");
  const list = el("submissionList");
  if (payload.final && payload.final.best) {
    finalBox.textContent = JSON.stringify(payload.final.best, null, 2);
  } else if (payload.final) {
    finalBox.textContent = JSON.stringify(payload.final, null, 2);
  }
  list.innerHTML = (payload.available || [])
    .map(
      (file) =>
        `<li><span><strong>${file.name}</strong><br/><code>${file.path}</code></span><span>${formatBytes(
          file.size_bytes
        )}</span></li>`
    )
    .join("");
}

function attachSampleTabs() {
  const wrap = el("sampleTabs");
  wrap.innerHTML = SAMPLES.map(
    (sample, i) => `<button data-i="${i}" class="${i === 1 ? "active" : ""}">${sample.label}</button>`
  ).join("");
  el("essayInput").value = SAMPLES[1].text;
  updateWordCounter();
  wrap.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () => {
      wrap.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const idx = Number(btn.dataset.i);
      el("essayInput").value = SAMPLES[idx].text;
      updateWordCounter();
      runPredict();
    });
  });
}

function updateWordCounter() {
  const text = el("essayInput").value.trim();
  const words = text ? text.split(/\s+/).length : 0;
  el("wordCounter").textContent = `${words} words`;
}

function renderPrediction(payload) {
  if (payload.error) {
    el("scoreBars").innerHTML = `<div class="signal-list"><li>${payload.error}</li></div>`;
    el("overallScore").textContent = "—";
    el("signalList").innerHTML = "";
    return;
  }
  const scores = payload.scores;
  const bars = TARGETS.map((target) => {
    const v = scores[target];
    const ratio = ((v - 1) / 4) * 100;
    return `<div class="score-row">
      <span class="name">${target}</span>
      <div class="bar-track"><div class="bar-fill" style="width: ${ratio.toFixed(1)}%"></div></div>
      <span class="value">${v.toFixed(2)}</span>
    </div>`;
  }).join("");
  el("scoreBars").innerHTML = bars;
  el("overallScore").textContent = payload.overall.toFixed(2);
  el("signalList").innerHTML = (payload.notes || [])
    .map((note) => `<li>${note}</li>`)
    .join("");
}

async function runPredict() {
  const text = el("essayInput").value;
  const payload = await postJson("/api/predict", { text });
  renderPrediction(payload);
}

function attachNavObserver() {
  const sections = ["overview", "pipeline", "data", "models", "errors", "demo", "team", "submission"]
    .map((id) => el(id))
    .filter(Boolean);
  const nav = document.querySelectorAll("#navLinks a");
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          nav.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${entry.target.id}`));
        }
      });
    },
    { rootMargin: "-40% 0px -55% 0px" }
  );
  sections.forEach((section) => observer.observe(section));
}

async function init() {
  attachSampleTabs();
  attachNavObserver();
  el("essayInput").addEventListener("input", updateWordCounter);
  el("predictBtn").addEventListener("click", runPredict);
  el("clearBtn").addEventListener("click", () => {
    el("essayInput").value = "";
    updateWordCounter();
    el("scoreBars").innerHTML = "";
    el("overallScore").textContent = "—";
    el("signalList").innerHTML = "";
  });

  try {
    auditState = await getJson("/api/audit");
    renderAuditCards(auditState);
    renderMeanChart(auditState);
    renderCorrHeatmap(auditState);
    renderLengthChart(auditState);

    renderTeam(await getJson("/api/team"));
    renderMetrics(await getJson("/api/metrics"));
    renderErrorBuckets(await getJson("/api/error_analysis"));
    renderSubmission(await getJson("/api/submission"));
    el("statusText").textContent = "live";

    runPredict();
  } catch (error) {
    el("statusText").textContent = "offline";
    console.error(error);
  }
}

init();
