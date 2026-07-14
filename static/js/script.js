// =============================================================
// Customer Churn Prediction — front-end behaviour
// =============================================================

document.addEventListener("DOMContentLoaded", () => {
  initNavbar();
  initMobileNav();
  initScrollReveal();
  initHeroCanvas();
  initPredictionForm();
  initCharts();
});

/* ---------------------------- Navbar shrink + active link ---------------------------- */
function initNavbar() {
  const nav = document.querySelector(".navbar");
  if (!nav) return;
  const onScroll = () => nav.classList.toggle("scrolled", window.scrollY > 30);
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });

  const links = document.querySelectorAll(".nav-links a[href^='#']");
  const sections = [...links].map((l) => document.querySelector(l.getAttribute("href"))).filter(Boolean);

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          links.forEach((l) => l.classList.remove("active"));
          const match = document.querySelector(`.nav-links a[href="#${entry.target.id}"]`);
          if (match) match.classList.add("active");
        }
      });
    },
    { rootMargin: "-45% 0px -50% 0px" }
  );
  sections.forEach((s) => io.observe(s));
}

/* ---------------------------- Mobile nav ---------------------------- */
function initMobileNav() {
  const toggle = document.querySelector(".nav-toggle");
  const links = document.querySelector(".nav-links");
  if (!toggle || !links) return;
  toggle.addEventListener("click", () => {
    const open = links.classList.toggle("mobile-open");
    if (open) {
      links.style.cssText =
        "display:flex;flex-direction:column;position:fixed;top:66px;left:16px;right:16px;background:rgba(10,15,31,0.97);border:1px solid rgba(167,139,250,0.2);border-radius:16px;padding:10px;backdrop-filter:blur(14px);";
    } else {
      links.removeAttribute("style");
    }
  });
  links.querySelectorAll("a").forEach((a) =>
    a.addEventListener("click", () => {
      links.classList.remove("mobile-open");
      links.removeAttribute("style");
    })
  );
}

/* ---------------------------- Scroll reveal ---------------------------- */
function initScrollReveal() {
  const els = document.querySelectorAll(".reveal");
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in-view");
          io.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  els.forEach((el) => io.observe(el));
}

/* ---------------------------- Hero canvas: data-flow network ---------------------------- */
function initHeroCanvas() {
  const canvas = document.getElementById("hero-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  let w, h, nodes;
  const NODE_COUNT = 46;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function resize() {
    w = canvas.width = canvas.offsetWidth * devicePixelRatio;
    h = canvas.height = canvas.offsetHeight * devicePixelRatio;
  }

  function makeNodes() {
    nodes = Array.from({ length: NODE_COUNT }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.25 * devicePixelRatio,
      vy: (Math.random() - 0.5) * 0.25 * devicePixelRatio,
      r: (Math.random() * 1.6 + 1) * devicePixelRatio,
    }));
  }

  function step() {
    ctx.clearRect(0, 0, w, h);
    nodes.forEach((n) => {
      n.x += n.vx;
      n.y += n.vy;
      if (n.x < 0 || n.x > w) n.vx *= -1;
      if (n.y < 0 || n.y > h) n.vy *= -1;
    });

    const maxDist = 150 * devicePixelRatio;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < maxDist) {
          ctx.strokeStyle = `rgba(167,139,250,${0.16 * (1 - dist / maxDist)})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    nodes.forEach((n) => {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(196,181,253,0.65)";
      ctx.fill();
    });

    if (!reduceMotion) requestAnimationFrame(step);
  }

  resize();
  makeNodes();
  step();
  window.addEventListener("resize", () => {
    resize();
    makeNodes();
  });
}

/* ---------------------------- Prediction form ---------------------------- */
function initPredictionForm() {
  const form = document.getElementById("predict-form");
  if (!form) return;
  const resultCard = document.getElementById("result-card");
  const submitBtn = document.getElementById("predict-submit-btn");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(form).entries());

    submitBtn.disabled = true;
    const originalLabel = submitBtn.textContent;
    submitBtn.textContent = "Running model…";

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      const json = await res.json();

      if (!res.ok) throw new Error(json.error || "Prediction failed");
      renderResult(json);
    } catch (err) {
      renderResult({ error: err.message });
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = originalLabel;
    }
  });

  function renderResult(result) {
    if (result.error) {
      resultCard.innerHTML = `
        <div class="result-empty">
          <p style="color:var(--churn)">${escapeHtml(result.error)}</p>
          <p style="font-size:0.78rem;">Tip: run <code>python model/train_model.py</code> once, then restart the Flask server, to enable live predictions.</p>
        </div>`;
      return;
    }

    const isChurn = result.prediction === "Churn";
    const primary = isChurn ? result.churn_probability : result.stay_probability;

    resultCard.innerHTML = `
      <div class="result-verdict">
        <div class="verdict-label">Model output</div>
        <div class="verdict-value ${isChurn ? "churn" : "stay"}">
          ${isChurn ? "Customer Will Churn" : "Customer Will Stay"}
        </div>
      </div>
      <div class="prob-bar-track">
        <div class="prob-bar-fill ${isChurn ? "churn" : "stay"}" style="width:0%"></div>
      </div>
      <div class="prob-labels">
        <span>Confidence</span>
        <span>${primary}%</span>
      </div>
      <div style="margin-top:22px; display:flex; justify-content:space-between; font-size:0.82rem;">
        <span style="color:var(--ink-muted)">Stay probability</span>
        <span style="font-family:var(--font-mono); color:var(--stay)">${result.stay_probability}%</span>
      </div>
      <div style="display:flex; justify-content:space-between; font-size:0.82rem; margin-top:6px;">
        <span style="color:var(--ink-muted)">Churn probability</span>
        <span style="font-family:var(--font-mono); color:var(--churn)">${result.churn_probability}%</span>
      </div>
    `;
    requestAnimationFrame(() => {
      const fill = resultCard.querySelector(".prob-bar-fill");
      if (fill) fill.style.width = `${primary}%`;
    });
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

/* ---------------------------- Charts (Chart.js) ---------------------------- */
function initCharts() {
  if (typeof Chart === "undefined" || typeof METRICS === "undefined" || !METRICS) return;

  Chart.defaults.color = "#94a3b8";
  Chart.defaults.font.family = "JetBrains Mono, monospace";
  Chart.defaults.font.size = 11;

  const best = METRICS.models[METRICS.best_model];
  const modelNames = Object.keys(METRICS.models);

  // Comparison bar chart: F1 across models
  const cmpEl = document.getElementById("chart-comparison");
  if (cmpEl) {
    new Chart(cmpEl, {
      type: "bar",
      data: {
        labels: modelNames,
        datasets: [
          { label: "Accuracy", data: modelNames.map((m) => METRICS.models[m].accuracy), backgroundColor: "rgba(139,92,246,0.75)" },
          { label: "F1 Score", data: modelNames.map((m) => METRICS.models[m].f1_score), backgroundColor: "rgba(167,139,250,0.4)" },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { boxWidth: 12 } } },
        scales: {
          y: { min: 0, max: 1, grid: { color: "rgba(148,163,184,0.08)" } },
          x: { grid: { display: false } },
        },
      },
    });
  }

  // ROC curve
  const rocEl = document.getElementById("chart-roc");
  if (rocEl && METRICS.roc_curve) {
    new Chart(rocEl, {
      type: "line",
      data: {
        labels: METRICS.roc_curve.map((p) => p.fpr.toFixed(2)),
        datasets: [
          {
            label: `ROC (AUC = ${best.roc_auc})`,
            data: METRICS.roc_curve.map((p) => p.tpr),
            borderColor: "#a78bfa",
            backgroundColor: "rgba(139,92,246,0.15)",
            fill: true,
            tension: 0.3,
            pointRadius: 0,
          },
          {
            label: "Random baseline",
            data: METRICS.roc_curve.map((p) => p.fpr),
            borderColor: "rgba(148,163,184,0.35)",
            borderDash: [4, 4],
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { boxWidth: 12 } } },
        scales: {
          x: { title: { display: true, text: "False Positive Rate" }, grid: { color: "rgba(148,163,184,0.06)" } },
          y: { title: { display: true, text: "True Positive Rate" }, grid: { color: "rgba(148,163,184,0.06)" } },
        },
      },
    });
  }
}
