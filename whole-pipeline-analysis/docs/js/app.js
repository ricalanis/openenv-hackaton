/* DataSage App — Init, scroll animations, dynamic content */

(function () {
  'use strict';

  /* ── Scroll fade-in ── */
  function initScrollAnimations() {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1 }
    );
    document.querySelectorAll('.fade-in').forEach((el) => observer.observe(el));
  }

  /* ── Training config grid ── */
  function renderConfigGrid() {
    const grid = document.getElementById('config-grid');
    if (!grid) return;
    const cfg = DATA.trainingConfig;
    const items = [
      ['Base Model', cfg.base_model],
      ['Quantization', cfg.quantization],
      ['LoRA Rank', cfg.lora_r],
      ['LoRA Alpha', cfg.lora_alpha],
      ['Optimizer', cfg.optimizer],
      ['Epochs', cfg.epochs],
      ['Beta (KL)', cfg.beta],
      ['Epsilon (clip)', cfg.epsilon],
      ['Loss Type', cfg.loss_type],
      ['Steps/Task', cfg.max_steps_per_task],
    ];
    grid.innerHTML = items
      .map(
        ([k, v]) =>
          `<div class="config-item"><span class="config-key">${k}</span><span class="config-val">${v}</span></div>`
      )
      .join('');
  }

  /* ── Environment cards ── */
  function renderEnvCards() {
    const container = document.getElementById('env-cards');
    if (!container) return;
    const envs = DATA.environments;

    container.innerHTML = Object.entries(envs)
      .map(
        ([key, env]) => `
      <div class="card fade-in">
        <div class="card-header">
          <div class="card-dot" style="background: ${env.color};"></div>
          <span class="card-title">${env.name}</span>
        </div>
        <p class="card-desc">${env.description}</p>
        <div class="card-metric">
          <span class="card-metric-label">DataSage Reward</span>
          <span class="card-metric-value" style="color: ${env.color};">${env.datasage_reward.toFixed(3)}</span>
        </div>
        <div class="card-metric">
          <span class="card-metric-label">Base Qwen2.5-3B</span>
          <span class="card-metric-value">${env.base_reward.toFixed(3)}</span>
        </div>
        <div class="card-metric">
          <span class="card-metric-label">Metric</span>
          <span class="card-metric-value">${env.metric}</span>
        </div>
        <a class="card-link" href="${env.hf_space}" target="_blank" rel="noopener">HF Space ↗</a>
        &nbsp;&middot;&nbsp;
        <a class="card-link" href="https://huggingface.co/${env.lora_repo}" target="_blank" rel="noopener">LoRA Adapter ↗</a>
      </div>`
      )
      .join('');

    // Re-observe new fade-in elements
    document.querySelectorAll('#env-cards .fade-in').forEach((el) => {
      const obs = new IntersectionObserver(
        (entries) => {
          entries.forEach((e) => {
            if (e.isIntersecting) { e.target.classList.add('visible'); obs.unobserve(e.target); }
          });
        },
        { threshold: 0.1 }
      );
      obs.observe(el);
    });
  }

  /* ── Heatmap ── */
  function renderHeatmap() {
    const body = document.getElementById('heatmap-body');
    if (!body) return;
    const hm = DATA.heatmap;
    const domains = ['hr', 'sales', 'pm', 'it_ops'];
    const tasks = ['cleaning', 'enrichment', 'answering'];
    const taskColors = {
      cleaning: '#F59E0B',
      enrichment: '#EF4444',
      answering: '#3B82F6',
    };

    function cellColor(val, task) {
      const base = taskColors[task];
      // Opacity based on value (0-1 range)
      const opacity = Math.min(1, Math.max(0.15, val));
      return `${base}${Math.round(opacity * 40 + 15).toString(16).padStart(2, '0')}`;
    }

    body.innerHTML = domains
      .map((dom) => {
        const cells = tasks
          .map((t) => {
            const val = hm[t]?.[dom] ?? 0;
            return `<td style="background: ${cellColor(val, t)}; color: ${taskColors[t]};">${val.toFixed(4)}</td>`;
          })
          .join('');
        return `<tr><td class="domain-label">${dom.replace('_', ' ')}</td>${cells}</tr>`;
      })
      .join('');
  }

  /* ── Q&A Showcase ── */
  function renderQAShowcase() {
    const container = document.getElementById('qa-showcase');
    if (!container) return;
    const items = DATA.qaShowcase;

    container.innerHTML = items
      .map(
        (qa) => `
      <div class="qa-card">
        <div class="qa-header">
          <div class="qa-meta">
            <span class="qa-badge model">${qa.model}</span>
            <span class="qa-badge domain">${qa.domain}</span>
            <span class="qa-badge persona">${qa.persona}</span>
          </div>
          <span class="qa-reward" style="color: ${qa.reward >= 0.8 ? '#10B981' : qa.reward >= 0.5 ? '#F59E0B' : '#EF4444'};">${qa.reward.toFixed(3)}</span>
        </div>
        <div class="qa-question">${qa.question}</div>
        <div class="qa-answer">${qa.answer}</div>
        ${
          qa.cited_columns.length > 0
            ? `<div class="qa-columns">Cited columns: ${qa.cited_columns.map((c) => `<span>${c}</span>`).join('')}</div>`
            : ''
        }
      </div>`
      )
      .join('');
  }

  /* ── Links grid ── */
  function renderLinks() {
    const grid = document.getElementById('links-grid');
    if (!grid) return;
    const l = DATA.links;

    const items = [
      { icon: '📦', text: 'GitHub Repository', sub: 'Source code & notebooks', href: l.github },
      { icon: '🧹', text: 'Cleaning LoRA', sub: l.lora_repos.cleaning, href: `https://huggingface.co/${l.lora_repos.cleaning}` },
      { icon: '🔧', text: 'Enrichment LoRA', sub: l.lora_repos.enrichment, href: `https://huggingface.co/${l.lora_repos.enrichment}` },
      { icon: '💬', text: 'Answering LoRA', sub: l.lora_repos.answering, href: `https://huggingface.co/${l.lora_repos.answering}` },
      { icon: '🧹', text: 'Cleaning Space', sub: 'Live environment', href: l.hf_spaces.cleaning },
      { icon: '🔧', text: 'Enrichment Space', sub: 'Live environment', href: l.hf_spaces.enrichment },
      { icon: '💬', text: 'Answering Space', sub: 'Live environment', href: l.hf_spaces.answering },
    ];

    grid.innerHTML = items
      .map(
        (it) => `
      <a class="link-card" href="${it.href}" target="_blank" rel="noopener">
        <span class="link-icon">${it.icon}</span>
        <div>
          <div class="link-text">${it.text}</div>
          <div class="link-sub">${it.sub}</div>
        </div>
      </a>`
      )
      .join('');
  }

  /* ── Training curves ── */
  function loadTrainingCurves() {
    // Use global TRAINING_CURVES (loaded via <script> tag) — works on file:// and https://
    var curves = typeof TRAINING_CURVES !== 'undefined' ? TRAINING_CURVES : null;
    if (!curves) {
      console.error('TRAINING_CURVES not loaded');
      document.querySelectorAll('[id^="chart-reward-"], [id^="chart-components-"], #chart-loss-all').forEach(function (el) {
        el.parentElement.innerHTML = '<div class="loading">Training curves unavailable</div>';
      });
      return;
    }

    var tasks = ['cleaning', 'enrichment', 'answering'];
    var taskColors = {
      cleaning: '#F59E0B',
      enrichment: '#EF4444',
      answering: '#3B82F6',
    };

    tasks.forEach(function (task) {
      if (curves[task]) {
        renderRewardChart('chart-reward-' + task, curves[task], taskColors[task]);
        renderComponentChart('chart-components-' + task, curves[task]);
      }
    });

    renderLossChart('chart-loss-all', curves);
  }

  /* ── Init ── */
  function init() {
    renderConfigGrid();
    renderEnvCards();
    renderHeatmap();
    renderQAShowcase();
    renderLinks();

    // Static charts
    renderDatasageVsBase();
    renderGptVsQwen();
    renderRadar();
    renderByDomain();
    renderByPersona();

    // Async charts
    loadTrainingCurves();

    // Scroll animations (run after dynamic content is rendered)
    requestAnimationFrame(initScrollAnimations);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
