/* DataSage Charts — Chart.js 4.x rendering functions */

const COLORS = {
  cleaning: '#F59E0B',
  enrichment: '#EF4444',
  answering: '#3B82F6',
  green: '#10B981',
  purple: '#7C3AED',
  pink: '#EC4899',
  gpt4o: '#10A37F',
  qwen3: '#7C3AED',
};

const COMPONENT_COLORS = [
  '#3B82F6', '#F59E0B', '#EF4444', '#10B981', '#EC4899', '#7C3AED', '#06B6D4',
];

// Chart.js defaults
Chart.defaults.color = '#a0a0b8';
Chart.defaults.borderColor = '#2a2a4a';
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyle = 'circle';
Chart.defaults.plugins.legend.labels.padding = 16;
Chart.defaults.animation.duration = 800;

function makeLineOpts(title) {
  return {
    responsive: true,
    maintainAspectRatio: true,
    aspectRatio: 1.6,
    plugins: {
      legend: { display: false },
      tooltip: { mode: 'index', intersect: false },
    },
    scales: {
      x: {
        title: { display: true, text: 'Epoch' },
        grid: { color: 'rgba(42,42,74,0.5)' },
      },
      y: {
        title: { display: true, text: title || 'Value' },
        grid: { color: 'rgba(42,42,74,0.5)' },
      },
    },
    elements: {
      point: { radius: 0 },
      line: { borderWidth: 2 },
    },
    interaction: { mode: 'nearest', axis: 'x', intersect: false },
  };
}

function samplePoints(arr, maxPoints) {
  if (arr.length <= maxPoints) return arr;
  const step = arr.length / maxPoints;
  const result = [];
  for (let i = 0; i < maxPoints; i++) {
    result.push(arr[Math.floor(i * step)]);
  }
  return result;
}

/* ── Training Curves ── */

function renderRewardChart(canvasId, taskData, color) {
  const maxPts = 100;
  const epochs = samplePoints(taskData.epochs, maxPts);
  const rewards = samplePoints(taskData.reward, maxPts);

  new Chart(document.getElementById(canvasId), {
    type: 'line',
    data: {
      labels: epochs.map(e => e.toFixed(2)),
      datasets: [{
        label: 'Total Reward',
        data: rewards,
        borderColor: color,
        backgroundColor: color + '20',
        fill: true,
      }],
    },
    options: makeLineOpts('Reward'),
  });
}

function renderComponentChart(canvasId, taskData) {
  const maxPts = 100;
  const epochs = samplePoints(taskData.epochs, maxPts);
  const components = taskData.component_rewards;
  const keys = Object.keys(components);

  const datasets = keys.map((key, i) => ({
    label: key.replace(/_/g, ' ').replace(/mean$/, '').trim(),
    data: samplePoints(components[key], maxPts),
    borderColor: COMPONENT_COLORS[i % COMPONENT_COLORS.length],
    borderWidth: 1.5,
  }));

  const opts = makeLineOpts('Reward');
  opts.plugins.legend = { display: true, position: 'bottom' };

  new Chart(document.getElementById(canvasId), {
    type: 'line',
    data: { labels: epochs.map(e => e.toFixed(2)), datasets },
    options: opts,
  });
}

function renderLossChart(canvasId, curves) {
  const maxPts = 100;
  const tasks = ['cleaning', 'enrichment', 'answering'];
  const taskColors = [COLORS.cleaning, COLORS.enrichment, COLORS.answering];

  // Use longest epoch array for labels
  let longestEpochs = [];
  tasks.forEach(t => {
    if (curves[t] && curves[t].epochs.length > longestEpochs.length) {
      longestEpochs = curves[t].epochs;
    }
  });
  const labels = samplePoints(longestEpochs, maxPts).map(e => e.toFixed(2));

  const datasets = tasks.map((t, i) => ({
    label: t.charAt(0).toUpperCase() + t.slice(1),
    data: curves[t] ? samplePoints(curves[t].loss, maxPts) : [],
    borderColor: taskColors[i],
    borderWidth: 2,
  }));

  const opts = makeLineOpts('Loss');
  opts.plugins.legend = { display: true, position: 'bottom' };

  new Chart(document.getElementById(canvasId), {
    type: 'line',
    data: { labels, datasets },
    options: opts,
  });
}

/* ── Results Comparison ── */

function renderGroupedBar(canvasId, groups, datasets, yLabel) {
  const barOpts = {
    responsive: true,
    maintainAspectRatio: true,
    aspectRatio: 1.4,
    plugins: {
      legend: { display: true, position: 'bottom' },
      tooltip: { mode: 'index', intersect: false },
    },
    scales: {
      x: { grid: { display: false } },
      y: {
        beginAtZero: true,
        max: 1.0,
        title: { display: true, text: yLabel || 'Reward' },
        grid: { color: 'rgba(42,42,74,0.5)' },
      },
    },
  };

  new Chart(document.getElementById(canvasId), {
    type: 'bar',
    data: { labels: groups, datasets },
    options: barOpts,
  });
}

function renderDatasageVsBase() {
  const d = DATA.datasageVsBase;
  const tasks = ['Cleaning', 'Enrichment', 'Answering'];

  renderGroupedBar('chart-datasage-vs-base', tasks, [
    {
      label: 'DataSage LoRA',
      data: [d.datasage.cleaning, d.datasage.enrichment, d.datasage.answering],
      backgroundColor: COLORS.answering + 'CC',
      borderColor: COLORS.answering,
      borderWidth: 1,
    },
    {
      label: 'Base Qwen2.5-3B',
      data: [d.base_qwen.cleaning, d.base_qwen.enrichment, d.base_qwen.answering],
      backgroundColor: COLORS.purple + '88',
      borderColor: COLORS.purple,
      borderWidth: 1,
    },
  ], 'Avg Reward');

  const impNote = document.getElementById('note-improvement');
  if (impNote) {
    impNote.textContent = `Answering: +${d.improvement.answering}% improvement over base model.`;
  }
}

function renderGptVsQwen() {
  const b = DATA.benchmarkComparison;
  const tasks = ['Cleaning', 'Enrichment', 'Answering'];

  renderGroupedBar('chart-gpt-vs-qwen', tasks, [
    {
      label: 'GPT-4o-mini',
      data: [b.gpt4o_mini.cleaning, b.gpt4o_mini.enrichment, b.gpt4o_mini.answering],
      backgroundColor: COLORS.gpt4o + 'CC',
      borderColor: COLORS.gpt4o,
      borderWidth: 1,
    },
    {
      label: 'Qwen3-8B',
      data: [b.qwen3_8b.cleaning, b.qwen3_8b.enrichment, b.qwen3_8b.answering],
      backgroundColor: COLORS.qwen3 + '88',
      borderColor: COLORS.qwen3,
      borderWidth: 1,
    },
  ], 'Avg Reward');
}

function renderRadar() {
  const r = DATA.radar;
  const labels = ['Cleaning', 'Enrichment', 'Answering'];

  new Chart(document.getElementById('chart-radar'), {
    type: 'radar',
    data: {
      labels,
      datasets: [
        {
          label: 'DataSage LoRA',
          data: r.datasage,
          borderColor: COLORS.answering,
          backgroundColor: COLORS.answering + '30',
          borderWidth: 2,
          pointRadius: 4,
        },
        {
          label: 'Base Qwen2.5-3B',
          data: r.base_qwen,
          borderColor: COLORS.purple,
          backgroundColor: COLORS.purple + '20',
          borderWidth: 2,
          pointRadius: 4,
        },
        {
          label: 'GPT-4o-mini',
          data: r.gpt4o_mini,
          borderColor: COLORS.gpt4o,
          backgroundColor: COLORS.gpt4o + '20',
          borderWidth: 2,
          pointRadius: 4,
        },
        {
          label: 'Qwen3-8B',
          data: r.qwen3_8b,
          borderColor: COLORS.qwen3,
          backgroundColor: COLORS.qwen3 + '20',
          borderWidth: 2,
          pointRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 1,
      plugins: {
        legend: { display: true, position: 'bottom' },
      },
      scales: {
        r: {
          beginAtZero: true,
          max: 1.0,
          grid: { color: 'rgba(42,42,74,0.5)' },
          angleLines: { color: 'rgba(42,42,74,0.5)' },
          pointLabels: { font: { size: 13, weight: 600 } },
          ticks: { backdropColor: 'transparent', stepSize: 0.2 },
        },
      },
    },
  });
}

/* ── Breakdown ── */

function renderByDomain() {
  const d = DATA.perDomainAnswering;
  const domains = [...new Set([
    ...Object.keys(d.datasage || {}),
    ...Object.keys(d.base_qwen || {}),
    ...Object.keys(d.gpt4o_mini || {}),
    ...Object.keys(d.qwen3_8b || {}),
  ])].sort();

  const labels = domains.map(dom => dom.replace('_', ' ').toUpperCase());

  renderGroupedBar('chart-by-domain', labels, [
    {
      label: 'DataSage LoRA',
      data: domains.map(dom => d.datasage?.[dom] ?? 0),
      backgroundColor: COLORS.answering + 'CC',
      borderColor: COLORS.answering,
      borderWidth: 1,
    },
    {
      label: 'Base Qwen2.5-3B',
      data: domains.map(dom => d.base_qwen?.[dom] ?? 0),
      backgroundColor: COLORS.purple + '88',
      borderColor: COLORS.purple,
      borderWidth: 1,
    },
    {
      label: 'GPT-4o-mini',
      data: domains.map(dom => d.gpt4o_mini?.[dom] ?? 0),
      backgroundColor: COLORS.gpt4o + '88',
      borderColor: COLORS.gpt4o,
      borderWidth: 1,
    },
    {
      label: 'Qwen3-8B',
      data: domains.map(dom => d.qwen3_8b?.[dom] ?? 0),
      backgroundColor: COLORS.cleaning + '88',
      borderColor: COLORS.cleaning,
      borderWidth: 1,
    },
  ], 'Answering Reward');
}

function renderByPersona() {
  const p = DATA.perPersona;
  const personas = [...new Set([
    ...Object.keys(p.datasage || {}),
    ...Object.keys(p.base_qwen || {}),
    ...Object.keys(p.gpt4o_mini || {}),
    ...Object.keys(p.qwen3_8b || {}),
  ])].sort();

  renderGroupedBar('chart-by-persona', personas, [
    {
      label: 'DataSage LoRA',
      data: personas.map(per => p.datasage?.[per] ?? 0),
      backgroundColor: COLORS.answering + 'CC',
      borderColor: COLORS.answering,
      borderWidth: 1,
    },
    {
      label: 'Base Qwen2.5-3B',
      data: personas.map(per => p.base_qwen?.[per] ?? 0),
      backgroundColor: COLORS.purple + '88',
      borderColor: COLORS.purple,
      borderWidth: 1,
    },
    {
      label: 'GPT-4o-mini',
      data: personas.map(per => p.gpt4o_mini?.[per] ?? 0),
      backgroundColor: COLORS.gpt4o + '88',
      borderColor: COLORS.gpt4o,
      borderWidth: 1,
    },
    {
      label: 'Qwen3-8B',
      data: personas.map(per => p.qwen3_8b?.[per] ?? 0),
      backgroundColor: COLORS.cleaning + '88',
      borderColor: COLORS.cleaning,
      borderWidth: 1,
    },
  ], 'Answering Reward');
}
