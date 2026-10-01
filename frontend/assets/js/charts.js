/* Dependency-free SVG charts: bars, columns/trend, per-class bars, confusion heatmap, rings. */
window.Charts = (function () {
  const NS = 'http://www.w3.org/2000/svg';
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function el(tag, attrs, text) {
    const node = document.createElementNS(NS, tag);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function mount(target, svg) {
    target.innerHTML = '';
    target.appendChild(svg);
    return svg;
  }

  function empty(target, message) {
    target.innerHTML = '<p class="fine">' + esc(message) + '</p>';
  }

  /* ── horizontal bars ───────────────────────────────────────────── */
  function bars(target, data, opts) {
    opts = opts || {};
    const rows = (data || []).filter((d) => d && d.value !== null && d.value !== undefined);
    if (!rows.length) return empty(target, opts.emptyMessage || 'No data yet.');
    const labelW = opts.labelWidth || 130, barH = 22, gap = 12, pad = 6;
    const width = 560, height = rows.length * (barH + gap) + pad * 2;
    const max = Math.max(1, ...rows.map((d) => d.value));
    const svg = el('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img' });
    rows.forEach((row, i) => {
      const y = pad + i * (barH + gap);
      const w = Math.max(2, ((width - labelW - 56) * row.value) / max);
      svg.appendChild(el('text', { x: 0, y: y + barH * 0.72, class: 'bar-label' },
        String(row.label).slice(0, 20)));
      const rect = el('rect', {
        x: labelW, y, width: 0, height: barH, rx: 6,
        fill: row.color || 'url(#barGrad)',
      });
      const defs = el('defs');
      defs.innerHTML = '<linearGradient id="barGrad" x1="0" y1="0" x2="1" y2="0">' +
        '<stop offset="0%" stop-color="#22d3ee"/><stop offset="100%" stop-color="#a3e635"/></linearGradient>';
      svg.insertBefore(defs, svg.firstChild);
      svg.appendChild(rect);
      svg.appendChild(el('text', {
        x: labelW + w + 8, y: y + barH * 0.72, class: 'val',
      }, opts.format ? opts.format(row.value) : String(row.value)));
      requestAnimationFrame(() => rect.setAttribute('width', w));
    });
    return mount(target, svg);
  }

  /* ── stacked column / trend chart ──────────────────────────────── */
  function trend(target, rows, opts) {
    opts = opts || {};
    if (!rows || !rows.length) return empty(target, 'No activity recorded yet.');
    const width = 560, height = 220, padL = 34, padB = 26, padT = 12, padR = 8;
    const series = opts.series || [
      { key: 'identified', label: 'Identified', color: '#22d3ee' },
      { key: 'unknown', label: 'Unknown', color: '#ff8a3d' },
    ];
    const max = Math.max(1, ...rows.map((r) => series.reduce((s, k) => s + (r[k.key] || 0), 0)));
    const innerW = width - padL - padR, innerH = height - padT - padB;
    const labelEvery = opts.labelEvery || Math.max(1, Math.ceil(rows.length / 6));
    const step = innerW / rows.length;
    const bw = Math.min(26, step * 0.55);
    const svg = el('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img' });
    for (let g = 0; g <= 4; g++) {
      const y = padT + (innerH * g) / 4;
      svg.appendChild(el('line', { x1: padL, x2: width - padR, y1: y, y2: y, class: 'grid-line' }));
      svg.appendChild(el('text', { x: 4, y: y + 4 }, String(Math.round(max - (max * g) / 4))));
    }
    rows.forEach((row, i) => {
      const x = padL + i * step + (step - bw) / 2;
      let acc = 0;
      series.forEach((s, si) => {
        const value = row[s.key] || 0;
        if (!value) return;
        const h = (innerH * value) / max;
        const rect = el('rect', {
          x, y: padT + innerH - acc - h, width: bw, height: 0, rx: 3,
          fill: s.color, opacity: si === 0 ? 0.95 : 0.75,
        });
        svg.appendChild(rect);
        requestAnimationFrame(() => {
          rect.setAttribute('height', h);
          rect.setAttribute('y', padT + innerH - acc - h);
        });
        acc += h;
      });
      if (i % labelEvery === 0 || i === rows.length - 1) {
        svg.appendChild(el('text', {
          x: x + bw / 2, y: height - 8, 'text-anchor': 'middle', class: 'bar-label',
        }, row.label));
      }
    });
    return mount(target, svg);
  }

  /* ── confusion-matrix heatmap ─────────────────────────────────── */
  function confusion(target, payload, opts) {
    opts = opts || {};
    const labels = (payload && payload.labels) || [];
    const matrix = (payload && payload.matrix) || [];
    if (!labels.length || !matrix.length) return empty(target, 'Train a model to see the confusion matrix.');
    const cell = 46, padL = 96, padT = 54;
    const width = padL + labels.length * cell + 12;
    const height = padT + labels.length * cell + 10;
    const max = Math.max(1, ...matrix.flat().map((v) => Number(v) || 0));
    const svg = el('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img' });
    labels.forEach((name, j) => {
      const t = el('text', { x: padL + j * cell + cell / 2, y: padT - 34, 'text-anchor': 'start',
        transform: `rotate(35 ${padL + j * cell + cell / 2} ${padT - 34})`, class: 'bar-label' },
        name.slice(0, 14));
      svg.appendChild(t);
    });
    labels.forEach((name, i) => {
      svg.appendChild(el('text', { x: padL - 10, y: padT + i * cell + cell / 2 + 4,
        'text-anchor': 'end', class: 'bar-label' }, name.slice(0, 16)));
      labels.forEach((_, j) => {
        const value = Number(matrix[i][j]) || 0;
        const intensity = value / max;
        const good = i === j;
        const color = good
          ? `rgba(74,222,128,${0.14 + intensity * 0.7})`
          : `rgba(251,113,133,${0.05 + intensity * 0.75})`;
        svg.appendChild(el('rect', {
          x: padL + j * cell, y: padT + i * cell, width: cell - 4, height: cell - 4, rx: 7, fill: color,
        }));
        svg.appendChild(el('text', {
          x: padL + j * cell + (cell - 4) / 2, y: padT + i * cell + cell / 2 + 4,
          'text-anchor': 'middle', class: 'cm-cell', fill: value ? '#eaf1ff' : 'rgba(148,193,255,.35)',
        }, String(value)));
      });
    });
    svg.appendChild(el('text', { x: padL - 84, y: 14, class: 'bar-label' }, 'true ↓ / predicted →'));
    return mount(target, svg);
  }

  /* ── confidence ring (used in the result panel) ────────────────── */
  function ring(percent, size = 88, label) {
    const r = (size - 12) / 2, c = 2 * Math.PI * r;
    const pct = Math.max(0, Math.min(1, percent || 0));
    const stroke = pct >= 0.55 ? '#a3e635' : pct >= 0.35 ? '#f5c451' : '#fb7185';
    return `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
      <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="rgba(148,193,255,.18)" stroke-width="8"/>
      <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="${stroke}" stroke-width="8"
        stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${c * (1 - pct)}"
        style="transition:stroke-dashoffset .9s cubic-bezier(.2,.8,.2,1)"/>
    </svg>`;
  }

  /* ── per-player F1 bars ────────────────────────────────────────── */
  function perClass(target, rows) {
    const data = (rows || []).map((r) => ({
      label: r.class, value: r.f1 === null || r.f1 === undefined ? 0 : r.f1,
    }));
    if (!data.length) return empty(target, 'No per-player metrics yet.');
    return bars(target, data, {
      labelWidth: 120,
      format: (v) => v.toFixed(2),
      emptyMessage: 'No per-player metrics yet.',
    });
  }

  return { bars, trend, confusion, ring, perClass };
})();
