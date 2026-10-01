/* ============================================================================
   INDIAN CRICKET FACE ID · frontend controller
   Hash router + view renderers. All data comes from the FastAPI backend.
   ========================================================================== */
(function () {
  'use strict';

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const state = {
    config: null,
    status: null,
    players: [],
    dataset: null,
    filter: 'all',
    query: '',
    lastResult: null,
    trainJob: null,
    trainTimer: null,
  };

  /* ── small helpers ─────────────────────────────────────────────── */
  const esc = (s) => String(s === null || s === undefined ? '' : s)
    .replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const pct = (v, digits = 1) => (v === null || v === undefined || isNaN(v))
    ? '—' : (v * 100).toFixed(digits) + '%';
  const num = (v) => (v === null || v === undefined) ? '—' : String(v);
  const relTime = (iso) => {
    if (!iso) return '—';
    const t = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : iso + 'Z');
    const diff = (Date.now() - t.getTime()) / 1000;
    if (isNaN(diff)) return iso;
    if (diff < 60) return 'just now';
    if (diff < 3600) return Math.round(diff / 60) + ' min ago';
    if (diff < 86400) return Math.round(diff / 3600) + ' h ago';
    return t.toLocaleDateString() + ' ' + t.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  const initials = (name) => (name || '?').replace(/[^A-Za-z ]/g, '').trim().split(/\s+/)
    .slice(0, 2).map((w) => w[0] || '').join('').toUpperCase();
  const monogram = (name) =>
    'data:image/svg+xml;utf8,' + encodeURIComponent(
      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#1e6fff"/><stop offset="100%" stop-color="#22d3ee"/></linearGradient></defs><rect width="200" height="200" rx="26" fill="url(#g)"/><text x="100" y="128" text-anchor="middle" font-family="Inter,Segoe UI,sans-serif" font-size="76" font-weight="700" fill="rgba(255,255,255,.95)">${initials(name)}</text></svg>`);

  function toast(message, kind = 'ok', ms = 4200) {
    const stack = $('#toasts');
    const node = document.createElement('div');
    node.className = 'toast ' + kind;
    node.innerHTML = esc(message);
    stack.appendChild(node);
    setTimeout(() => { node.style.opacity = '0'; node.style.transition = 'opacity .4s'; }, ms - 450);
    setTimeout(() => node.remove(), ms);
  }

  /* ── router ────────────────────────────────────────────────────── */
  const VIEWS = ['landing', 'identify', 'players', 'profile', 'dashboard', 'model', 'dataset', 'about'];

  function currentRoute() {
    const raw = (location.hash || '#/').replace(/^#\/?/, '');
    const [name, param] = raw.split('/');
    return { name: name || 'landing', param: param || '' };
  }

  function showView(name) {
    VIEWS.forEach((v) => {
      const node = $('#view-' + v);
      if (node) node.classList.toggle('hidden', v !== name);
    });
    const navKey = name === 'profile' ? 'players' : name;
    $$('#nav a').forEach((a) => a.classList.toggle('active', a.dataset.nav === navKey));
    window.scrollTo({ top: 0, behavior: 'smooth' });
    $('#nav').classList.remove('open');
  }

  async function route() {
    const { name, param } = currentRoute();
    showView(name);
    try {
      if (name === 'landing') await renderLanding();
      if (name === 'identify') await renderIdentify();
      if (name === 'players') await renderPlayers();
      if (name === 'profile') await renderProfile(param);
      if (name === 'dashboard') await renderDashboard();
      if (name === 'model') await renderModel();
      if (name === 'dataset') await renderDataset();
      if (name === 'about') await renderAbout();
    } catch (error) {
      console.error(error);
      toast(error.message || 'Something went wrong', 'err');
    }
  }

  /* ── global chrome ─────────────────────────────────────────────── */
  function setModelPill(status) {
    const pill = $('#pill-model');
    const span = $('span', pill);
    pill.classList.remove('ok', 'warn', 'demo');
    if (!status.ready) {
      pill.classList.add('warn');
      span.textContent = 'no model';
      pill.title = status.error || 'No trained model';
    } else if (status.mode === 'demo') {
      pill.classList.add('demo');
      span.textContent = 'demo model';
      pill.title = 'Active model was trained on the fictional demo personas';
    } else {
      pill.classList.add('ok');
      span.textContent = (status.engine && status.engine.n_classes ? status.engine.n_classes + ' players' : 'model ready');
      pill.title = status.active_model || '';
    }
  }

  function renderBanners() {
    const warn = $('#banner-warn');
    const demo = $('#banner-demo');
    const ds = state.dataset || {};
    const configured = (ds.dataset && ds.dataset.configured);
    const demoConfigured = (ds.demo_dataset && ds.demo_dataset.configured);

    if (!configured) {
      warn.classList.remove('hidden');
      warn.innerHTML = '<b>Dataset not configured.</b> The model can only recognise players whose '
        + 'photos you add to <code>dataset/&lt;Player_Name&gt;/</code>. '
        + (demoConfigured
          ? 'A demonstration model trained on <b>fictional, AI-generated</b> personas is active so you can explore the app — no real player is recognised.'
          : 'Train after adding images: <code>python -m ml.train --dataset dataset</code>.');
    } else {
      warn.classList.add('hidden');
    }

    if (state.status && state.status.mode === 'demo' && configured) {
      demo.classList.remove('hidden');
      demo.innerHTML = '<b>Demo model active.</b> The loaded model was trained on the fictional demo '
        + 'personas. Train on your own dataset (<code>python -m ml.train --dataset dataset</code>) or use '
        + 'the Model page to switch.';
    } else if (state.status && state.status.mode === 'demo') {
      demo.classList.remove('hidden');
      demo.innerHTML = '<b>Demo mode.</b> Every face in the demo dataset was generated by an AI image '
        + 'model — these people do not exist. Replace <code>dataset_demo/</code> with your own '
        + '<code>dataset/</code> images to identify real players.';
    } else {
      demo.classList.add('hidden');
    }
  }

  async function refreshGlobal() {
    const [status, dataset] = await Promise.all([API.modelStatus(), API.dataset()]);
    state.status = status;
    state.dataset = dataset;
    setModelPill(status);
    renderBanners();
    const engine = status.engine || {};
    $('#pill-threshold').textContent = 'threshold ' + pct(status.engine && status.engine.confidence_threshold);
    $('#pill-engine').textContent = (engine.feature_extractor || '—') + ' + ' + (engine.detector || '—');
    $('#pill-classes').textContent = (engine.n_classes || 0) + ' classes';
    return status;
  }

  /* ── landing ───────────────────────────────────────────────────── */
  let pipelineTimer = null;

  function animatePipeline() {
    const steps = $$('#pipeline-flow .pipe-step');
    if (!steps.length) return;
    let i = 0;
    clearInterval(pipelineTimer);
    const advance = () => {
      steps.forEach((s, idx) => s.classList.toggle('lit', idx === i));
      i = (i + 1) % steps.length;
    };
    advance();
    pipelineTimer = setInterval(advance, 1400);
  }

  async function renderLanding() {
    animatePipeline();
    const stats = $('#hero-stats');
    const status = state.status || await refreshGlobal();
    const dataset = state.dataset || await API.dataset();
    const active = (dataset.dataset && dataset.dataset.configured) ? dataset.dataset : dataset.demo_dataset;
    const engine = status.engine || {};

    stats.innerHTML = [
      ['players', engine.n_classes || 0, 'in the active model'],
      ['images', active.total_images || 0, 'in the training dataset'],
      ['engine', (engine.feature_extractor || '—').toUpperCase(), (engine.detector || '—') + ' detector'],
      ['threshold', pct(status.engine && status.engine.confidence_threshold), 'below this → Unknown'],
    ].map(([label, value, note]) => `
      <div class="hero-stat"><b>${esc(value)}</b><span>${esc(label)}</span>
      <i class="fine" style="display:block;margin:0">${esc(note)}</i></div>`).join('');

    const grid = $('#landing-players');
    try {
      const payload = await API.players();
      state.players = payload.players;
      const list = payload.players.filter((p) => p.supported).slice(0, 4);
      $('#landing-supported-note').textContent = list.length
        ? `The active model can recognise ${payload.players.filter((p) => p.supported).length} person(s). `
          + 'Everyone else is reported as Unknown.'
        : 'No model is trained yet — add images and run the trainer to make players recognisable.';
      grid.innerHTML = list.length
        ? list.map(playerCard).join('')
        : `<div class="card glass"><h3>Dataset not configured</h3>
             <p>Add <code>dataset/&lt;Player_Name&gt;/*.jpg</code> then run
             <code>python -m ml.train --dataset dataset</code>, or explore the built-in demo.</p>
             <a class="btn btn-ghost btn-sm" href="#/dataset">Dataset instructions</a></div>`;
      wirePlayerCards(grid);
    } catch (error) {
      grid.innerHTML = '';
      toast(error.message, 'err');
    }
  }

  /* ── player cards ──────────────────────────────────────────────── */
  function playerCard(p) {
    const badge = p.fictional
      ? '<span class="badge badge-fiction">fictional demo</span>'
      : (p.supported
        ? '<span class="badge badge-supported">recognised by model</span>'
        : '<span class="badge badge-idle">no trained images</span>');
    return `
      <article class="card glass pcard" data-player="${esc(p.id)}" tabindex="0" role="button">
        <div class="pcard-media">
          <img src="${esc(p.avatar_url)}" alt="${esc(p.name)}" loading="lazy"
               onerror="this.src='${monogram(p.name)}'"/>
          ${p.jersey_number ? `<span class="jersey">#${esc(p.jersey_number)}</span>` : ''}
          <span class="flagline"></span>
        </div>
        <div class="pcard-body">
          <h3>${esc(p.name)}</h3>
          <p class="pcard-role">${esc(p.role || '—')}</p>
          <div class="pcard-meta">${badge}
            <span class="badge">${esc(p.dataset_images)} image${p.dataset_images === 1 ? '' : 's'}</span></div>
          <div class="pcard-foot">
            <span>${esc((p.batting_style || '').split(' ')[0] || '—')} bat</span>
            <span>${p.model && p.model.f1 !== null && p.model.f1 !== undefined ? 'F1 ' + p.model.f1.toFixed(2) : 'not in model'}</span>
          </div>
        </div>
      </article>`;
  }

  function wirePlayerCards(root) {
    $$('[data-player]', root).forEach((card) => {
      const go = () => { location.hash = '#/player/' + card.dataset.player; };
      card.addEventListener('click', go);
      card.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
    });
  }

  async function renderPlayers() {
    const grid = $('#players-grid');
    if (!state.players.length) {
      grid.innerHTML = '<div class="card glass skeleton" style="height:200px"></div>'.repeat(4);
      const payload = await API.players();
      state.players = payload.players;
      $('#players-sub').textContent = payload.supported
        ? `${payload.players.filter((p) => p.supported).length} of ${payload.count} profiles are supported by the active model.`
        : 'No model is trained yet — profiles above are catalog information only.';
    }
    paintPlayers();
  }

  function paintPlayers() {
    const grid = $('#players-grid');
    const q = state.query.toLowerCase();
    const rows = state.players.filter((p) => {
      const matches = !q || p.name.toLowerCase().includes(q) || (p.role || '').toLowerCase().includes(q);
      if (!matches) return false;
      if (state.filter === 'supported') return p.supported;
      if (state.filter === 'dataset') return p.dataset_images > 0;
      return true;
    });
    grid.innerHTML = rows.length ? rows.map(playerCard).join('')
      : '<div class="card glass"><h3>No match</h3><p>Try another name or clear the filters.</p></div>';
    wirePlayerCards(grid);
  }

  /* ── profile ───────────────────────────────────────────────────── */
  async function renderProfile(id) {
    const body = $('#profile-body');
    if (!id) { body.innerHTML = '<div class="card glass"><h3>No player selected</h3></div>'; return; }
    body.innerHTML = '<div class="card glass skeleton" style="height:340px"></div>';
    const p = await API.player(id);
    const m = p.model || {};
    body.innerHTML = `
      <div class="card glass panel">
        <div class="profile-head">
          <div class="profile-face"><img src="${esc(p.avatar_url)}" alt="${esc(p.name)}"
              onerror="this.src='${monogram(p.name)}'"/></div>
          <div>
            <span class="eyebrow">${p.fictional ? 'fictional demo persona' : 'player profile'}</span>
            <h2 style="margin-top:.6rem">${esc(p.name)}</h2>
            <p class="lede" style="font-size:.95rem">${esc(p.role || '—')}</p>
            <div class="badges">
              ${p.supported ? '<span class="badge ok">recognised by the active model</span>'
                            : '<span class="badge warn">not in the trained model</span>'}
              <span class="badge">${esc(p.dataset_images)} dataset image(s)</span>
              ${p.dataset_source !== 'none' ? `<span class="badge">${esc(p.dataset_source)} dataset</span>` : ''}
              ${p.fictional ? '<span class="badge badge-fiction">AI-generated person (does not exist)</span>' : ''}
            </div>
            <div class="profile-facts">
              <div class="fact"><span>Batting</span><b>${esc(p.batting_style || '—')}</b></div>
              <div class="fact"><span>Bowling</span><b>${esc(p.bowling_style || '—')}</b></div>
              <div class="fact"><span>Jersey</span><b>${p.jersey_number ? '#' + esc(p.jersey_number) : '—'}</b></div>
              <div class="fact"><span>Born</span><b>${esc(p.born || '—')}</b></div>
            </div>
          </div>
        </div>
      </div>

      <div class="grid grid-2">
        <div class="card glass panel">
          <h3>Biography</h3>
          <p>${esc(p.bio || 'No biography stored for this profile.')}</p>
          <h3 style="margin-top:1.1rem">Career highlights</h3>
          <ul class="highlights">
            ${(p.career_highlights || []).map((h) => `<li>${esc(h)}</li>`).join('') || '<li>—</li>'}
          </ul>
          <p class="fine">${esc(p.source_note || '')}</p>
        </div>

        <div class="card glass panel">
          <h3>Model recognition information</h3>
          <p class="fine">${esc(m.note || '')}</p>
          <div class="gauge-grid">
            <div class="gauge"><b>${num(m.dataset_images)}</b><span>dataset images</span></div>
            <div class="gauge"><b>${num(m.augmented_variants)}</b><span>augmented variants</span></div>
            <div class="gauge"><b>${num(m.training_vectors)}</b><span>training vectors</span></div>
            <div class="gauge"><b>${num(m.validation_support)}</b><span>validation samples</span></div>
            <div class="gauge"><b>${m.precision === null || m.precision === undefined ? '—' : m.precision.toFixed(3)}</b><span>precision</span></div>
            <div class="gauge"><b>${m.recall === null || m.recall === undefined ? '—' : m.recall.toFixed(3)}</b><span>recall</span></div>
            <div class="gauge"><b>${m.f1 === null || m.f1 === undefined ? '—' : m.f1.toFixed(3)}</b><span>F1</span></div>
          </div>
          <p class="fine">Metrics are per-player, measured on the held-out split of this dataset only.
            They do not describe behaviour on photographs outside the dataset.</p>
          ${p.sample_images && p.sample_images.length ? `
            <h3 style="margin-top:1rem">Dataset samples</h3>
            <div class="samples">${p.sample_images.map((src) =>
              `<img src="${esc(src)}" alt="dataset sample" loading="lazy"/>`).join('')}</div>` : ''}
        </div>
      </div>

      <div class="card glass panel">
        <h3>Try it</h3>
        <p>Run this profile through the identification pipeline or open the documentation.</p>
        <div class="hero-actions">
          <a class="btn btn-primary" href="#/identify">Identify a player</a>
          <a class="btn btn-ghost" href="#/model">Model metrics</a>
          <a class="btn btn-ghost" href="/docs" target="_blank" rel="noopener">API docs</a>
        </div>
      </div>`;
  }

  /* ── identify ──────────────────────────────────────────────────── */
  let pendingFile = null;
  let pendingCapture = null;

  async function renderIdentify() {
    await refreshGlobal();
    const status = state.status;
    const notice = $('#dataset-notice');
    const ds = state.dataset || {};
    if (!status.ready || !(ds.dataset && ds.dataset.configured) && !(ds.demo_dataset && ds.demo_dataset.configured)) {
      notice.hidden = false;
      notice.innerHTML = `
        <h3>⚠️ Dataset not configured</h3>
        <p>The model has nothing to learn from yet, so it will not invent an answer.</p>
        <div class="code-block">dataset/
  Virat_Kohli/    image1.jpg  image2.jpg …
  Rohit_Sharma/   image1.jpg …
  Jasprit_Bumrah/ …</div>
        <div class="code-block">python scripts/prepare_dataset.py --init --audit
python -m ml.train --dataset dataset
# or use the Model page to start training from the browser</div>
        <p class="fine">Only legally usable images. The app never downloads player photographs for you.</p>`;
    } else {
      notice.hidden = true;
    }
  }

  function setMode(mode) {
    $$('.tab').forEach((t) => t.classList.toggle('active', t.dataset.mode === mode));
    $('#mode-upload').classList.toggle('hidden', mode !== 'upload');
    $('#mode-camera').classList.toggle('hidden', mode !== 'camera');
    if (mode !== 'camera') {
      Camera.stop($('#cam-video'));
      $('#btn-cam-start').disabled = false;
      $('#btn-cam-capture').disabled = true;
      $('#btn-cam-stop').disabled = true;
    }
  }

  function handleFile(file) {
    if (!file) return;
    const ok = /image\/(jpeg|jpg|png|webp|bmp)/.test(file.type) ||
      /\.(jpe?g|png|webp|bmp)$/i.test(file.name);
    if (!ok) { toast('Unsupported file type. Use JPG, JPEG, PNG or WEBP.', 'err'); return; }
    if (file.size > 12 * 1024 * 1024) { toast('That image is larger than 12 MB.', 'err'); return; }
    pendingFile = file;
    const row = $('#upload-preview-row');
    row.classList.remove('hidden');
    $('#upload-preview').src = URL.createObjectURL(file);
    $('#upload-name').textContent = file.name;
    $('#upload-size').textContent = (file.size / 1024).toFixed(0) + ' KB · ' + (file.type || 'image');
  }

  const PROC_STEPS = ['Decoding image', 'Detecting face', 'Aligning & normalising',
    'Encoding descriptor', 'Classifying'];
  let procTimer = null;

  function startProcessing(title) {
    const box = $('#processing');
    box.classList.remove('hidden');
    $('#proc-title').textContent = title;
    const items = $$('#proc-list li');
    items.forEach((li, i) => {
      li.textContent = PROC_STEPS[i];
      li.className = i === 0 ? 'active' : '';
    });
    let step = 0;
    clearInterval(procTimer);
    procTimer = setInterval(() => {
      if (step < items.length - 1) {
        items[step].className = 'done';
        step += 1;
        items[step].className = 'active';
      }
    }, 320);
  }

  function stopProcessing() {
    clearInterval(procTimer);
    $$('#proc-list li').forEach((li) => { li.className = 'done'; });
    setTimeout(() => $('#processing').classList.add('hidden'), 220);
  }

  async function analyze(action, title) {
    startProcessing(title);
    try {
      const result = await action();
      state.lastResult = result;
      stopProcessing();
      renderResult(result);
      if (result.status === 'identified') toast('Identified: ' + result.player.name, 'ok');
      else if (result.status === 'unknown') toast('Unknown — below threshold, no name returned.', 'warn');
      else if (result.status === 'no_face') toast('No face detected in that image.', 'warn');
      else if (result.status === 'model_missing') toast('No trained model yet — see the Dataset page.', 'err');
      if (state.status) refreshGlobal().catch(() => {});
    } catch (error) {
      stopProcessing();
      toast(error.message, 'err');
      $('#result-empty').classList.remove('hidden');
      $('#result-body').classList.add('hidden');
    }
  }

  function renderResult(r) {
    const empty = $('#result-empty');
    const body = $('#result-body');
    empty.classList.add('hidden');
    body.classList.remove('hidden');

    if (r.status === 'model_missing' || r.status === 'invalid_image') {
      body.innerHTML = `
        <div class="notice">${esc(r.message || 'The model is not available.')}</div>
        <h3 style="margin-top:1rem">Dataset not configured</h3>
        <p>Add images to <code>dataset/&lt;Player_Name&gt;/</code> and train the model — the app never
           fabricates a prediction.</p>
        <a class="btn btn-ghost btn-sm" href="#/dataset">Dataset instructions</a>`;
      return;
    }

    const face = r.detected_face || {};
    const engine = r.engine || {};
    const identified = r.status === 'identified';
    const confidence = r.confidence || 0;
    const player = r.player;
    const timings = r.timings_ms || {};

    body.innerHTML = `
      <div class="result-head">
        ${face.image ? `<img class="result-face" src="${face.image}" alt="Detected face crop"/>`
                     : `<div class="result-face" style="display:grid;place-items:center;background:#0b1220;color:#7c8dab">no face</div>`}
        <div class="result-main">
          <span class="verdict">${identified ? 'Player identified' : (r.status === 'no_face' ? 'No face detected' : 'Not identified')}</span>
          <div class="result-name ${identified ? '' : 'miss'}">
            ${identified ? esc(player.name) : 'Unknown / Low Confidence'}
          </div>
          <div class="${identified ? 'stamp' : 'stamp miss'}">
            ${identified ? '✓ match accepted' : '✕ no confident match'}
          </div>
          <div class="badges">
            ${player && player.role ? `<span class="badge">${esc(player.role)}</span>` : ''}
            ${player && player.jersey_number ? `<span class="badge">#${esc(player.jersey_number)}</span>` : ''}
            <span class="badge">threshold ${pct(r.threshold)}</span>
            <span class="badge">margin ${r.margin === null || r.margin === undefined ? '—' : (r.margin * 100).toFixed(1) + ' pts'}</span>
            <span class="badge">${esc(r.faces_detected)} face(s) detected</span>
          </div>
        </div>
        <div class="conf-row">
          <div class="conf-ring">
            ${Charts.ring(confidence, 88)}
            <span class="val">${pct(confidence)}</span>
          </div>
        </div>
      </div>

      ${r.message ? `<div class="notice">${esc(r.message)}</div>` : ''}
      ${!identified && r.status === 'unknown' ? `
        <p class="fine">The system refuses to name a player below the configured confidence
        threshold${r.reason === 'ambiguous' ? ' or when the top two classes are too close' : ''}.
        That is intentional: a wrong name is worse than "unknown".</p>` : ''}
      ${r.status === 'no_face' ? `
        <p class="fine">No face was detected. Try a clearer, front-facing photo with the face
        occupying a decent share of the frame.</p>` : ''}

      ${r.top_predictions && r.top_predictions.length ? `
        <h3 style="margin-top:1.2rem">Top predictions</h3>
        <div class="pred-list">
          ${r.top_predictions.map((p, i) => `
            <div class="pred-row ${i === 0 ? 'top' : ''}">
              <span class="rank">${i + 1}</span>
              <span class="nm">${esc(p.display_name)}</span>
              <span class="pc">${pct(p.confidence)}</span>
              <span class="meter"><i style="width:${(p.confidence * 100).toFixed(1)}%"></i></span>
            </div>`).join('')}
        </div>` : ''}

      <div class="stat-strip">
        <span class="mini-pill">detector · ${esc(engine.detector || '—')}</span>
        <span class="mini-pill">extractor · ${esc(engine.feature_extractor || '—')}</span>
        <span class="mini-pill">classifier · ${esc(engine.classifier || '—')}</span>
        <span class="mini-pill">model · ${esc(engine.model_version || '—')}</span>
        <span class="mini-pill">detect ${num(timings.detect)} ms</span>
        <span class="mini-pill">encode ${num(timings.encode)} ms</span>
        <span class="mini-pill">classify ${num(timings.classify)} ms</span>
        <span class="mini-pill">total ${num(timings.total)} ms</span>
      </div>

      ${identified && player ? `
        <div class="hero-actions" style="margin-top:1.1rem">
          <a class="btn btn-primary btn-sm" href="#/player/${esc(player.id)}">Open player profile</a>
          <button class="btn btn-ghost btn-sm" id="btn-again">Analyze another image</button>
        </div>` : `
        <div class="hero-actions" style="margin-top:1.1rem">
          <button class="btn btn-ghost btn-sm" id="btn-again">Try another image</button>
          <a class="btn btn-ghost btn-sm" href="#/dataset">Add more dataset images</a>
        </div>`}

      <p class="fine">${esc(r.disclaimer || '')}</p>`;

    const again = $('#btn-again');
    if (again) again.addEventListener('click', () => {
      pendingFile = null;
      pendingCapture = null;
      $('#upload-preview-row').classList.add('hidden');
      $('#file-input').value = '';
      body.classList.add('hidden');
      $('#result-empty').classList.remove('hidden');
    });
  }

  /* ── dashboard ─────────────────────────────────────────────────── */
  async function renderDashboard() {
    const cards = $('#stat-cards');
    cards.innerHTML = '<div class="card glass skeleton" style="height:120px"></div>'.repeat(4);
    const d = await API.dashboard();

    cards.innerHTML = `
      ${statCard('Total players', d.total_players, `${d.supported_players} recognised by the model`)}
      ${statCard('Total images', d.total_images, d.dataset_configured ? 'in dataset/' : 'demo dataset')}
      ${statCard('Model accuracy', d.model_accuracy === null ? '—' : pct(d.model_accuracy),
                 'held-out split of your dataset')}
      ${statCard('Predictions today', d.predictions_today,
                 `${d.identified_today} identified · ${d.unknown_today} unknown`)}
      ${statCard('All-time attempts', d.predictions_total, 'stored in the predictions table')}
      ${statCard('Avg confidence', d.avg_confidence === null ? '—' : pct(d.avg_confidence), 'today')}
      ${statCard('Avg latency', d.avg_latency_ms === null ? '—' : d.avg_latency_ms + ' ms',
                 'detect + encode + classify')}
      ${statCard('Engine', (d.model.feature_extractor || '—').toUpperCase(),
                 (d.model.detector || '—') + ' detector')}`;

    $('#performance-note').textContent = d.model_accuracy_note || '';

    Charts.bars($('#chart-player-dist'), d.prediction_distribution, { labelWidth: 118 });
    Charts.bars($('#chart-confidence'), d.confidence_distribution, { labelWidth: 90 });
    Charts.perClass($('#chart-performance'), (d.performance && d.performance.per_class) || []);
    Charts.trend($('#chart-trend'), d.daily_trend, { labelEvery: 2 });

    const tbody = $('#recent-table tbody');
    tbody.innerHTML = d.recent.length ? d.recent.map((row) => `
      <tr>
        <td>${row.face_image ? `<img class="crop" src="${esc(row.face_image)}" alt="face"/>` : '<span class="mono">—</span>'}</td>
        <td><b>${esc(row.predicted_name || '—')}</b></td>
        <td class="mono">${row.confidence === null ? '—' : pct(row.confidence)}</td>
        <td>${row.status === 'identified' ? '<span class="tag ok">identified</span>'
              : row.status === 'unknown' ? '<span class="tag miss">unknown</span>'
              : '<span class="tag info">' + esc(row.status) + '</span>'}</td>
        <td class="mono">${row.latency_ms ? Math.round(row.latency_ms) + ' ms' : '—'}</td>
        <td class="mono">${esc(relTime(row.created_at))}</td>
      </tr>`).join('')
      : '<tr><td colspan="6" class="empty-row">No predictions yet — try the Identify page.</td></tr>';
  }

  function statCard(label, value, note) {
    return `<div class="card glass stat"><span>${esc(label)}</span><b>${esc(value)}</b><i>${esc(note)}</i></div>`;
  }

  /* ── model page ────────────────────────────────────────────────── */
  async function renderModel() {
    const body = $('#model-body');
    body.innerHTML = '<div class="card glass skeleton" style="height:300px"></div>';
    const [metrics, status, versions] = await Promise.all([
      API.modelMetrics(), API.modelStatus(), API.modelVersions().catch(() => ({ versions: [] })),
    ]);

    if (!metrics.ready) {
      body.innerHTML = `
        <div class="card glass panel">
          <h3>No trained model</h3>
          <p>${esc(metrics.message || '')}</p>
          <p class="fine">Train one below, with the CLI (<code>python -m ml.train --dataset dataset</code>),
             or place the bundled demo model at <code>models/demo_model.pkl</code>.</p>
        </div>`;
    } else {
      const t = metrics.training || {};
      const e = metrics.evaluation || {};
      const cv = metrics.cross_validation || {};
      body.innerHTML = `
        <div class="grid grid-4">
          ${statCard('Accuracy (held-out)', e.accuracy === undefined || e.accuracy === null ? '—' : pct(e.accuracy), 'same-source split')}
          ${statCard('Precision (macro)', e.precision_macro === undefined ? '—' : pct(e.precision_macro), 'per-class average')}
          ${statCard('Recall (macro)', e.recall_macro === undefined ? '—' : pct(e.recall_macro), 'per-class average')}
          ${statCard('F1 (macro)', e.f1_macro === undefined ? '—' : pct(e.f1_macro), 'balanced score')}
          ${statCard('Top-3 accuracy', e.top3_accuracy === undefined ? '—' : pct(e.top3_accuracy), 'correct in top 3')}
          ${statCard('Cross-validation', cv.accuracy_mean === undefined ? '—' : pct(cv.accuracy_mean),
                     cv.folds ? cv.folds + '-fold ± ' + pct(cv.accuracy_std) : 'not run')}
          ${statCard('Training samples', num(t.training_samples), `${num(t.validation_samples)} held out`)}
          ${statCard('Classes · features', `${num(t.n_classes)} · ${num(t.feature_dim)}`,
                     `${esc(t.feature_extractor || '—')} + ${esc(t.detector || '—')}`)}
        </div>

        <div class="card glass panel">
          <div class="panel-head">
            <h3>Confusion matrix</h3>
            <span class="chip-soft">rows = truth · columns = prediction</span>
          </div>
          <div class="chart" id="chart-confusion"></div>
          <p class="fine">Green diagonal = correct. Off-diagonal cells show which players get
             confused with each other.</p>
        </div>

        <div class="grid grid-2">
          <div class="card glass panel">
            <h3>Per-player metrics</h3>
            <div class="table-wrap">
              <table class="table">
                <thead><tr><th>Player</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
                <tbody>
                  ${(metrics.per_class || []).map((row) => `
                    <tr><td><b>${esc(row.class)}</b></td>
                      <td class="mono">${row.precision.toFixed(3)}</td>
                      <td class="mono">${row.recall.toFixed(3)}</td>
                      <td class="mono">${row.f1.toFixed(3)}</td>
                      <td class="mono">${row.support}</td></tr>`).join('')
                    || '<tr><td colspan="5" class="empty-row">No held-out split was possible.</td></tr>'}
                </tbody>
              </table>
            </div>
          </div>

          <div class="card glass panel">
            <h3>Training setup</h3>
            <div class="gauge-grid">
              <div class="gauge"><b>${esc(t.classifier || '—')}</b><span>classifier</span></div>
              <div class="gauge"><b>${esc(t.feature_extractor || '—')}</b><span>extractor</span></div>
              <div class="gauge"><b>${esc(t.detector || '—')}</b><span>detector</span></div>
              <div class="gauge"><b>${num(t.training_seconds)}s</b><span>fit time</span></div>
              <div class="gauge"><b>${(t.class_samples ? Object.values(t.class_samples).reduce((s, c) => s + (c.augmented || 0), 0) : 0)}</b><span>augmented variants</span></div>
              <div class="gauge"><b>${esc((metrics.model_version || '—'))}</b><span>model version</span></div>
            </div>
            <p class="fine">Trained ${esc(relTime(metrics.trained_at))} · dataset
              <code>${esc((t.dataset && t.dataset.path) || '—')}</code></p>
            <h3 style="margin-top:1rem">Engine</h3>
            <p class="fine">detector: <b>${esc((status.engine && status.engine.detector) || '—')}</b> ·
               extractor: <b>${esc((status.engine && status.engine.feature_extractor) || '—')}</b> ·
               threshold: <b>${pct(status.engine && status.engine.confidence_threshold)}</b> ·
               min margin: <b>${pct(status.engine && status.engine.min_margin)}</b></p>
          </div>
        </div>

        <div class="card glass panel">
          <h3>Read this before trusting the numbers</h3>
          <ul class="checklist">
            ${(metrics.transparency || []).map((line) => `<li>${esc(line)}</li>`).join('')}
            ${(metrics.notes || []).map((line) => `<li>${esc(line)}</li>`).join('')}
          </ul>
        </div>`;
      Charts.confusion($('#chart-confusion'), metrics.confusion);
    }

    const tbody = $('#versions-table tbody');
    const rows = (versions.versions || []);
    tbody.innerHTML = rows.length ? rows.map((v) => `
      <tr>
        <td class="mono">${esc(v.version)} ${v.is_active ? '<span class="tag ok">active</span>' : ''}</td>
        <td>${esc(v.classifier || '—')}</td>
        <td>${esc(v.feature_extractor || '—')}</td>
        <td class="mono">${num(v.n_classes)}</td>
        <td class="mono">${num(v.training_samples)} / ${num(v.validation_samples)}</td>
        <td class="mono">${v.accuracy === null ? '—' : pct(v.accuracy)}</td>
        <td class="mono">${v.f1_macro === null ? '—' : pct(v.f1_macro)}</td>
        <td class="mono">${v.cross_val_accuracy === null ? '—' : pct(v.cross_val_accuracy)}</td>
        <td class="mono">${esc(relTime(v.trained_at || v.created_at))}</td>
      </tr>`).join('') : '<tr><td colspan="9" class="empty-row">No models recorded yet.</td></tr>';

    // prefill the training form
    try {
      const caps = await API.capabilities();
      const select = $('#train-extractor');
      $$('option', select).forEach((opt) => {
        const info = (caps.extractors || []).find((x) => x.name === opt.value);
        if (info && !info.available) {
          opt.disabled = true;
          opt.textContent += ' — unavailable';
        }
      });
      const det = caps.detector || {};
      if (!det.yunet_available) {
        toast('Tip: run python scripts/fetch_models.py for the stronger YuNet detector and SFace embeddings.', 'warn', 6000);
      }
    } catch (e) { /* capabilities are optional */ }
    if (state.status && state.status.engine && state.status.engine.confidence_threshold) {
      $('#train-threshold').value = state.status.engine.confidence_threshold;
    }
  }

  /* ── dataset page ──────────────────────────────────────────────── */
  async function renderDataset() {
    const body = $('#dataset-body');
    body.innerHTML = '<div class="card glass skeleton" style="height:260px"></div>';
    const [ds, players] = await Promise.all([API.dataset(), API.players()]);
    const real = ds.dataset, demo = ds.demo_dataset;

    const datasetCard = (title, info, source) => `
      <div class="card glass panel">
        <div class="panel-head">
          <h3>${esc(title)}</h3>
          <span class="chip-soft">${info.configured ? 'configured' : 'not configured'}</span>
        </div>
        <p class="fine"><code>${esc(info.path)}</code></p>
        <div class="gauge-grid">
          <div class="gauge"><b>${num(info.num_classes)}</b><span>player folders</span></div>
          <div class="gauge"><b>${num(info.total_images)}</b><span>images</span></div>
        </div>
        ${info.num_classes ? `<div class="table-wrap" style="margin-top:.8rem">
          <table class="table"><thead><tr><th>Folder</th><th>Images</th><th></th></tr></thead><tbody>
          ${Object.entries(info.images || {}).map(([name, count]) => `
            <tr><td class="mono">${esc(name)}</td><td class="mono">${count}</td>
            <td>${count >= 8 ? '<span class="tag ok">good</span>'
                 : count >= 4 ? '<span class="tag info">ok</span>'
                 : '<span class="tag miss">add more</span>'}</td></tr>`).join('')}
          </tbody></table></div>` : ''}
        ${source === 'demo' ? '<p class="fine">Fictional, AI-generated personas — safe to ship, useless for real identification.</p>' : ''}
      </div>`;

    body.innerHTML = `
      ${datasetCard('Your dataset (dataset/)', real, 'local')}
      ${datasetCard('Demo dataset (dataset_demo/)', demo, 'demo')}

      <div class="card glass panel">
        <h3>Expected layout</h3>
        <div class="folder-tree">dataset/
├── Virat_Kohli/
│   ├── 01_face.jpg
│   ├── 02_face.jpg
│   └── …
├── Rohit_Sharma/
│   ├── 01.jpg
│   └── …
└── Jasprit_Bumrah/
    └── …</div>
        <p class="fine">Folder name = player label (underscores become spaces in the UI). Folders
           starting with <code>_</code> or <code>.</code> are ignored. 10–30 clear, front-facing
           photos per player is the sweet spot.</p>
      </div>

      <div class="card glass warn-box">
        <h3>⚠️ You are responsible for the images</h3>
        <p>This project deliberately does <b>not</b> download player photographs. Place only images you
           are legally allowed to use — your own photographs, CC-licensed media with attribution,
           or material you have explicit permission to process. Do not scrape news agencies,
           broadcasters or social platforms, and check the personality/publicity rights that apply
           where you are.</p>
      </div>

      <div class="card glass panel">
        <h3>Commands</h3>
        <div class="code-block"># 1 · create the folder tree (optional roster is printed)
python scripts/prepare_dataset.py --init

# 2 · drop your images into dataset/&lt;Player_Name&gt;/ then audit them
python scripts/prepare_dataset.py --audit --min-images 8

# 3 · (optional) stronger engines: YuNet detector + SFace embeddings
python scripts/fetch_models.py --yes

# 4 · train and evaluate
python -m ml.train --dataset dataset --classifier svc --extractor auto

# 5 · check the pipeline end to end
python scripts/smoke_test.py</div>
        <p class="fine">Training can also be started from the Model page (POST /api/train).</p>
      </div>

      <div class="card glass panel">
        <div class="panel-head"><h3>Catalogued players</h3>
          <span class="chip-soft">facts only — no photos are bundled</span></div>
        <div class="table-wrap">
          <table class="table">
            <thead><tr><th>Folder name</th><th>Player</th><th>Role</th><th>Images</th><th>Status</th></tr></thead>
            <tbody>
            ${players.players.map((p) => `
              <tr>
                <td class="mono">${esc(p.folder)}</td>
                <td><a href="#/player/${esc(p.id)}">${esc(p.name)}</a></td>
                <td>${esc(p.role || '—')}</td>
                <td class="mono">${num(p.dataset_images)}</td>
                <td>${p.supported ? '<span class="tag ok">in model</span>'
                      : p.dataset_images ? '<span class="tag info">has images</span>'
                      : '<span class="tag">no images</span>'}</td>
              </tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>`;
  }

  /* ── about ─────────────────────────────────────────────────────── */
  async function renderAbout() {
    const config = state.config || await API.config();
    $('#api-list').innerHTML = (config.endpoints || []).map((line) => `<div>${esc(line)}</div>`).join('');
    const caps = await API.capabilities().catch(() => null);
    const steps = $('#about-steps');
    if (caps) {
      const extra = document.createElement('li');
      extra.innerHTML = '<b>Active engines</b><span>detector: ' +
        esc((caps.detector && caps.detector.active) || '—') + ' · extractors: ' +
        esc((caps.extractors || []).filter((x) => x.available).map((x) => x.name).join(', ') || '—') + '</span>';
      steps.appendChild(extra);
    }
  }

  /* ── training ──────────────────────────────────────────────────── */
  function pollTraining(jobId) {
    clearInterval(state.trainTimer);
    state.trainTimer = setInterval(async () => {
      try {
        const job = await API.trainStatus(jobId);
        $('#train-bar').style.width = (job.progress || 0) + '%';
        $('#train-status').textContent = `${job.status} · ${job.stage || ''} · ${(job.progress || 0).toFixed(0)}%`;
        $('#train-log').textContent = (job.log || '').split('\n').slice(-14).join('\n');
        if (job.status === 'succeeded' || job.status === 'failed') {
          clearInterval(state.trainTimer);
          if (job.status === 'succeeded') {
            toast('Training finished — model reloaded.', 'ok');
            await refreshGlobal();
            await renderModel();
          } else {
            toast('Training failed: ' + (job.error || 'unknown error'), 'err', 9000);
          }
        }
      } catch (error) {
        clearInterval(state.trainTimer);
        toast(error.message, 'err');
      }
    }, 2000);
  }

  async function startTraining() {
    const payload = {
      dataset: $('#train-dataset').value,
      classifier: $('#train-classifier').value,
      extractor: $('#train-extractor').value,
      augment: parseInt($('#train-augment').value, 10) || 0,
      confidence_threshold: parseFloat($('#train-threshold').value) || 0.55,
      activate: true,
    };
    const key = $('#train-key').value.trim();
    const progress = $('#train-progress');
    progress.classList.remove('hidden');
    $('#train-bar').style.width = '2%';
    $('#train-status').textContent = 'starting…';
    $('#train-log').textContent = '';
    try {
      const accepted = await API.train(payload, key);
      toast('Training started: ' + accepted.job_id, 'ok');
      pollTraining(accepted.job_id);
    } catch (error) {
      progress.classList.remove('hidden');
      $('#train-status').textContent = 'could not start';
      $('#train-log').textContent = error.message;
      toast(error.message, 'err', 7000);
    }
  }

  /* ── wiring ────────────────────────────────────────────────────── */
  function wire() {
    window.addEventListener('hashchange', route);
    $('#btn-menu').addEventListener('click', () => $('#nav').classList.toggle('open'));

    // identify: tabs
    $$('.tab').forEach((tab) => tab.addEventListener('click', () => setMode(tab.dataset.mode)));

    // identify: upload
    const dz = $('#dropzone');
    const input = $('#file-input');
    dz.addEventListener('click', () => input.click());
    dz.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') input.click(); });
    ['dragenter', 'dragover'].forEach((evt) => dz.addEventListener(evt, (e) => {
      e.preventDefault(); dz.classList.add('drag');
    }));
    ['dragleave', 'drop'].forEach((evt) => dz.addEventListener(evt, (e) => {
      e.preventDefault(); dz.classList.remove('drag');
    }));
    dz.addEventListener('drop', (e) => {
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      handleFile(file);
    });
    input.addEventListener('change', () => handleFile(input.files[0]));
    $('#btn-analyze-upload').addEventListener('click', () => {
      if (!pendingFile) { toast('Choose an image first.', 'warn'); return; }
      analyze(() => API.predictFile(pendingFile), 'Analyzing uploaded image…');
    });

    // identify: camera
    $('#btn-cam-start').addEventListener('click', async () => {
      const video = $('#cam-video');
      try {
        await Camera.start(video);
        $('#cam-hint').textContent = 'Camera running. Frame one face inside the guide and capture.';
        $('#btn-cam-start').disabled = true;
        $('#btn-cam-capture').disabled = false;
        $('#btn-cam-stop').disabled = false;
        $('#cam-overlay').style.display = '';
      } catch (error) {
        toast(Camera.friendlyError(error), 'err', 8000);
      }
    });
    $('#btn-cam-stop').addEventListener('click', () => {
      Camera.stop($('#cam-video'));
      $('#btn-cam-start').disabled = false;
      $('#btn-cam-capture').disabled = true;
      $('#btn-cam-stop').disabled = true;
      $('#btn-cam-capture').classList.remove('hidden');
      $('#btn-cam-analyze').classList.add('hidden');
      $('#btn-cam-analyze').disabled = true;
      $('#cam-hint').textContent = 'Camera stopped.';
    });
    $('#btn-cam-capture').addEventListener('click', () => {
      try {
        pendingCapture = Camera.capture($('#cam-video'), $('#cam-canvas'));
        $('#btn-cam-analyze').classList.remove('hidden');
        $('#btn-cam-analyze').disabled = false;
        $('#cam-hint').textContent = 'Frame captured — analyze it, or capture again.';
      } catch (error) {
        toast(error.message, 'err');
      }
    });
    $('#btn-cam-analyze').addEventListener('click', () => {
      if (!pendingCapture) { toast('Capture a frame first.', 'warn'); return; }
      analyze(() => API.predictBase64(pendingCapture, 'camera'), 'Analyzing captured frame…');
    });

    // players
    $('#player-search').addEventListener('input', (e) => {
      state.query = e.target.value;
      if (!$('#view-players').classList.contains('hidden')) paintPlayers();
    });
    $$('#player-filters .chip').forEach((chip) => chip.addEventListener('click', () => {
      $$('#player-filters .chip').forEach((c) => c.classList.remove('active'));
      chip.classList.add('active');
      state.filter = chip.dataset.filter;
      paintPlayers();
    }));

    // dashboard
    $('#btn-clear-history').addEventListener('click', async () => {
      if (!confirm('Delete the stored prediction history? Uploaded images and face crops stay on disk.')) return;
      try {
        const res = await API.clearPredictions();
        toast('Deleted ' + res.deleted + ' prediction row(s).', 'ok');
        renderDashboard();
      } catch (error) { toast(error.message, 'err'); }
    });

    // model
    $('#btn-train').addEventListener('click', startTraining);

    // reveal on scroll
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) { entry.target.classList.add('in'); io.unobserve(entry.target); }
      });
    }, { threshold: 0.12 });
    $$('.reveal').forEach((node) => io.observe(node));
  }

  /* ── boot ──────────────────────────────────────────────────────── */
  async function boot() {
    wire();
    try {
      state.config = await API.config();
      if (state.config.confidence_threshold) $('#train-threshold').value = state.config.confidence_threshold;
    } catch (error) {
      toast('Backend unreachable: ' + error.message, 'err', 9000);
    }
    await refreshGlobal().catch((e) => toast('Model status unavailable: ' + e.message, 'warn'));
    await route();
    // keep the header honest while a demo model trains in the background
    setInterval(() => { if (state.status && !state.status.ready) refreshGlobal().catch(() => {}); }, 15000);
    setInterval(() => { $('#nav').classList.remove('open'); }, 30000);
  }

  document.addEventListener('DOMContentLoaded', boot);
})();
