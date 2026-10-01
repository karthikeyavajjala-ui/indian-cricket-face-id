/* Animated stadium particle field — dependency-free, respects reduced motion. */
(function () {
  const canvas = document.getElementById('fx-particles');
  if (!canvas) return;
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ctx = canvas.getContext('2d');
  let width = 0, height = 0, dpr = 1, particles = [], raf = null, running = true;
  const COLORS = ['rgba(34,211,238,', 'rgba(163,230,53,', 'rgba(245,196,81,', 'rgba(148,193,255,'];

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = canvas.clientWidth = window.innerWidth;
    height = canvas.clientHeight = window.innerHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const target = Math.min(90, Math.round((width * height) / 26000));
    particles = Array.from({ length: target }, () => spawn());
  }

  function spawn() {
    const depth = Math.random();
    return {
      x: Math.random() * width,
      y: Math.random() * height,
      r: 0.6 + depth * 2.1,
      vx: (Math.random() - 0.5) * (0.12 + depth * 0.3),
      vy: -(0.05 + Math.random() * (0.22 + depth * 0.4)),
      a: 0.12 + depth * 0.4,
      c: COLORS[(Math.random() * COLORS.length) | 0],
      tw: Math.random() * Math.PI * 2,
    };
  }

  function frame(t) {
    if (!running) return;
    ctx.clearRect(0, 0, width, height);
    for (const p of particles) {
      p.x += p.vx; p.y += p.vy; p.tw += 0.02;
      if (p.y < -12) { p.y = height + 10; p.x = Math.random() * width; }
      if (p.x < -12) p.x = width + 10;
      if (p.x > width + 12) p.x = -10;
      const alpha = p.a * (0.65 + 0.35 * Math.sin(p.tw));
      ctx.beginPath();
      ctx.fillStyle = p.c + alpha.toFixed(3) + ')';
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }
    raf = requestAnimationFrame(frame);
  }

  function start() { if (!reduce && !raf) { running = true; raf = requestAnimationFrame(frame); } }
  function stop() { running = false; if (raf) cancelAnimationFrame(raf); raf = null; }

  resize();
  window.addEventListener('resize', resize, { passive: true });
  document.addEventListener('visibilitychange', () => document.hidden ? stop() : start());
  start();
})();
