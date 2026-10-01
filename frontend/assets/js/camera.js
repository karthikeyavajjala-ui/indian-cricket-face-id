/* Browser camera helper: permission-aware, with clear failure messages. */
window.Camera = (function () {
  let stream = null;

  function support() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  function friendlyError(error) {
    const name = (error && error.name) || '';
    if (name === 'NotAllowedError' || name === 'SecurityError') {
      return 'Camera permission was blocked. Allow camera access for this page, or use the '
        + 'upload mode — browsers only permit the camera on https:// or localhost.';
    }
    if (name === 'NotFoundError' || name === 'OverconstrainedError') {
      return 'No usable camera was found on this device. Use the upload mode instead.';
    }
    if (name === 'NotReadableError') {
      return 'The camera is busy in another application. Close it and try again.';
    }
    return 'Could not start the camera (' + (name || 'unknown error') + '). Use the upload mode.';
  }

  async function start(videoEl) {
    if (!support()) throw new Error('This browser does not expose a camera API (getUserMedia).');
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 960 } },
      audio: false,
    });
    videoEl.srcObject = stream;
    await videoEl.play().catch(() => {});
    return stream;
  }

  function stop(videoEl) {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    if (videoEl) videoEl.srcObject = null;
  }

  function active() { return !!stream; }

  /* Capture the current frame to a JPEG data URL, mirroring the preview. */
  function capture(videoEl, canvasEl, maxSide = 1024, mirror = true) {
    const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
    if (!vw || !vh) throw new Error('The camera has not produced a frame yet.');
    const scale = Math.min(1, maxSide / Math.max(vw, vh));
    canvasEl.width = Math.round(vw * scale);
    canvasEl.height = Math.round(vh * scale);
    const ctx = canvasEl.getContext('2d');
    if (mirror) {
      ctx.translate(canvasEl.width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
    return canvasEl.toDataURL('image/jpeg', 0.92);
  }

  return { support, start, stop, active, capture, friendlyError };
})();
