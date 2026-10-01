/* Thin fetch wrapper for the FastAPI backend. */
window.API = (function () {
  async function parse(response) {
    const text = await response.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch (e) { data = { detail: text }; }
    if (!response.ok) {
      const detail = (data && (data.detail || data.message)) || response.statusText;
      const error = new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
      error.status = response.status;
      error.payload = data;
      throw error;
    }
    return data;
  }

  function get(path) {
    return fetch(path, { headers: { Accept: 'application/json' } }).then(parse);
  }

  function postJSON(path, body, headers) {
    return fetch(path, {
      method: 'POST',
      headers: Object.assign({ 'Content-Type': 'application/json' }, headers || {}),
      body: JSON.stringify(body || {}),
    }).then(parse);
  }

  function postFile(path, file, extraHeaders) {
    const form = new FormData();
    form.append('file', file, file.name || 'capture.jpg');
    return fetch(path, { method: 'POST', body: form, headers: extraHeaders || {} }).then(parse);
  }

  function del(path) { return fetch(path, { method: 'DELETE' }).then(parse); }

  function fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error('Could not read the selected file'));
      reader.readAsDataURL(file);
    });
  }

  return {
    get, postJSON, postFile, del, fileToBase64,
    health: () => get('/api/health'),
    config: () => get('/api/config'),
    modelStatus: () => get('/api/model/status'),
    modelMetrics: () => get('/api/model/metrics'),
    modelVersions: () => get('/api/model/versions'),
    dataset: () => get('/api/model/dataset'),
    players: (includeCatalog = true) => get('/api/players?include_catalog=' + includeCatalog),
    player: (id) => get('/api/players/' + encodeURIComponent(id)),
    dashboard: () => get('/api/dashboard/summary'),
    predictions: (limit = 25) => get('/api/predictions?limit=' + limit),
    clearPredictions: () => del('/api/predictions'),
    predictFile: (file) => postFile('/api/predict?source=upload', file),
    predictBase64: (dataUrl, source = 'camera') =>
      postJSON('/api/predict/image', { image_base64: dataUrl, source, top_k: 3, save: true }),
    train: (payload, apiKey) =>
      postJSON('/api/train', payload, apiKey ? { 'X-API-Key': apiKey } : {}),
    trainStatus: (jobId) => get('/api/train/status/' + encodeURIComponent(jobId)),
    trainJobs: () => get('/api/train/jobs'),
    capabilities: () => get('/api/train/capabilities'),
    reloadModel: () => postJSON('/api/model/reload', {}),
  };
})();
