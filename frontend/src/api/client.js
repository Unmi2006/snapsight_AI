const BASE_URL = "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, options);
  if (!res.ok) {
    throw new Error(`Request to ${path} failed: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function requestForm(path, formData) {
  const res = await fetch(`${BASE_URL}${path}`, { method: "POST", body: formData });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request to ${path} failed: ${res.status}`);
  }
  return res.json();
}

export const api = {
  health: () => request("/api/health"),
  analyzeDocument: (file) => {
    const fd = new FormData();
    fd.append("file", file);
    return requestForm("/api/documents/analyze", fd);
  },
  captureFrame: (blob, filename = "frame.jpg") => {
    const fd = new FormData();
    fd.append("file", blob, filename);
    return requestForm("/api/camera/capture", fd);
  },
  hardwareInfo: () => request("/api/hardware/info"),
  hardwareVerify: () => request("/api/hardware/verify"),
  benchmarkSummary: (modelName) =>
    request(`/api/benchmark/summary${modelName ? `?model_name=${encodeURIComponent(modelName)}` : ""}`),
  benchmarkRecords: (modelName, limit = 200) =>
    request(
      `/api/benchmark/records?limit=${limit}${modelName ? `&model_name=${encodeURIComponent(modelName)}` : ""}`
    ),
  visionModes: () => request("/api/vision/modes"),
  llmStatus: () => request("/api/vision/llm-status"),
  studyVision: (file, { mode = "explain", contentType = "auto", question = "", maxNewTokens = 400 } = {}) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("mode", mode);
    fd.append("content_type", contentType);
    fd.append("question", question);
    fd.append("max_new_tokens", String(maxNewTokens));
    return requestForm("/api/vision/study", fd);
  },

  // Phase 3 -- Voice Assistant
  voiceStatus: () => request("/api/voice/status"),
  transcribe: (blob, filename = "speech.webm") => {
    const fd = new FormData();
    fd.append("file", blob, filename);
    return requestForm("/api/voice/transcribe", fd);
  },
  speak: async (text) => {
    const fd = new FormData();
    fd.append("text", text);
    const res = await fetch(`${BASE_URL}/api/voice/speak`, { method: "POST", body: fd });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request to /api/voice/speak failed: ${res.status}`);
    }
    return {
      blob: await res.blob(),
      engine: res.headers.get("x-tts-engine"),
      activeProvider: res.headers.get("x-tts-provider"),
      latencyMs: Number(res.headers.get("x-tts-latency-ms")),
      sampleRate: Number(res.headers.get("x-sample-rate")),
    };
  },
  voiceAsk: (audioBlob, { image = null, maxNewTokens = 300, speakResponse = true, filename = "speech.webm" } = {}) => {
    const fd = new FormData();
    fd.append("audio", audioBlob, filename);
    if (image) fd.append("image", image);
    fd.append("max_new_tokens", String(maxNewTokens));
    fd.append("speak_response", String(speakResponse));
    return requestForm("/api/voice/ask", fd);
  },

  // Phase 4 -- Conversation History
  historySummary: () => request("/api/history/summary"),
  historyList: (kind, limit = 50, offset = 0) =>
    request(`/api/history/list?limit=${limit}&offset=${offset}${kind ? `&kind=${kind}` : ""}`),
  historyItem: (id) => request(`/api/history/${id}`),
  historyDelete: (id) => fetch(`${BASE_URL}/api/history/${id}`, { method: "DELETE" }).then((r) => r.json()),
  historyClear: (kind) =>
    fetch(`${BASE_URL}/api/history${kind ? `?kind=${kind}` : ""}`, { method: "DELETE" }).then((r) => r.json()),
  historyReplayAudio: async (id) => {
    const res = await fetch(`${BASE_URL}/api/history/${id}/replay-audio`, { method: "POST" });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Replay failed: ${res.status}`);
    }
    return {
      blob: await res.blob(),
      engine: res.headers.get("x-tts-engine"),
      activeProvider: res.headers.get("x-tts-provider"),
      latencyMs: Number(res.headers.get("x-tts-latency-ms")),
      note: res.headers.get("x-replay-note"),
    };
  },

  // Phase 4 -- Settings
  getSettings: () => request("/api/settings"),
  updateSettings: (patch) =>
    fetch(`${BASE_URL}/api/settings`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }).then(async (res) => {
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Settings update failed: ${res.status}`);
      }
      return res.json();
    }),
  resetSettings: () => fetch(`${BASE_URL}/api/settings/reset`, { method: "POST" }).then((r) => r.json()),
};
