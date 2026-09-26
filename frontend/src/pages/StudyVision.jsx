import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";

export default function StudyVision() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  const [source, setSource] = useState("camera"); // "camera" | "upload"
  const [streamError, setStreamError] = useState(null);
  const [capturedBlob, setCapturedBlob] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const [modes, setModes] = useState([]);
  const [contentTypes, setContentTypes] = useState([]);
  const [mode, setMode] = useState("explain");
  const [contentType, setContentType] = useState("auto");
  const [question, setQuestion] = useState("");

  const [llmStatus, setLlmStatus] = useState(null);
  const [result, setResult] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState(null);

  // Load available modes/content-types from the backend (never hardcoded
  // here) and the current LLM availability banner.
  useEffect(() => {
    api.visionModes().then((r) => {
      setModes(r.modes);
      setContentTypes(r.content_types);
    }).catch(() => {});
    api.llmStatus().then(setLlmStatus).catch(() => {});
  }, []);

  // Camera stream lifecycle -- only active while source === "camera".
  useEffect(() => {
    if (source !== "camera") return;
    let stream;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
        if (videoRef.current) videoRef.current.srcObject = stream;
        setStreamError(null);
      } catch (err) {
        setStreamError(err.message);
      }
    })();
    return () => stream?.getTracks().forEach((t) => t.stop());
  }, [source]);

  const capture = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        setCapturedBlob(blob);
        setPreviewUrl(URL.createObjectURL(blob));
        setResult(null);
      },
      "image/jpeg",
      0.9
    );
  };

  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setCapturedBlob(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
  };

  const analyze = async () => {
    if (!capturedBlob) return;
    setAnalyzing(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.studyVision(capturedBlob, { mode, contentType, question });
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setAnalyzing(false);
    }
  };

  const selectedModeIsCustom = mode === "custom";

  return (
    <div className="page">
      <h1>Study Vision Mode</h1>
      <p className="subtitle">
        Point the camera (or upload a photo) at a circuit diagram, equation, code, graph, flowchart, textbook
        page, or handwritten notes — then pick what you want SnapSight AI to do with it.
      </p>

      {llmStatus && !llmStatus.available && (
        <p className="error" style={{ marginBottom: 16 }}>
          Local LLM reasoning isn't installed yet ({llmStatus.reason?.split(".")[0]}). OCR still works below —
          the "ask AI" answer will show as unavailable until a model is exported. See
          backend/scripts/export_llm_model.py.
        </p>
      )}
      {llmStatus && llmStatus.available && (
        <div className="badges" style={{ marginBottom: 16 }}>
          <span className="badge badge-ok">LLM: {llmStatus.model_name}</span>
          <span className="badge badge-ok">{llmStatus.active_provider}</span>
        </div>
      )}

      <div className="badges" style={{ marginBottom: 12 }}>
        <button
          onClick={() => setSource("camera")}
          style={{ opacity: source === "camera" ? 1 : 0.6, marginBottom: 0 }}
        >
          Use camera
        </button>
        <button
          onClick={() => {
            setSource("upload");
            fileInputRef.current?.click();
          }}
          style={{ opacity: source === "upload" ? 1 : 0.6, marginBottom: 0 }}
        >
          Upload image
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleUpload}
          style={{ display: "none" }}
        />
      </div>

      {source === "camera" && (
        <>
          {streamError && (
            <p className="error">
              Could not access the camera: {streamError}. Camera access needs HTTPS or localhost.
            </p>
          )}
          <div style={{ position: "relative", maxWidth: 480 }}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ width: "100%", display: streamError ? "none" : "block" }}
            />
          </div>
          <canvas ref={canvasRef} style={{ display: "none" }} />
          <button onClick={capture} disabled={!!streamError} style={{ marginTop: 12 }}>
            Capture frame
          </button>
        </>
      )}

      {previewUrl && (
        <div style={{ marginTop: 12, marginBottom: 12 }}>
          <img src={previewUrl} alt="Captured" style={{ maxWidth: 320, borderRadius: 8, border: "1px solid var(--border)" }} />
        </div>
      )}

      {capturedBlob && (
        <section style={{ marginTop: 8 }}>
          <h2 style={{ fontSize: "1.1rem" }}>What should SnapSight AI do?</h2>
          <div className="badges">
            {modes.map((m) => (
              <button
                key={m.id}
                onClick={() => setMode(m.id)}
                style={{ marginBottom: 0, background: mode === m.id ? "var(--accent)" : "var(--panel-2)" }}
              >
                {m.label}
              </button>
            ))}
          </div>

          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center", marginBottom: 16 }}>
            <label style={{ fontSize: "0.85rem", color: "var(--text-dim)" }}>
              Content type (optional hint):{" "}
              <select value={contentType} onChange={(e) => setContentType(e.target.value)}>
                {contentTypes.map((c) => (
                  <option key={c} value={c}>
                    {c.replace(/_/g, " ")}
                  </option>
                ))}
              </select>
            </label>
          </div>

          {selectedModeIsCustom && (
            <textarea
              placeholder='e.g. "Explain this circuit" or "What is wrong with this code?"'
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              rows={3}
              style={{ width: "100%", maxWidth: 500, marginBottom: 12 }}
            />
          )}

          <div>
            <button onClick={analyze} disabled={analyzing || (selectedModeIsCustom && !question.trim())}>
              {analyzing ? "Analyzing…" : "Analyze & Ask"}
            </button>
          </div>
        </section>
      )}

      {error && <p className="error">{error}</p>}

      {result && (
        <section>
          <h2>OCR result</h2>
          <div className="badges">
            <span className="badge badge-ok">{result.ocr.engine}</span>
            <span className="badge badge-ok">{result.ocr.active_provider}</span>
            <span className="badge badge-ok">{result.ocr.latency_ms} ms</span>
            <span className="badge badge-ok">{result.ocr.box_count} regions</span>
          </div>
          <pre style={{ whiteSpace: "pre-wrap" }}>{result.ocr.text || "(nothing recognized)"}</pre>

          <h2>AI answer</h2>
          {result.llm.available ? (
            <>
              <div className="badges">
                <span className="badge badge-ok">{result.llm.model_name}</span>
                <span className="badge badge-ok">{result.llm.active_provider}</span>
                <span className="badge badge-ok">{result.llm.latency_ms} ms</span>
                <span className="badge badge-ok">
                  {result.llm.prompt_tokens}→{result.llm.completion_tokens} tokens
                </span>
              </div>
              <pre style={{ whiteSpace: "pre-wrap" }}>{result.llm.answer}</pre>
            </>
          ) : (
            <p className="error">
              LLM unavailable: {result.llm.reason?.split(".")[0]}. Run
              backend/scripts/export_llm_model.py to enable this step.
            </p>
          )}
        </section>
      )}
    </div>
  );
}
