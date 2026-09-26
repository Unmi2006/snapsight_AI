import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";

export default function CameraMode() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [streamError, setStreamError] = useState(null);
  const [result, setResult] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState(null);

  useEffect(() => {
    let stream;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch (err) {
        setStreamError(err.message);
      }
    })();
    return () => {
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const captureAndAnalyze = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      async (blob) => {
        if (!blob) return;
        setAnalyzing(true);
        setAnalyzeError(null);
        setResult(null);
        try {
          const res = await api.captureFrame(blob);
          setResult(res);
          drawOverlay(canvas, res.boxes);
        } catch (err) {
          setAnalyzeError(err.message);
        } finally {
          setAnalyzing(false);
        }
      },
      "image/jpeg",
      0.9
    );
  };

  const drawOverlay = (canvas, boxes) => {
    const ctx = canvas.getContext("2d");
    ctx.strokeStyle = "#4f46e5";
    ctx.lineWidth = 2;
    ctx.font = "14px sans-serif";
    ctx.fillStyle = "#4f46e5";
    for (const b of boxes) {
      ctx.strokeRect(b.left, b.top, b.width, b.height);
    }
  };

  return (
    <div className="page">
      <h1>Camera Mode</h1>
      <p className="subtitle">
        Point the camera at text, then capture a frame to run it through the local OCR pipeline.
      </p>

      {streamError && (
        <p className="error">
          Could not access the camera: {streamError}. Check browser permissions (camera access needs HTTPS or
          localhost).
        </p>
      )}

      <div style={{ position: "relative", maxWidth: 640 }}>
        <video ref={videoRef} autoPlay playsInline muted style={{ width: "100%", display: streamError ? "none" : "block" }} />
        <canvas ref={canvasRef} style={{ width: "100%", marginTop: 12, background: "#000" }} />
      </div>

      <button onClick={captureAndAnalyze} disabled={analyzing || !!streamError} style={{ marginTop: 12 }}>
        {analyzing ? "Analyzing…" : "Capture frame"}
      </button>

      {analyzeError && <p className="error">{analyzeError}</p>}

      {result && (
        <section>
          <div className="badges">
            <span className="badge badge-ok">{result.engine}</span>
            <span className="badge badge-ok">{result.active_provider}</span>
            <span className="badge badge-ok">{result.latency_ms} ms</span>
          </div>
          <h2>Extracted text</h2>
          <pre style={{ whiteSpace: "pre-wrap" }}>{result.text || "(nothing recognized — try moving closer or improving lighting)"}</pre>
        </section>
      )}
    </div>
  );
}
