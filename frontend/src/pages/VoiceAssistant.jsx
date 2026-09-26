import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";

// Decodes the base64 WAV the backend returns from /api/voice/ask into a
// playable blob URL. Done client-side so the JSON response can carry
// audio + all the engine/provider/latency metadata in one round trip.
function base64ToAudioUrl(base64, mime = "audio/wav") {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return URL.createObjectURL(new Blob([bytes], { type: mime }));
}

export default function VoiceAssistant() {
  const [status, setStatus] = useState(null);
  const [statusError, setStatusError] = useState(null);

  // -- Microphone recording --
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);
  const [recording, setRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [recordedUrl, setRecordedUrl] = useState(null);
  const [micError, setMicError] = useState(null);

  // -- Optional visual context (camera or upload), same pattern as Study Vision --
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const [imageSource, setImageSource] = useState("none"); // "none" | "camera" | "upload"
  const [streamError, setStreamError] = useState(null);
  const [imageBlob, setImageBlob] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);

  const [asking, setAsking] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [answerAudioUrl, setAnswerAudioUrl] = useState(null);

  useEffect(() => {
    api.voiceStatus().then(setStatus).catch((err) => setStatusError(err.message));
  }, []);

  // Camera stream lifecycle -- only active while imageSource === "camera".
  useEffect(() => {
    if (imageSource !== "camera") return;
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
  }, [imageSource]);

  const startRecording = async () => {
    setMicError(null);
    setResult(null);
    setRecordedBlob(null);
    setRecordedUrl(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        setRecordedBlob(blob);
        setRecordedUrl(URL.createObjectURL(blob));
        stream.getTracks().forEach((t) => t.stop());
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      setRecording(true);
    } catch (err) {
      setMicError(err.message);
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  };

  const captureImage = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        setImageBlob(blob);
        setImagePreviewUrl(URL.createObjectURL(blob));
      },
      "image/jpeg",
      0.9
    );
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setImageBlob(file);
    setImagePreviewUrl(URL.createObjectURL(file));
  };

  const clearImage = () => {
    setImageSource("none");
    setImageBlob(null);
    setImagePreviewUrl(null);
  };

  const ask = async () => {
    if (!recordedBlob) return;
    setAsking(true);
    setError(null);
    setResult(null);
    setAnswerAudioUrl(null);
    try {
      const res = await api.voiceAsk(recordedBlob, {
        image: imageBlob,
        filename: (recordedBlob.type.split("/")[1] || "webm").includes("webm") ? "speech.webm" : "speech.ogg",
      });
      setResult(res);
      if (res.tts?.available) {
        setAnswerAudioUrl(base64ToAudioUrl(res.tts.audio_base64));
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setAsking(false);
    }
  };

  const stageBadge = (label, stageStatus) =>
    stageStatus?.available ? (
      <span className="badge badge-ok">
        {label}: {stageStatus.engine || stageStatus.model_name} ({stageStatus.active_provider})
      </span>
    ) : (
      <span className="badge badge-off">{label}: unavailable</span>
    );

  return (
    <div className="page">
      <h1>Voice Assistant</h1>
      <p className="subtitle">
        Speak a question -- optionally point the camera at something first -- and SnapSight AI will transcribe it,
        reason over it locally, and read the answer back.
      </p>

      {statusError && <p className="error">Could not reach the backend: {statusError}</p>}
      {status && (
        <div className="badges">
          {stageBadge("STT", status.stt)}
          {stageBadge("TTS", status.tts)}
          {stageBadge("LLM", status.llm)}
        </div>
      )}
      {status && !status.stt.available && (
        <p className="error" style={{ marginBottom: 16 }}>
          Speech-to-text isn't installed yet ({status.stt.reason?.split(".")[0]}). See
          backend/scripts/export_stt_model.py, or install the CPU fallback: `pip install SpeechRecognition
          pocketsphinx`.
        </p>
      )}

      <h2 style={{ fontSize: "1.1rem" }}>1. Ask your question</h2>
      {micError && <p className="error">Could not access the microphone: {micError}</p>}
      <div className="badges">
        {!recording ? (
          <button onClick={startRecording} style={{ marginBottom: 0 }}>
            🎙️ Start recording
          </button>
        ) : (
          <button onClick={stopRecording} style={{ marginBottom: 0, background: "var(--off)" }}>
            ⏹ Stop recording
          </button>
        )}
      </div>
      {recordedUrl && (
        <div style={{ marginBottom: 16 }}>
          <audio controls src={recordedUrl} />
        </div>
      )}

      <h2 style={{ fontSize: "1.1rem" }}>2. Add visual context (optional)</h2>
      <div className="badges" style={{ marginBottom: 12 }}>
        <button
          onClick={() => setImageSource("camera")}
          style={{ opacity: imageSource === "camera" ? 1 : 0.6, marginBottom: 0 }}
        >
          Use camera
        </button>
        <button
          onClick={() => {
            setImageSource("upload");
            fileInputRef.current?.click();
          }}
          style={{ opacity: imageSource === "upload" ? 1 : 0.6, marginBottom: 0 }}
        >
          Upload image
        </button>
        {imageBlob && (
          <button onClick={clearImage} style={{ marginBottom: 0, background: "var(--panel-2)" }}>
            Clear image
          </button>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          style={{ display: "none" }}
        />
      </div>

      {imageSource === "camera" && !imageBlob && (
        <>
          {streamError && (
            <p className="error">
              Could not access the camera: {streamError}. Camera access needs HTTPS or localhost.
            </p>
          )}
          <div style={{ position: "relative", maxWidth: 360 }}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ width: "100%", display: streamError ? "none" : "block" }}
            />
          </div>
          <canvas ref={canvasRef} style={{ display: "none" }} />
          <button onClick={captureImage} disabled={!!streamError} style={{ marginTop: 12 }}>
            Capture frame
          </button>
        </>
      )}
      {imagePreviewUrl && (
        <div style={{ marginBottom: 16 }}>
          <img
            src={imagePreviewUrl}
            alt="Visual context"
            style={{ maxWidth: 280, borderRadius: 8, border: "1px solid var(--border)" }}
          />
        </div>
      )}

      <h2 style={{ fontSize: "1.1rem" }}>3. Ask</h2>
      <div>
        <button onClick={ask} disabled={!recordedBlob || asking || !status?.stt?.available}>
          {asking ? "Thinking…" : "Ask SnapSight AI"}
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      {result && (
        <section>
          <h2>Transcript</h2>
          <div className="badges">
            <span className="badge badge-ok">{result.stt.engine}</span>
            <span className="badge badge-ok">{result.stt.active_provider}</span>
            <span className="badge badge-ok">{result.stt.latency_ms} ms</span>
          </div>
          <pre style={{ whiteSpace: "pre-wrap" }}>{result.stt.text || "(nothing recognized -- try speaking closer to the mic)"}</pre>

          {result.ocr && (
            <>
              <h2>Visual context (OCR)</h2>
              {result.ocr.available !== false ? (
                <>
                  <div className="badges">
                    <span className="badge badge-ok">{result.ocr.engine}</span>
                    <span className="badge badge-ok">{result.ocr.active_provider}</span>
                    <span className="badge badge-ok">{result.ocr.latency_ms} ms</span>
                  </div>
                  <pre style={{ whiteSpace: "pre-wrap" }}>{result.ocr.text || "(nothing recognized)"}</pre>
                </>
              ) : (
                <p className="error">OCR unavailable: {result.ocr.reason}</p>
              )}
            </>
          )}

          <h2>AI answer</h2>
          {result.llm.available ? (
            <>
              <div className="badges">
                <span className="badge badge-ok">{result.llm.model_name}</span>
                <span className="badge badge-ok">{result.llm.active_provider}</span>
                <span className="badge badge-ok">{result.llm.latency_ms} ms</span>
              </div>
              <pre style={{ whiteSpace: "pre-wrap" }}>{result.llm.answer}</pre>
            </>
          ) : (
            <p className="error">
              LLM unavailable: {result.llm.reason?.split(".")[0]}. Run backend/scripts/export_llm_model.py to enable
              this step.
            </p>
          )}

          <h2>Spoken answer</h2>
          {answerAudioUrl ? (
            <>
              <div className="badges">
                <span className="badge badge-ok">{result.tts.engine}</span>
                <span className="badge badge-ok">{result.tts.active_provider}</span>
                <span className="badge badge-ok">{result.tts.latency_ms} ms</span>
              </div>
              <audio controls autoPlay src={answerAudioUrl} />
            </>
          ) : (
            <p className="error">{result.tts?.reason || "No spoken answer available."}</p>
          )}
        </section>
      )}
    </div>
  );
}
