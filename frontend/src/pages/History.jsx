import { useEffect, useState } from "react";
import { api } from "../api/client";

const KIND_LABEL = { study_vision: "Study Vision", voice: "Voice Assistant" };

function formatTimestamp(unixSeconds) {
  return new Date(unixSeconds * 1000).toLocaleString();
}

export default function History() {
  const [summary, setSummary] = useState(null);
  const [kindFilter, setKindFilter] = useState("");
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailError, setDetailError] = useState(null);

  const [replayAudioUrl, setReplayAudioUrl] = useState(null);
  const [replaying, setReplaying] = useState(false);
  const [replayError, setReplayError] = useState(null);

  const refresh = () => {
    setLoading(true);
    setError(null);
    Promise.all([api.historySummary(), api.historyList(kindFilter || undefined, 100)])
      .then(([s, l]) => {
        setSummary(s);
        setItems(l.items);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kindFilter]);

  const openDetail = (id) => {
    setSelectedId(id);
    setDetail(null);
    setDetailError(null);
    setReplayAudioUrl(null);
    setReplayError(null);
    api.historyItem(id).then(setDetail).catch((err) => setDetailError(err.message));
  };

  const closeDetail = () => {
    setSelectedId(null);
    setDetail(null);
    setReplayAudioUrl(null);
  };

  const deleteItem = async (id, e) => {
    e?.stopPropagation();
    await api.historyDelete(id);
    if (selectedId === id) closeDetail();
    refresh();
  };

  const clearAll = async () => {
    await api.historyClear(kindFilter || undefined);
    closeDetail();
    refresh();
  };

  const replay = async (id) => {
    setReplaying(true);
    setReplayError(null);
    setReplayAudioUrl(null);
    try {
      const res = await api.historyReplayAudio(id);
      setReplayAudioUrl(URL.createObjectURL(res.blob));
    } catch (err) {
      setReplayError(err.message);
    } finally {
      setReplaying(false);
    }
  };

  return (
    <div className="page">
      <h1>Conversation History</h1>
      <p className="subtitle">
        Every Study Vision and Voice Assistant turn is saved locally as it happens. Nothing here is sent anywhere --
        it's read straight out of the backend's on-disk store.
      </p>

      {summary && (
        <div className="stat-grid">
          <div className="stat-card">
            <div className="stat-value">{summary.total}</div>
            <div>Total turns</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{summary.by_kind.study_vision || 0}</div>
            <div>Study Vision</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{summary.by_kind.voice || 0}</div>
            <div>Voice Assistant</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{summary.with_answer}</div>
            <div>With an AI answer</div>
          </div>
        </div>
      )}

      <div className="badges" style={{ alignItems: "center" }}>
        <select value={kindFilter} onChange={(e) => setKindFilter(e.target.value)}>
          <option value="">All kinds</option>
          <option value="study_vision">Study Vision</option>
          <option value="voice">Voice Assistant</option>
        </select>
        <button onClick={refresh} style={{ marginBottom: 0, background: "var(--panel-2)" }}>
          Refresh
        </button>
        {items.length > 0 && (
          <button onClick={clearAll} style={{ marginBottom: 0, background: "var(--off)" }}>
            Clear {kindFilter ? KIND_LABEL[kindFilter] : "all"} history
          </button>
        )}
      </div>

      {error && <p className="error">Could not reach the backend: {error}</p>}
      {loading && <p className="subtitle">Loading…</p>}

      {!loading && !error && items.length === 0 && (
        <p className="subtitle">
          No history yet. Ask something in Study Vision or Voice Assistant and it'll show up here.
        </p>
      )}

      {!loading && items.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>When</th>
              <th>Kind</th>
              <th>Mode</th>
              <th>Preview</th>
              <th>Answer</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr
                key={item.id}
                onClick={() => openDetail(item.id)}
                style={{ cursor: "pointer", background: selectedId === item.id ? "var(--panel-2)" : "transparent" }}
              >
                <td style={{ whiteSpace: "nowrap" }}>{formatTimestamp(item.timestamp)}</td>
                <td>
                  <span className={item.kind === "voice" ? "badge badge-ok" : "badge badge-off"}>
                    {KIND_LABEL[item.kind] || item.kind}
                  </span>
                </td>
                <td>{item.mode}</td>
                <td style={{ maxWidth: 320, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {item.input_preview || <em style={{ color: "var(--text-dim)" }}>(no text recognized)</em>}
                </td>
                <td>
                  {item.answer_available ? (
                    <span className="badge badge-ok">answered</span>
                  ) : (
                    <span className="badge badge-off">no answer</span>
                  )}
                </td>
                <td>
                  <button
                    onClick={(e) => deleteItem(item.id, e)}
                    style={{ marginBottom: 0, background: "var(--panel-2)", padding: "4px 10px" }}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {selectedId && (
        <section style={{ borderTop: "1px solid var(--border)", paddingTop: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h2>Turn #{selectedId}</h2>
            <button onClick={closeDetail} style={{ marginBottom: 0, background: "var(--panel-2)" }}>
              Close
            </button>
          </div>

          {detailError && <p className="error">{detailError}</p>}

          {detail && (
            <>
              <p className="subtitle">
                {KIND_LABEL[detail.kind] || detail.kind} -- {detail.mode}
                {detail.content_type ? ` -- ${detail.content_type}` : ""} -- {formatTimestamp(detail.timestamp)}
              </p>

              <h2 style={{ fontSize: "1.05rem" }}>
                {detail.kind === "voice" ? "Transcript" : "Extracted content (OCR)"}
              </h2>
              <pre style={{ whiteSpace: "pre-wrap" }}>
                {detail.input_text || "(nothing recognized)"}
              </pre>

              {detail.question && detail.kind === "study_vision" && (
                <>
                  <h2 style={{ fontSize: "1.05rem" }}>Question asked</h2>
                  <pre style={{ whiteSpace: "pre-wrap" }}>{detail.question}</pre>
                </>
              )}

              <h2 style={{ fontSize: "1.05rem" }}>AI answer</h2>
              {detail.answer_available ? (
                <>
                  <div className="badges">
                    <span className="badge badge-ok">{detail.answer_engine}</span>
                    <span className="badge badge-ok">{detail.answer_provider}</span>
                    <span className="badge badge-ok">{detail.answer_latency_ms} ms</span>
                  </div>
                  <pre style={{ whiteSpace: "pre-wrap" }}>{detail.answer_text}</pre>

                  <button onClick={() => replay(detail.id)} disabled={replaying}>
                    {replaying ? "Synthesizing…" : "🔊 Replay (re-speak this answer)"}
                  </button>
                  <p className="subtitle" style={{ marginTop: -12, marginBottom: 12 }}>
                    Re-synthesizes the saved text through whichever TTS backend is active now -- the original audio
                    wasn't kept, so this is a fresh reading, not a recording.
                  </p>
                  {replayError && <p className="error">{replayError}</p>}
                  {replayAudioUrl && <audio controls autoPlay src={replayAudioUrl} />}
                </>
              ) : (
                <p className="error">No AI answer was available for this turn.</p>
              )}

              <details style={{ marginTop: 16 }}>
                <summary>Raw metadata (engine/provider/latency per stage)</summary>
                <pre>{JSON.stringify(detail.metadata, null, 2)}</pre>
              </details>
            </>
          )}
        </section>
      )}
    </div>
  );
}
