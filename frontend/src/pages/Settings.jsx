import { useEffect, useState } from "react";
import { api } from "../api/client";

function ProviderSelect({ label, value, options, onChange, autoLabel = "Auto (NPU → GPU → CPU priority)" }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <label style={{ display: "block", marginBottom: 6, fontWeight: 600 }}>{label}</label>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="auto">{autoLabel}</option>
        {options.map((opt) => (
          <option key={opt.id} value={opt.id} disabled={!opt.installed}>
            {opt.label}
            {!opt.installed ? " -- not installed" : ""}
          </option>
        ))}
      </select>
    </div>
  );
}

export default function Settings() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(null);
  const [savedFlash, setSavedFlash] = useState(false);

  const load = () => {
    setLoading(true);
    api
      .getSettings()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const apply = async (patch) => {
    setSaving(true);
    setSaveError(null);
    setSavedFlash(false);
    try {
      const updated = await api.updateSettings(patch);
      setData(updated);
      setSavedFlash(true);
      setTimeout(() => setSavedFlash(false), 1500);
    } catch (err) {
      setSaveError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const reset = async () => {
    setSaving(true);
    setSaveError(null);
    try {
      const updated = await api.resetSettings();
      setData(updated);
    } catch (err) {
      setSaveError(err.message);
    } finally {
      setSaving(false);
    }
  };

  if (loading) return <div className="page"><h1>Settings</h1><p className="subtitle">Loading…</p></div>;
  if (error) return <div className="page"><h1>Settings</h1><p className="error">Could not reach the backend: {error}</p></div>;

  const { settings, available, live_status: liveStatus } = data;

  return (
    <div className="page">
      <h1>Settings</h1>
      <p className="subtitle">
        Model selection and provider overrides here apply immediately -- each change re-loads the relevant backend
        and reports honestly if the forced choice isn't installed on this machine, rather than silently using a
        different one.
      </p>

      {saveError && <p className="error">{saveError}</p>}
      {savedFlash && <p style={{ color: "var(--ok)" }}>Saved.</p>}

      <h2>Local LLM (Study Vision + Voice reasoning)</h2>
      <div className="badges">
        {liveStatus.llm.available ? (
          <span className="badge badge-ok">
            {liveStatus.llm.model_name} ({liveStatus.llm.active_provider})
          </span>
        ) : (
          <span className="badge badge-off">unavailable</span>
        )}
      </div>
      {!liveStatus.llm.available && (
        <p className="subtitle" style={{ marginTop: -8 }}>{liveStatus.llm.reason?.split(".")[0]}.</p>
      )}
      <ProviderSelect
        label="Provider preference"
        value={settings.llm_provider_preference}
        options={available.llm_providers}
        onChange={(v) => apply({ llm_provider_preference: v })}
      />

      <h2>Speech-to-text (Voice Assistant input)</h2>
      <div className="badges">
        {liveStatus.stt.available ? (
          <span className="badge badge-ok">
            {liveStatus.stt.engine} ({liveStatus.stt.active_provider})
          </span>
        ) : (
          <span className="badge badge-off">unavailable</span>
        )}
      </div>
      <ProviderSelect
        label="Backend preference"
        value={settings.stt_backend_preference}
        options={available.stt_backends}
        onChange={(v) => apply({ stt_backend_preference: v })}
        autoLabel="Auto (Whisper ONNX → PocketSphinx)"
      />

      <h2>Text-to-speech (Voice Assistant output)</h2>
      <div className="badges">
        {liveStatus.tts.available ? (
          <span className="badge badge-ok">
            {liveStatus.tts.engine} ({liveStatus.tts.active_provider})
          </span>
        ) : (
          <span className="badge badge-off">unavailable</span>
        )}
      </div>
      <ProviderSelect
        label="Backend preference"
        value={settings.tts_backend_preference}
        options={available.tts_backends}
        onChange={(v) => apply({ tts_backend_preference: v })}
        autoLabel="Auto (Piper ONNX → pyttsx3)"
      />

      {available.tts_voices.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <label style={{ display: "block", marginBottom: 6, fontWeight: 600 }}>
            pyttsx3 voice (only used when the pyttsx3 backend is active)
          </label>
          <select
            value={settings.tts_voice_id || ""}
            onChange={(e) => apply({ tts_voice_id: e.target.value })}
          >
            <option value="">Engine default</option>
            {available.tts_voices.map((v) => (
              <option key={v.id} value={v.id}>
                {v.name}
              </option>
            ))}
          </select>
        </div>
      )}

      <h2>Defaults</h2>
      <div style={{ marginBottom: 16 }}>
        <label style={{ display: "block", marginBottom: 6, fontWeight: 600 }}>Default Study Vision mode</label>
        <select
          value={settings.default_study_vision_mode}
          onChange={(e) => apply({ default_study_vision_mode: e.target.value })}
        >
          {available.study_vision_modes.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      <div style={{ marginBottom: 16 }}>
        <label style={{ display: "block", marginBottom: 6, fontWeight: 600 }}>
          Default max new tokens: {settings.default_max_new_tokens}
        </label>
        <input
          type="range"
          min={16}
          max={1024}
          step={16}
          value={settings.default_max_new_tokens}
          onChange={(e) => apply({ default_max_new_tokens: Number(e.target.value) })}
          style={{ width: 300 }}
        />
      </div>

      <div style={{ marginBottom: 20 }}>
        <label>
          <input
            type="checkbox"
            checked={settings.auto_speak_voice_responses}
            onChange={(e) => apply({ auto_speak_voice_responses: e.target.checked })}
            style={{ marginRight: 8 }}
          />
          Auto-speak Voice Assistant responses by default
        </label>
      </div>

      <button onClick={reset} disabled={saving} style={{ background: "var(--panel-2)" }}>
        Reset all settings to defaults
      </button>
    </div>
  );
}
