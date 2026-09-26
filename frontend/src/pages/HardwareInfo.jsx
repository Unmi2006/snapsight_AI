import { useEffect, useState } from "react";
import { api } from "../api/client";

function Badge({ ok, label }) {
  return <span className={`badge ${ok ? "badge-ok" : "badge-off"}`}>{label}</span>;
}

export default function HardwareInfo() {
  const [info, setInfo] = useState(null);
  const [verify, setVerify] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const [infoRes, verifyRes] = await Promise.all([api.hardwareInfo(), api.hardwareVerify()]);
      setInfo(infoRes);
      setVerify(verifyRes);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  if (error) {
    return (
      <div className="page">
        <h1>Hardware &amp; Runtime Info</h1>
        <p className="error">
          Could not reach the backend at http://127.0.0.1:8000 — is it running? ({error})
        </p>
        <button onClick={load}>Retry</button>
      </div>
    );
  }

  return (
    <div className="page">
      <h1>Hardware &amp; Runtime Info</h1>
      <p className="subtitle">
        These badges reflect a live micro-benchmark run just now on this machine — not a hardcoded claim.
      </p>
      <button onClick={load} disabled={loading}>
        {loading ? "Checking..." : "Re-check hardware"}
      </button>

      {info && (
        <section>
          <h2>System</h2>
          <table>
            <tbody>
              <tr><td>OS</td><td>{info.system.os} ({info.system.machine})</td></tr>
              <tr><td>Python</td><td>{info.system.python_version}</td></tr>
              <tr><td>CPU cores</td><td>{info.system.cpu_count_physical} physical / {info.system.cpu_count_logical} logical</td></tr>
              <tr><td>RAM</td><td>{info.system.total_ram_gb} GB</td></tr>
            </tbody>
          </table>

          <h2>Execution Providers Available to ONNX Runtime</h2>
          <div className="badges">
            <Badge ok={info.npu_provider_present} label={`NPU (QNN) ${info.npu_provider_present ? "available" : "not detected"}`} />
            <Badge ok={info.gpu_dml_provider_present} label={`GPU (DirectML) ${info.gpu_dml_provider_present ? "available" : "not detected"}`} />
            <Badge ok={info.cpu_provider_present} label="CPU available" />
          </div>
        </section>
      )}

      {verify && (
        <section>
          <h2>Live Micro-Benchmark (verified, not assumed)</h2>
          <table>
            <thead>
              <tr><th>Provider</th><th>Actually used</th><th>Verified</th><th>Session load (ms)</th><th>Op run (ms)</th></tr>
            </thead>
            <tbody>
              {verify.results.map((r) => (
                <tr key={r.requested}>
                  <td>{r.requested}</td>
                  <td>{r.actually_used ?? "—"}</td>
                  <td>{r.verified ? "✅" : "❌"}</td>
                  <td>{r.session_load_ms ?? "—"}</td>
                  <td>{r.op_run_ms ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {verify.results.some((r) => r.error) && (
            <details>
              <summary>Errors (why a provider wasn't usable)</summary>
              <pre>{JSON.stringify(verify.results.filter((r) => r.error), null, 2)}</pre>
            </details>
          )}
        </section>
      )}
    </div>
  );
}
