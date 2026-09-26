import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";
import { api } from "../api/client";

export default function PerformanceDashboard() {
  const [summary, setSummary] = useState(null);
  const [records, setRecords] = useState([]);
  const [error, setError] = useState(null);

  const load = async () => {
    try {
      const [s, r] = await Promise.all([api.benchmarkSummary(), api.benchmarkRecords(null, 100)]);
      setSummary(s);
      setRecords(r.records.slice().reverse());
    } catch (e) {
      setError(e.message);
    }
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, []);

  if (error) {
    return (
      <div className="page">
        <h1>Performance Dashboard</h1>
        <p className="error">Could not reach the backend: {error}</p>
      </div>
    );
  }

  return (
    <div className="page">
      <h1>Performance Dashboard</h1>
      {summary && summary.count === 0 && (
        <p className="subtitle">
          No inference calls recorded yet. Run something in Study Vision, Voice, or Document Analysis to populate this.
        </p>
      )}
      {summary && summary.count > 0 && (
        <>
          <div className="stat-grid">
            <div className="stat-card"><div className="stat-value">{summary.count}</div><div>Total calls</div></div>
            <div className="stat-card"><div className="stat-value">{summary.avg_latency_ms} ms</div><div>Avg latency</div></div>
            <div className="stat-card"><div className="stat-value">{summary.p95_latency_ms} ms</div><div>p95 latency</div></div>
            <div className="stat-card"><div className="stat-value">{summary.providers_seen.join(", ")}</div><div>Providers used</div></div>
          </div>

          <h2>Latency over time (last 100 calls)</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={records}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="id" />
              <YAxis label={{ value: "ms", angle: -90, position: "insideLeft" }} />
              <Tooltip />
              <Line type="monotone" dataKey="latency_ms" stroke="#4f46e5" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
}
