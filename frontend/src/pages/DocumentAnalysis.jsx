import { useState } from "react";
import { api } from "../api/client";

export default function DocumentAnalysis() {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [fileName, setFileName] = useState("");

  const handleFile = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setFileName(file.name);
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.analyzeDocument(file);
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <h1>Document Analysis</h1>
      <p className="subtitle">
        Upload a PDF or image. Digital PDFs extract text instantly; scanned PDFs and images run through the
        local OCR pipeline (ONNX/NPU when available, Tesseract otherwise).
      </p>

      <input type="file" accept=".pdf,.png,.jpg,.jpeg,.bmp,.webp" onChange={handleFile} />
      {loading && <p>Analyzing {fileName}…</p>}
      {error && <p className="error">{error}</p>}

      {result && (
        <section>
          <div className="badges">
            <span className="badge badge-ok">{result.kind}</span>
            <span className="badge badge-ok">{result.page_count} page(s)</span>
            <span className={`badge ${result.ocr_used ? "badge-off" : "badge-ok"}`}>
              {result.ocr_used ? "OCR used" : "Digital text (no OCR needed)"}
            </span>
          </div>

          {result.pages && (
            <table>
              <thead>
                <tr><th>Page</th><th>Engine</th><th>Provider</th><th>Latency</th><th>Words found</th></tr>
              </thead>
              <tbody>
                {result.pages.map((p, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>{p.engine}</td>
                    <td>{p.active_provider}</td>
                    <td>{p.latency_ms} ms</td>
                    <td>{p.box_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <h2>Extracted text</h2>
          <pre style={{ whiteSpace: "pre-wrap" }}>{result.text || "(no text found)"}</pre>
        </section>
      )}
    </div>
  );
}
