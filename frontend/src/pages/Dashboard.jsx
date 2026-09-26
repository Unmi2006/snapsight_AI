import { Link } from "react-router-dom";

export default function Dashboard() {
  return (
    <div className="page">
      <h1>SnapSight AI</h1>
      <p className="subtitle">Privacy-preserving on-device multimodal AI assistant.</p>
      <ul className="quick-links">
        <li><Link to="/hardware">Check hardware &amp; verify NPU/GPU/CPU providers →</Link></li>
        <li><Link to="/performance">View performance dashboard →</Link></li>
        <li><Link to="/study-vision">Open Study Vision Mode →</Link></li>
      </ul>
    </div>
  );
}
