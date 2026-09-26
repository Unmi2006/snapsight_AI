import { BrowserRouter, Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import HardwareInfo from "./pages/HardwareInfo";
import PerformanceDashboard from "./pages/PerformanceDashboard";
import CameraMode from "./pages/CameraMode";
import DocumentAnalysis from "./pages/DocumentAnalysis";
import StudyVision from "./pages/StudyVision";
import VoiceAssistant from "./pages/VoiceAssistant";
import History from "./pages/History";
import Settings from "./pages/Settings";
import "./App.css";

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <Sidebar />
        <main className="content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/hardware" element={<HardwareInfo />} />
            <Route path="/performance" element={<PerformanceDashboard />} />
            <Route path="/camera" element={<CameraMode />} />
            <Route path="/study-vision" element={<StudyVision />} />
            <Route path="/documents" element={<DocumentAnalysis />} />
            <Route path="/voice" element={<VoiceAssistant />} />
            <Route path="/history" element={<History />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
