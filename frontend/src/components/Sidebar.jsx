import { NavLink } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/camera", label: "Camera Mode" },
  { to: "/study-vision", label: "Study Vision" },
  { to: "/documents", label: "Document Analysis" },
  { to: "/voice", label: "Voice Assistant" },
  { to: "/history", label: "History" },
  { to: "/performance", label: "Performance" },
  { to: "/hardware", label: "Hardware Info" },
  { to: "/settings", label: "Settings" },
];

export default function Sidebar() {
  return (
    <nav className="sidebar">
      <div className="sidebar-title">SnapSight AI</div>
      <ul>
        {NAV_ITEMS.map((item) => (
          <li key={item.to}>
            <NavLink to={item.to} end={item.end} className={({ isActive }) => (isActive ? "active" : "")}>
              {item.label}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}
