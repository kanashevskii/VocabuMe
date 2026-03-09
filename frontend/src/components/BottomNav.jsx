import { PRIMARY_TABS } from "../constants";

const TAB_ICONS = {
  today: "🗓️",
  learn: "⭕",
  words: "📘",
  progress: "📊",
  more: "⚙️",
};

export default function BottomNav({
  isKeyboardOpen,
  onSelectTab,
  primaryTab,
}) {
  return (
    <nav className={`nav-grid-bottom${isKeyboardOpen ? " is-hidden" : ""}`}>
      {PRIMARY_TABS.map((item) => (
        <button
          key={item.id}
          className={item.id === primaryTab ? "nav-pill active" : "nav-pill"}
          type="button"
          onClick={() => onSelectTab(item.id)}
        >
          <span className="nav-icon" aria-hidden="true">{TAB_ICONS[item.id] || "•"}</span>
          <span className="nav-label">{item.label}</span>
        </button>
      ))}
    </nav>
  );
}
