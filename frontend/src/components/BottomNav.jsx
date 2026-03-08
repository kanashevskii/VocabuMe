import { PRIMARY_TABS } from "../constants";

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
          className={`${item.id === primaryTab ? "nav-pill active" : "nav-pill"}${
            item.compact ? " compact-nav" : ""
          }`}
          type="button"
          onClick={() => onSelectTab(item.id)}
        >
          <span className="nav-label">{item.label}</span>
        </button>
      ))}
    </nav>
  );
}
