import { LogoMark } from "./AuthPanel";

export default function AppTopbar({
  busy,
  currentTitle,
  extraClass = "",
  isMiniApp,
  onBack,
  onClose,
  onLogout,
  onToggleAddWords,
  primaryTab,
  showHeaderBack,
  showHeaderClose,
  showLibraryAdd,
}) {
  return (
    <header className={`glass-card topbar ${extraClass}`.trim()}>
      <div className="topbar-brand">
        <LogoMark />
        <div className="topbar-copy">
          <p className="overline">VocabuMe</p>
          <strong className="app-title">{currentTitle}</strong>
        </div>
      </div>
      {showHeaderBack ? (
        <button
          className="secondary-button header-action"
          type="button"
          onClick={onBack}
        >
          <span>← Назад</span>
        </button>
      ) : showHeaderClose ? (
        <button
          className="secondary-button header-action"
          type="button"
          onClick={onClose}
        >
          <span>Закрыть</span>
        </button>
      ) : primaryTab === "words" ? (
        <button
          className={
            showLibraryAdd
              ? "secondary-button header-action words-header-action active"
              : "secondary-button header-action words-header-action"
          }
          type="button"
          onClick={onToggleAddWords}
          aria-label="Добавить слова"
        >
          <span className="header-action-mark">＋</span>
        </button>
      ) : !isMiniApp ? (
        <button
          className="secondary-button header-action"
          type="button"
          onClick={onLogout}
          disabled={busy}
        >
          <span>Выйти</span>
        </button>
      ) : (
        <span className="mode-pill">Telegram</span>
      )}
    </header>
  );
}
