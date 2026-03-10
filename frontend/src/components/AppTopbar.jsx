import { LogoMark } from "./AuthPanel";

function getProfileInitials(user) {
  const source = (user?.display_name || user?.username || user?.email || "").trim();
  if (!source) {
    return "U";
  }
  const cleaned = source.replace(/^@/, "");
  const parts = cleaned.split(/[\s._-]+/).filter(Boolean);
  if (parts.length >= 2) {
    return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
  }
  return cleaned.slice(0, 2).toUpperCase();
}

export default function AppTopbar({
  busy,
  currentTitle,
  extraClass = "",
  isMiniApp,
  onBack,
  onClose,
  onLogout,
  onOpenProfile,
  onToggleAddWords,
  primaryTab,
  showHeaderBack,
  showHeaderClose,
  showLibraryAdd,
  user,
}) {
  const showProfileBadge = primaryTab === "today" && !showHeaderBack && !showHeaderClose;

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
              ? "secondary-button header-action"
              : "secondary-button header-action words-header-action"
          }
          type="button"
          onClick={onToggleAddWords}
          aria-label={showLibraryAdd ? "Закрыть добавление слов" : "Добавить слова"}
        >
          <span className={showLibraryAdd ? "" : "header-action-mark"}>
            {showLibraryAdd ? "Закрыть" : "＋"}
          </span>
        </button>
      ) : showProfileBadge ? (
        <button
          className="profile-badge"
          type="button"
          onClick={onOpenProfile}
          aria-label="Открыть профиль и настройки"
          title="Профиль и настройки"
        >
          {user?.avatar_url ? (
            <img
              key={user.avatar_url}
              src={user.avatar_url}
              alt={user.display_name || user.username || "Профиль"}
            />
          ) : (
            <span>{getProfileInitials(user)}</span>
          )}
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
