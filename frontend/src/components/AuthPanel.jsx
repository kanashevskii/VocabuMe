export function LogoMark() {
  return (
    <div className="logo-mark" aria-hidden="true">
      <img src="/static/old_logo.jpg" alt="" />
    </div>
  );
}

export default function AuthPanel({
  config,
  onOpenLogin,
  loginLink,
  loginPending,
  webAuthMode,
  onChangeWebAuthMode,
  onSubmitWebAuth,
  webAuthPending,
  webEmail,
  webPassword,
  onWebEmailChange,
  onWebPasswordChange,
}) {
  const webSubmitLabel =
    webAuthMode === "register" ? "Создать web-аккаунт" : "Войти по email";
  const telegramBotLink = `https://t.me/${config.bot_username || "VocabuMe_bot"}`;

  return (
    <section className="auth-shell">
      <div className="glass-card auth-copy">
        <div className="brand-lockup">
          <LogoMark />
          <div>
            <p className="overline">Lingua Voyage</p>
            <h1>Один вход. Один словарь. Один общий прогресс.</h1>
          </div>
        </div>
        <p className="lead">
          Выбери способ входа. Для локальной веб-версии можно использовать email
          и пароль, а Telegram остаётся основным способом синхронизации.
        </p>
        <div className="glass-card compact-section auth-method-panel">
          <div className="section-head section-head-wrap">
            <div>
              <p className="overline">Telegram</p>
              <h3>Вход через бота</h3>
            </div>
          </div>
          <p className="lead compact">
            Подходит для обычного Telegram flow и Mini App. Если тестируешь
            локальный веб, используй блок ниже.
          </p>
          <div className="auth-actions">
            <button
              className="primary-button"
              type="button"
              onClick={onOpenLogin}
              disabled={loginPending}
            >
              {loginPending ? "Готовим ссылку..." : "Войти через Telegram"}
            </button>
            <a
              className="secondary-button"
              href={telegramBotLink}
              target="_blank"
              rel="noreferrer"
            >
              Открыть бота
            </a>
          </div>
        </div>
        <div className="glass-card compact-section auth-web-panel">
          <div className="section-head section-head-wrap">
            <div>
              <p className="overline">Web</p>
              <h3>
                {webAuthMode === "register"
                  ? "Создать веб-аккаунт"
                  : "Вход в веб-версию"}
              </h3>
            </div>
            <div className="segment-wrap auth-mode-switch">
              <button
                className={
                  webAuthMode === "login"
                    ? "segment-button active"
                    : "segment-button"
                }
                type="button"
                onClick={() => onChangeWebAuthMode("login")}
              >
                Вход
              </button>
              <button
                className={
                  webAuthMode === "register"
                    ? "segment-button active"
                    : "segment-button"
                }
                type="button"
                onClick={() => onChangeWebAuthMode("register")}
              >
                Регистрация
              </button>
            </div>
          </div>
          <form className="stack-form auth-web-form" onSubmit={onSubmitWebAuth}>
            <label className="stack-label">
              <span>Email</span>
              <input
                type="email"
                value={webEmail}
                onChange={(event) => onWebEmailChange(event.target.value)}
                placeholder="you@example.com"
                autoComplete="email"
              />
            </label>
            <label className="stack-label">
              <span>Пароль</span>
              <input
                type="password"
                value={webPassword}
                onChange={(event) => onWebPasswordChange(event.target.value)}
                placeholder="Минимум 8 символов"
                autoComplete={
                  webAuthMode === "register"
                    ? "new-password"
                    : "current-password"
                }
              />
            </label>
            <button
              className="secondary-button"
              type="submit"
              disabled={webAuthPending}
            >
              {webAuthPending ? "Секунду..." : webSubmitLabel}
            </button>
          </form>
        </div>
        {loginLink ? (
          <div className="inline-note">
            <span>Открой бота и нажми Start, чтобы подтвердить Telegram-вход.</span>
            <a
              className="primary-link"
              href={loginLink}
              target="_blank"
              rel="noreferrer"
            >
              🤖 Открыть бота
            </a>
          </div>
        ) : null}
      </div>
      <div className="glass-card auth-preview">
        <div className="auth-preview-grid">
          <article className="mini-pane">
            <span>Учить</span>
            <strong>🧠 Карточки, практика, аудирование</strong>
          </article>
          <article className="mini-pane">
            <span>Словарь</span>
            <strong>📚 Быстрый поиск и редактирование</strong>
          </article>
          <article className="mini-pane">
            <span>Синхронизация</span>
            <strong>🔄 Telegram и web используют один профиль</strong>
          </article>
        </div>
      </div>
    </section>
  );
}
