export default function MoreScreen({ busy, onLogout, onSelectLanguage, onSetPracticePause, settings }) {
  const languages = settings?.available_studied_languages || [];
  const filters = settings?.temporary_practice_filters || settings || {};
  const listeningPaused = Boolean(filters.listening_temporarily_disabled);
  const speakingPaused = Boolean(filters.speaking_temporarily_disabled);

  return (
    <section className="screen stack-form" aria-label="Ещё и настройки">
      <div className="headline-row">
        <div>
          <p className="eyebrow">Профиль и обучение</p>
          <h2>Ещё</h2>
        </div>
      </div>
      <section className="glass-card stack-form">
        <h3>Язык обучения</h3>
        <div className="option-grid">
          {languages.map((language) => (
            <button
              className={language.code === settings?.active_studied_language ? "option-card active" : "option-card"}
              disabled={busy}
              key={language.code}
              onClick={() => onSelectLanguage(language.code, { source: "settings" })}
              type="button"
            >
              {language.label}
            </button>
          ))}
        </div>
      </section>
      <section className="glass-card stack-form">
        <h3>Временная пауза практики</h3>
        <label className="toggle-row">
          <input checked={listeningPaused} disabled={busy} onChange={(event) => onSetPracticePause("listening", event.target.checked)} type="checkbox" />
          <span>Без аудирования 15 минут</span>
        </label>
        <label className="toggle-row">
          <input checked={speakingPaused} disabled={busy} onChange={(event) => onSetPracticePause("speaking", event.target.checked)} type="checkbox" />
          <span>Без говорения 15 минут</span>
        </label>
      </section>
      <button className="secondary-button" disabled={busy} onClick={onLogout} type="button">Выйти</button>
    </section>
  );
}
