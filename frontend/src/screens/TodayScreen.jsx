export default function TodayScreen({
  hasMoreAchievements,
  hasWordsToLearn,
  onOpenAddWords,
  onOpenLearn,
  onOpenProgress,
  progress,
  todayAchievements,
  todayStats,
}) {
  const studiedToday = Boolean(progress?.studied_today);
  const learnedToday = progress?.learned_today ?? 0;

  return (
    <div className="screen-stack">
      <section className="glass-card today-hero">
        <div className="today-hero-grid">
          <div>
            <p className="overline">Today</p>
            <h2>Продолжай учить слова ✨</h2>
            <p className="lead compact">
              {studiedToday
                ? "🔥 Ты уже занимался сегодня. Продолжай в том же духе."
                : "🌱 Сегодня ты ещё не занимался. Давай начнём."}
            </p>
          </div>
          <div className="today-side-stat">
            <span>Сегодня выучено</span>
            <strong>{learnedToday}</strong>
          </div>
        </div>
        <button
          className="primary-button hero-button"
          type="button"
          onClick={() => (hasWordsToLearn ? onOpenLearn() : onOpenAddWords())}
        >
          {hasWordsToLearn ? "▶️ Продолжить" : "＋ Добавить слова"}
        </button>
      </section>

      <section className="today-stats-row">
        {todayStats.map((item) => (
          <article key={item.label} className="glass-card stat-card">
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </article>
        ))}
      </section>

      <section className="glass-card compact-section today-achievements">
        <div className="section-head">
          <div>
            <p className="overline">Achievements</p>
            <h3>Последние награды 🏆</h3>
          </div>
          {hasMoreAchievements ? (
            <button
              className="secondary-button"
              type="button"
              onClick={onOpenProgress}
            >
              В прогресс →
            </button>
          ) : null}
        </div>
        <div className="achievement-grid">
          {todayAchievements.length ? (
            todayAchievements.map((item) => (
              <span key={item} className="achievement-pill">
                {item}
              </span>
            ))
          ) : (
            <div className="empty-state">Пока без наград.</div>
          )}
        </div>
      </section>
    </div>
  );
}
