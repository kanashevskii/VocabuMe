export default function ProgressScreen({ progress, stats }) {
  const progressTopStats = stats.slice(0, 3);
  const progressStreakStat = stats[3];

  return (
    <div className="screen-stack">
      <section className="stats-grid progress-top-stats">
        {progressTopStats.map((item) => (
          <article key={item.label} className="glass-card stat-card">
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </article>
        ))}
      </section>
      {progressStreakStat ? (
        <section className="stats-grid progress-secondary-stats">
          <article className="glass-card stat-card">
            <span>{progressStreakStat.label}</span>
            <strong>{progressStreakStat.value}</strong>
          </article>
        </section>
      ) : null}
      <section className="glass-card compact-section">
        <p className="overline">Rank</p>
        <div className="headline-row">
          <h3>{progress?.rank_percent ? `${progress.rank_percent}%` : "—"}</h3>
          <span>🏅 Top range</span>
        </div>
      </section>
      <section className="glass-card compact-section">
        <p className="overline">Achievements</p>
        <div className="achievement-grid">
          {(progress?.achievements || []).length ? (
            progress.achievements.map((item) => (
              <span key={item} className="achievement-pill">
                {item}
              </span>
            ))
          ) : (
            <div className="empty-state">No achievements yet.</div>
          )}
        </div>
      </section>
      <section className="glass-card compact-section">
        <p className="overline">Next</p>
        <div className="section-head">
          <h3>К чему стремиться ✨</h3>
        </div>
        <div className="simple-list">
          {(progress?.pending_achievement_highlights || []).length ? (
            progress.pending_achievement_highlights.map((item) => (
              <div key={item.text} className="simple-row">
                <strong>{item.text}</strong>
                <span>
                  {item.current} / {item.target}
                </span>
              </div>
            ))
          ) : (
            <div className="empty-state">
              Все текущие ачивки уже получены.
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
