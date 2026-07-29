export function IrregularPracticeScreen({
  correctCount,
  formatPointsLabel,
  getSessionPraise,
  list,
  mode,
  onAnswer,
  onNextPage,
  onPreviousPage,
  onSkip,
  onStartNewSession,
  onAdvance,
  question,
  questionCount,
  result,
  sessionDone,
  sessionLimit,
}) {
  return (
    <div className="screen-stack">
      {mode === "review" ? (
        <section className="glass-card compact-section">
          <div className="section-head">
            <div>
              <p className="overline">Irregular</p>
              <h3>Повторять глаголы 📘</h3>
            </div>
          </div>
          <div className="simple-list">
            {(list?.items || []).map((item) => (
              <div key={item.base} className="simple-row four-cols">
                <strong>{item.base}</strong>
                <span>{item.past}</span>
                <span>{item.participle}</span>
                <span>{item.translation}</span>
              </div>
            ))}
          </div>
          <div className="button-row card-nav-row">
            <button
              className="secondary-button nav-arrow"
              type="button"
              onClick={onPreviousPage}
              disabled={!list?.has_prev}
              aria-label="Предыдущая страница"
            >
              ←
            </button>
            <button
              className="secondary-button nav-arrow"
              type="button"
              onClick={onNextPage}
              disabled={!list?.has_next}
              aria-label="Следующая страница"
            >
              →
            </button>
          </div>
        </section>
      ) : null}
      {mode === "test" ? (
        <section className="glass-card compact-section">
          <div className="section-head">
            <div>
              <p className="overline">Train</p>
              <h3>Тест по глаголам 🧩</h3>
            </div>
            <span className="status-tag">
              {Math.min(questionCount + 1, sessionLimit)} / {sessionLimit}
            </span>
          </div>
          {question ? (
            <div className="quiz-panel">
              <div className="prompt-card">
                <strong>{question.verb.base}</strong>
                <span>Выбери правильную форму</span>
              </div>
              <div className="option-grid">
                {question.options.map((option) => (
                  <button
                    key={option}
                    className="option-button"
                    type="button"
                    onClick={() => onAnswer(option)}
                  >
                    {option}
                  </button>
                ))}
              </div>
              {!result ? (
                <button className="secondary-button" type="button" onClick={onSkip}>
                  Пропустить
                </button>
              ) : null}
              {result ? (
                <div className={result.correct ? "result-box good" : "result-box bad"}>
                  <div className="result-copy">
                    <span>
                      {result.correct
                        ? "Верно"
                        : `Правильный ответ: ${result.correct_answer}`}
                    </span>
                    {result.points_earned ? (
                      <span className="points-burst">
                        🎉 +{result.points_earned} {formatPointsLabel(result.points_earned)}
                      </span>
                    ) : null}
                  </div>
                  <button className="secondary-button" type="button" onClick={onAdvance}>
                    Дальше
                  </button>
                </div>
              ) : null}
            </div>
          ) : (
            <div className="stack-form">
              <div className="empty-state">
                {sessionDone
                  ? `Тест завершён. Верно ${correctCount} из ${questionCount || sessionLimit}. ✨ +${correctCount} ${formatPointsLabel(correctCount)}. ${getSessionPraise(correctCount, questionCount || sessionLimit)}`
                  : "Сейчас нет вопроса по глаголам."}
              </div>
              {sessionDone ? (
                <button className="primary-button" type="button" onClick={onStartNewSession}>
                  Начать новый тест
                </button>
              ) : null}
            </div>
          )}
        </section>
      ) : null}
    </div>
  );
}

export function AlphabetPracticeScreen({
  audioLoadingSymbol,
  correctCount,
  formatDisplayAnswer,
  formatPointsLabel,
  getSessionPraise,
  list,
  mode,
  onAdvance,
  onAnswer,
  onNextPage,
  onPlayAudio,
  onPreviousPage,
  onSkip,
  onStartNewSession,
  question,
  questionCount,
  result,
  sessionDone,
  sessionLimit,
}) {
  return (
    <div className="screen-stack">
      {mode === "review" ? (
        <section className="glass-card compact-section">
          <div className="section-head">
            <div>
              <p className="overline">Alphabet</p>
              <h3>Повторять алфавит 🔤</h3>
            </div>
          </div>
          <div className="simple-list">
            {(list?.items || []).map((item) => (
              <div key={item.symbol} className="simple-row four-cols">
                <strong>{item.symbol}</strong>
                <span>{item.name}</span>
                <span>/{item.transcription}/</span>
                <div className="alphabet-audio-cell">
                  <span>{item.hint}</span>
                  <button
                    className="secondary-button mini-audio-button"
                    type="button"
                    onClick={() => onPlayAudio(item.symbol)}
                    disabled={audioLoadingSymbol === item.symbol}
                    aria-label={`Слушать букву ${item.symbol}`}
                  >
                    {audioLoadingSymbol === item.symbol ? "..." : "🔊"}
                  </button>
                </div>
              </div>
            ))}
          </div>
          <div className="button-row card-nav-row">
            <button
              className="secondary-button nav-arrow"
              type="button"
              onClick={onPreviousPage}
              disabled={!list?.has_prev}
              aria-label="Предыдущая страница"
            >
              ←
            </button>
            <button
              className="secondary-button nav-arrow"
              type="button"
              onClick={onNextPage}
              disabled={!list?.has_next}
              aria-label="Следующая страница"
            >
              →
            </button>
          </div>
        </section>
      ) : null}
      {mode === "test" ? (
        <section className="glass-card compact-section">
          <div className="section-head">
            <div>
              <p className="overline">Alphabet</p>
              <h3>Тест по алфавиту 🧠</h3>
            </div>
            <span className="status-tag">
              {Math.min(questionCount + 1, sessionLimit)} / {sessionLimit}
            </span>
          </div>
          {question ? (
            <div className="quiz-panel">
              <div className="prompt-card">
                <strong>/{question.letter.transcription}/</strong>
                <span>{question.letter.hint}</span>
              </div>
              <div className="option-grid">
                {question.options.map((option) => (
                  <button
                    key={option}
                    className="option-button"
                    type="button"
                    onClick={() => onAnswer(option)}
                  >
                    {formatDisplayAnswer(option, question.course_code)}
                  </button>
                ))}
              </div>
              {!result ? (
                <button className="secondary-button" type="button" onClick={onSkip}>
                  Пропустить
                </button>
              ) : null}
              {result ? (
                <div className={result.correct ? "result-box good" : "result-box bad"}>
                  <div className="result-copy">
                    <span>
                      {result.correct
                        ? "Верно"
                        : `Правильный ответ: ${formatDisplayAnswer(
                            result.correct_answer,
                            question.course_code,
                          )}`}
                    </span>
                    {result.points_earned ? (
                      <span className="points-burst">
                        🌟 +{result.points_earned} {formatPointsLabel(result.points_earned)}
                      </span>
                    ) : null}
                  </div>
                  <button className="secondary-button" type="button" onClick={onAdvance}>
                    Дальше
                  </button>
                </div>
              ) : null}
            </div>
          ) : (
            <div className="stack-form">
              <div className="empty-state">
                {sessionDone
                  ? `Тест завершён. Верно ${correctCount} из ${questionCount || sessionLimit}. 🌟 +${correctCount} ${formatPointsLabel(correctCount)}. ${getSessionPraise(correctCount, questionCount || sessionLimit)}`
                  : "Сейчас нет вопроса по алфавиту."}
              </div>
              {sessionDone ? (
                <button className="primary-button" type="button" onClick={onStartNewSession}>
                  Начать новый тест
                </button>
              ) : null}
            </div>
          )}
        </section>
      ) : null}
    </div>
  );
}
