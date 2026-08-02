import { ALPHABET_MODES, IRREGULAR_MODES } from "../constants";

export default function LearningOverviewScreen({
  alphabetMode,
  hasWordsToLearn,
  irregularMode,
  learnCorrectCount,
  learnQuestionCount,
  learnSessionDone,
  learnSessionLimit,
  onOpenPacks,
  onSelectAlphabetMode,
  onSelectIrregularMode,
  onStartLearning,
  supportsIrregularPractice,
  getSessionPraise,
}) {
  return (
    <div className="screen-stack">
      <section className="glass-card compact-section practice-overview-card">
        <div className="section-head">
          <div>
            <p className="overline">Practice</p>
            <h3>Учить слова 🎯</h3>
            <p className="lead compact">
              {learnSessionDone
                ? getSessionPraise(learnCorrectCount, learnQuestionCount)
                : hasWordsToLearn
                  ? "Случайные задания по словам, которые ты сейчас изучаешь."
                  : "Сейчас нет подходящих заданий, потому что у тебя пока нет новых слов для изучения."}
            </p>
          </div>
        </div>
        {learnSessionDone ? (
          <div className="inline-note status-note">
            <strong>Сессия завершена.</strong> Верно: {learnCorrectCount} из{" "}
            {learnQuestionCount || learnSessionLimit}.
          </div>
        ) : null}
        <div className="button-row">
          {hasWordsToLearn ? (
            <button
              className="primary-button"
              type="button"
              onClick={onStartLearning}
            >
              {learnSessionDone ? "Начать новую сессию" : "Начать сессию"}
            </button>
          ) : null}
          <button
            className="secondary-button"
            type="button"
            onClick={onOpenPacks}
          >
            ＋ Добавить слова
          </button>
        </div>
      </section>

      {supportsIrregularPractice ? (
        <section className="glass-card compact-section practice-overview-card">
          <div className="section-head">
            <div>
              <p className="overline">Irregular</p>
              <h3>Неправильные глаголы 📘</h3>
              <p className="lead compact">
                Можно быстро повторять формы или пройти отдельный тест.
              </p>
            </div>
          </div>
          <div className="segment-wrap main-segment">
            {IRREGULAR_MODES.map((item) => (
              <button
                key={item.id}
                className={
                  irregularMode === item.id
                    ? "segment-button active"
                    : "segment-button"
                }
                type="button"
                onClick={() => onSelectIrregularMode(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </section>
      ) : null}

      <section className="glass-card compact-section practice-overview-card">
        <div className="section-head">
          <div>
            <p className="overline">Alphabet</p>
            <h3>Алфавит 🔤</h3>
            <p className="lead compact">
              Буквы, названия и транскрипция для текущего языка обучения.
            </p>
          </div>
        </div>
        <div className="segment-wrap main-segment">
          {ALPHABET_MODES.map((item) => (
            <button
              key={item.id}
              className={
                alphabetMode === item.id
                  ? "segment-button active"
                  : "segment-button"
              }
              type="button"
              onClick={() => onSelectAlphabetMode(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
