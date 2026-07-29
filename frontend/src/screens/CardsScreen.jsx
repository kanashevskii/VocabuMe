export default function CardsScreen({
  audioRevision,
  cardIndex,
  cardQueue,
  cardReveal,
  currentCard,
  formatDisplayLine,
  onNext,
  onOpenPacks,
  onPrevious,
  onReveal,
}) {
  const displayLine = currentCard
    ? formatDisplayLine(currentCard.word, currentCard.course_code)
    : null;

  return (
    <section className="glass-card learn-card">
      <div className="section-head">
        <div>
          <p className="overline">Cards</p>
          <h3>Карточки 🧠</h3>
        </div>
        <div className="button-row card-nav-row">
          <button
            className="secondary-button nav-arrow"
            type="button"
            onClick={onPrevious}
            disabled={cardIndex === 0}
            aria-label="Предыдущая карточка"
          >
            ←
          </button>
          <button
            className="secondary-button nav-arrow"
            type="button"
            onClick={onNext}
            disabled={!cardQueue.length || cardIndex >= cardQueue.length - 1}
            aria-label="Следующая карточка"
          >
            →
          </button>
        </div>
      </div>
      {currentCard ? (
        <div className="study-layout">
          <div className="study-main">
            {currentCard.has_image ? (
              <div className="card-visual">
                <img
                  src={`/api/image/${currentCard.id}`}
                  alt={currentCard.word}
                  loading="eager"
                />
              </div>
            ) : null}
            <div className="study-meta">
              <span>
                {cardIndex + 1} / {cardQueue.length}
              </span>
              <span>{currentCard.part_of_speech || "word"}</span>
            </div>
            <h2 className="study-word">{displayLine.primary}</h2>
            {displayLine.secondary ? (
              <p className="word-romanization">{displayLine.secondary}</p>
            ) : null}
            <p className="transcription">
              /{currentCard.transcription || "no IPA"}/
            </p>
            <p className="example">{currentCard.example}</p>
            <audio
              controls
              src={`/api/audio/${currentCard.id}?v=${audioRevision}`}
              className="audio-player"
            />
          </div>
          <div className="study-side">
            {cardReveal ? (
              <>
                <strong>{currentCard.translation}</strong>
                <span>{currentCard.example_translation}</span>
                <span className="study-hint">
                  Посмотри перевод и переходи к следующей карточке.
                </span>
                <button
                  className="primary-button"
                  type="button"
                  onClick={onNext}
                >
                  Дальше
                </button>
              </>
            ) : (
              <button
                className="primary-button"
                type="button"
                onClick={onReveal}
              >
                Показать перевод
              </button>
            )}
          </div>
        </div>
      ) : (
        <div className="stack-form">
          <div className="empty-state">
            Пока нет карточек. Добавь новые слова для изучения.
          </div>
          <button
            className="primary-button add-words-ghost"
            type="button"
            onClick={onOpenPacks}
          >
            ＋ Добавить слова
          </button>
        </div>
      )}
    </section>
  );
}
