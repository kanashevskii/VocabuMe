import { MAX_ADD_BATCH_WORDS } from "../constants";

export default function AddWordsScreen({
  activeStudiedLanguage,
  addBusy,
  addBusyLabel,
  addDraft,
  addDraftStep,
  addDrafts,
  addText,
  addTranslationInput,
  batchTranslations,
  closeAddWords,
  confirmDraftTranslation,
  draftImageVersions,
  formatDisplayLine,
  handleAddWords,
  openPacks,
  regenerateBatchDraftImage,
  regenerateDraftImage,
  saveBatchDrafts,
  saveDraft,
  setAddText,
  setAddTranslationInput,
  setBatchTranslations,
}) {
  const isTranslationStep = addDraftStep === "confirm_translation";
  const isImageStep = addDraftStep === "confirm_image";
  const isBatchReview = addDraftStep === "batch_review";
  const addWordPlaceholder =
    activeStudiedLanguage === "ka"
      ? "გამარჯობა\nმადლობა\nგზა - дорога"
      : "stare\nfigure out\ntravel - путешествие";
  const addWordHint =
    "По одному слову или фразе на строку. Перевод можно указать через дефис.";

  return (
    <section className="glass-card compact-section add-wizard">
      <div className="section-head">
        <div>
          <p className="overline">Добавление</p>
          <h3>{isBatchReview ? "Проверить слова ✨" : "Добавить слово ✨"}</h3>
          <p className="lead compact">
            {isBatchReview
              ? "Проверь переводы. Фото загружаются автоматически и не тормозят добавление."
              : `До ${MAX_ADD_BATCH_WORDS} слов или фраз за раз.`}
          </p>
        </div>
        <button
          className="secondary-button"
          type="button"
          onClick={closeAddWords}
          disabled={addBusy}
        >
          Закрыть
        </button>
      </div>
      <div className="wizard-steps">
        <span
          className={
            addDraftStep === "input" ? "mode-pill active-pill" : "mode-pill"
          }
        >
          1. Слово
        </span>
        <span
          className={
            isTranslationStep || isBatchReview
              ? "mode-pill active-pill"
              : "mode-pill"
          }
        >
          2. Перевод
        </span>
        <span
          className={
            isImageStep || isBatchReview ? "mode-pill active-pill" : "mode-pill"
          }
        >
          3. Картинка
        </span>
      </div>

      {addBusyLabel ? (
        <div className="inline-note status-note">
          <strong>{addBusyLabel}</strong>
        </div>
      ) : null}

      {!addDraft && !isBatchReview ? (
        <div className="stack-form">
          <form className="stack-form" onSubmit={handleAddWords}>
            <div className="inline-note">{addWordHint}</div>
            <textarea
              rows={4}
              value={addText}
              onChange={(event) => setAddText(event.target.value)}
              placeholder={addWordPlaceholder}
            />
            <button className="primary-button" type="submit" disabled={addBusy}>
              {addBusy ? "Обрабатываем..." : "Добавить слово"}
            </button>
          </form>
          <button
            className="secondary-button"
            type="button"
            onClick={openPacks}
            disabled={addBusy}
          >
            Открыть наборы
          </button>
        </div>
      ) : null}

      {isBatchReview && addDrafts.length ? (
        <div className="word-list batch-draft-list">
          <div className="inline-note status-note">
            <strong>Можно сохранять слова сразу.</strong> Если часть фото ещё не
            появилась, они догрузятся автоматически позже и подтянутся в
            карточках.
          </div>
          {addDrafts.map((draft) => {
            const displayLine = formatDisplayLine(
              draft.word,
              draft.course_code
            );
            const imageReady =
              draft.has_image &&
              draftImageVersions[draft.id] === draft.updated_at;
            return (
              <article className="glass-card word-item" key={draft.id}>
                <div className="word-item-head">
                  <div>
                    <strong>{draft.word}</strong>
                    {displayLine.secondary ? (
                      <p className="word-item-romanization">
                        {displayLine.secondary}
                      </p>
                    ) : null}
                    <p>{draft.example}</p>
                  </div>
                  <span className="status-tag">
                    {draft.part_of_speech || "word"}
                  </span>
                </div>
                <input
                  value={batchTranslations[draft.id] ?? draft.translation}
                  onChange={(event) =>
                    setBatchTranslations((current) => ({
                      ...current,
                      [draft.id]: event.target.value,
                    }))
                  }
                />
                {imageReady ? (
                  <div className="word-image-preview">
                    <img
                      key={draftImageVersions[draft.id]}
                      src={`/api/draft-image/${draft.id}?v=${draftImageVersions[draft.id]}`}
                      alt={draft.word}
                    />
                  </div>
                ) : (
                  <div className="empty-card">
                    {draft.image_generation_in_progress
                      ? "Фото готовится автоматически. Можно не ждать и сохранить слова сразу."
                      : draft.has_image
                        ? "Фото уже почти готово к показу..."
                        : "Фото появится автоматически позже."}
                  </div>
                )}
                <div className="button-row">
                  {draft.has_image ? (
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() => regenerateBatchDraftImage(draft.id)}
                      disabled={addBusy}
                    >
                      ♻️ Другое фото
                    </button>
                  ) : null}
                </div>
              </article>
            );
          })}
          <div className="button-row batch-save-row">
            <button
              className="primary-button"
              type="button"
              onClick={saveBatchDrafts}
              disabled={addBusy}
            >
              {addBusy ? "Сохраняем..." : "Сохранить всё"}
            </button>
          </div>
        </div>
      ) : null}

      {addDraft && isTranslationStep ? (
        <div className="draft-card">
          <div className="prompt-card">
            <strong>{addDraft.word}</strong>
            {formatDisplayLine(addDraft.word, addDraft.course_code)
              .secondary ? (
              <span className="word-romanization inline-romanization">
                {
                  formatDisplayLine(addDraft.word, addDraft.course_code)
                    .secondary
                }
              </span>
            ) : null}
            <span>{addDraft.part_of_speech || "word"}</span>
          </div>
          <div className="stack-form">
            <label className="stack-label">
              <span>Подтверди перевод</span>
              <input
                value={addTranslationInput}
                onChange={(event) => setAddTranslationInput(event.target.value)}
                placeholder="Перевод"
              />
            </label>
            <div className="button-row">
              <button
                className="primary-button"
                type="button"
                onClick={confirmDraftTranslation}
                disabled={addBusy}
              >
                {addBusy ? "Проверяем..." : "Подтвердить перевод"}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {addDraft && isImageStep ? (
        <div className="draft-card">
          <div className="draft-preview-grid">
            <div className="study-main">
              {addDraft.has_image &&
              draftImageVersions[addDraft.id] === addDraft.updated_at ? (
                <div className="card-visual">
                  <img
                    key={draftImageVersions[addDraft.id]}
                    src={`/api/draft-image/${addDraft.id}?v=${draftImageVersions[addDraft.id]}`}
                    alt={addDraft.word}
                  />
                </div>
              ) : (
                <div className="empty-card">
                  {addDraft.image_generation_in_progress
                    ? "Фото готовится автоматически. Можно сохранить слово, оно появится позже."
                    : addDraft.has_image
                      ? "Фото уже почти готово к показу..."
                      : "Фото появится автоматически позже."}
                </div>
              )}
            </div>
            <div className="study-side">
              <strong>{addDraft.word}</strong>
              {formatDisplayLine(addDraft.word, addDraft.course_code)
                .secondary ? (
                <span className="word-romanization inline-romanization">
                  {
                    formatDisplayLine(addDraft.word, addDraft.course_code)
                      .secondary
                  }
                </span>
              ) : null}
              <span>{addTranslationInput || addDraft.translation}</span>
              <span>{addDraft.part_of_speech || "word"}</span>
              {addDraft.example ? <span>{addDraft.example}</span> : null}
              <div className="button-row draft-action-row">
                <button
                  className="primary-button"
                  type="button"
                  onClick={() => saveDraft(true)}
                  disabled={addBusy}
                >
                  {addBusy ? "Сохраняем..." : "Сохранить"}
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={regenerateDraftImage}
                  disabled={addBusy}
                >
                  Другое изображение
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
