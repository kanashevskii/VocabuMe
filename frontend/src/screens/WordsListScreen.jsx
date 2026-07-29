export default function WordsListScreen({
  draftTranslation,
  expandedWordId,
  formatDisplayLine,
  onDeleteWord,
  onOpenPacks,
  onRegenerateWordImage,
  onSaveTranslation,
  previewWordId,
  regeneratingWordId,
  search,
  settings,
  setDraftTranslation,
  setExpandedWordId,
  setPreviewWordId,
  setSearch,
  setStatusFilter,
  setWordImageErrors,
  statusFilter,
  wordImageErrors,
  wordImageVersions,
  words,
}) {
  return (
    <>
      <div className="glass-card compact-section">
        <div className="filters">
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search"
          />
          <select
            value={statusFilter}
            onChange={(event) => setStatusFilter(event.target.value)}
          >
            <option value="all">Все</option>
            <option value="learning">В процессе</option>
            <option value="learned">Выучено</option>
          </select>
        </div>
        <div className="button-row">
          <button className="secondary-button" type="button" onClick={onOpenPacks}>
            + Добавить записи
          </button>
        </div>
      </div>
      <div className="word-list">
        {words.map((item) => {
          const displayLine = formatDisplayLine(item.word, item.course_code);
          const imageVersion = wordImageVersions[item.id] || item.updated_at;
          const isExpanded = expandedWordId === item.id;
          const isPreviewing = previewWordId === item.id;
          const isRegenerating =
            regeneratingWordId === item.id || item.image_generation_in_progress;

          return (
            <article className="glass-card word-item" key={item.id}>
              <div className="word-item-head">
                <div>
                  <strong>{item.word}</strong>
                  {displayLine.secondary ? (
                    <p className="word-item-romanization">{displayLine.secondary}</p>
                  ) : null}
                  <p className="word-item-example">{item.example}</p>
                </div>
                <span className={item.is_learned ? "status-tag good" : "status-tag"}>
                  {item.is_learned
                    ? "Выучено"
                    : `${item.correct_count}/${settings?.exercise_goal || 4}`}
                </span>
              </div>
              <input
                value={draftTranslation[item.id] ?? item.translation}
                onChange={(event) =>
                  setDraftTranslation((current) => ({
                    ...current,
                    [item.id]: event.target.value,
                  }))
                }
              />
              <div className="button-row word-item-actions word-item-actions-primary">
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => onSaveTranslation(item.id)}
                >
                  Сохранить
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => {
                    setExpandedWordId((current) =>
                      current === item.id ? null : item.id,
                    );
                    setPreviewWordId((current) =>
                      current === item.id ? null : current,
                    );
                  }}
                >
                  {isExpanded ? "Скрыть" : "Ещё"}
                </button>
              </div>
              {isExpanded ? (
                <div className="word-item-extra">
                  <div className="button-row word-item-actions">
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() =>
                        setPreviewWordId((current) =>
                          current === item.id ? null : item.id,
                        )
                      }
                    >
                      {isPreviewing ? "🫥 Скрыть фото" : "🖼 Фото"}
                    </button>
                    <button
                      className={
                        isRegenerating
                          ? "secondary-button is-loading"
                          : "secondary-button"
                      }
                      type="button"
                      onClick={() => onRegenerateWordImage(item.id)}
                      disabled={isRegenerating}
                    >
                      {isRegenerating ? "⏳ Генерируем..." : "♻️ Обновить фото"}
                    </button>
                    <button
                      className="danger-button"
                      type="button"
                      onClick={() => onDeleteWord(item.id)}
                    >
                      Удалить
                    </button>
                  </div>
                  {item.image_generation_in_progress ? (
                    <div className="inline-note status-note">
                      <strong>Генерируем новое фото...</strong> Старое изображение
                      останется до обновления.
                    </div>
                  ) : null}
                  {isPreviewing ? (
                    item.has_image && !wordImageErrors[item.id] ? (
                      <div className="word-image-preview">
                        <img
                          key={imageVersion}
                          src={`/api/image/${item.id}?v=${imageVersion}`}
                          alt={item.word}
                          onLoad={() =>
                            setWordImageErrors((current) => ({
                              ...current,
                              [item.id]: false,
                            }))
                          }
                          onError={() =>
                            setWordImageErrors((current) => ({
                              ...current,
                              [item.id]: true,
                            }))
                          }
                        />
                      </div>
                    ) : (
                      <div className="empty-card">
                        {item.image_generation_in_progress
                          ? "Новое изображение ещё готовится."
                          : "Изображение недоступно. Попробуй обновить фото ещё раз."}
                      </div>
                    )
                  ) : null}
                </div>
              ) : null}
            </article>
          );
        })}
        {!words.length ? (
          <div className="glass-card empty-card">
            <p>Пока записей нет.</p>
            <button className="secondary-button" type="button" onClick={onOpenPacks}>
              Открыть наборы
            </button>
          </div>
        ) : null}
      </div>
    </>
  );
}
