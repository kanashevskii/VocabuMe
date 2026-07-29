import { useMemo } from "react";

const DIFFICULTY_RANK = {
  Легкий: 0,
  Средний: 1,
  Сложный: 2,
};

function sortPacks(packs) {
  return [...packs]
    .map((pack) => ({
      ...pack,
      levels: [...pack.levels].sort((left, right) => {
        const leftDifficulty = DIFFICULTY_RANK[left.difficulty] ?? 9;
        const rightDifficulty = DIFFICULTY_RANK[right.difficulty] ?? 9;
        if (leftDifficulty !== rightDifficulty) {
          return leftDifficulty - rightDifficulty;
        }
        return left.title.localeCompare(right.title, "ru");
      }),
    }))
    .sort((left, right) => {
      const leftDifficulty = DIFFICULTY_RANK[left.difficulty] ?? 9;
      const rightDifficulty = DIFFICULTY_RANK[right.difficulty] ?? 9;
      if (leftDifficulty !== rightDifficulty) {
        return leftDifficulty - rightDifficulty;
      }
      const leftPriority = left.id === "travel" ? 2 : 1;
      const rightPriority = right.id === "travel" ? 2 : 1;
      if (leftPriority !== rightPriority) {
        return leftPriority - rightPriority;
      }
      return left.title.localeCompare(right.title, "ru");
    });
}

export default function PacksScreen({
  addBusy,
  billing,
  isPackExpanded,
  onAddSelectedPack,
  onOpenAddWords,
  onPremiumPackLocked,
  packs,
  packsLoading,
  selectedPackId,
  selectedPackLevelId,
  selectedPackWords,
  setIsPackExpanded,
  setSelectedPackId,
  setSelectedPackLevelId,
  setSelectedPackWords,
}) {
  const displayPacks = useMemo(() => sortPacks(packs), [packs]);
  const selectedPack =
    displayPacks.find((pack) => pack.id === selectedPackId)
    || displayPacks[0]
    || null;
  const selectedLevel =
    selectedPack?.levels.find((level) => level.id === selectedPackLevelId)
    || selectedPack?.levels?.[0]
    || null;
  const selectedWordCount = selectedLevel
    ? selectedLevel.items.filter(
        (item) => selectedPackWords[item.normalized_word] ?? !item.already_added,
      ).length
    : 0;

  return (
    <section className="glass-card compact-section pack-section">
      <div className="section-head pack-section-head">
        <div>
          <p className="overline">Сценарии</p>
          <h3>Сценарии для переезда ✈️</h3>
          <p className="lead compact">
            Выбери ситуацию: банк, документы, жилье, магазин, почта.
          </p>
        </div>
        <button
          className="secondary-button pack-manual-button"
          type="button"
          onClick={onOpenAddWords}
        >
          Свои слова и фразы
        </button>
      </div>
      {displayPacks.length ? (
        <div className="pack-card-list">
          {displayPacks.map((pack) => {
            const isActivePack = selectedPackId === pack.id;
            const totalWords = pack.levels.reduce(
              (sum, level) => sum + level.size,
              0,
            );
            const activeScenario =
              pack.levels.find((level) => level.id === selectedPackLevelId)
              || pack.levels[0];
            const isLockedPack =
              Boolean(pack.premium_required) && !billing.premium_active;

            return (
              <article
                key={pack.id}
                className={isActivePack ? "pack-card active" : "pack-card"}
              >
                <div className="pack-card-copy">
                  <div className="pack-card-title-row">
                    <span className="pack-card-emoji">{pack.emoji}</span>
                    <strong>{pack.title}</strong>
                  </div>
                  <p className="pack-card-description">{pack.description}</p>
                </div>
                <button
                  className={
                    isActivePack && isPackExpanded
                      ? "secondary-button pack-open-button"
                      : "primary-button pack-open-button"
                  }
                  type="button"
                  onClick={() => {
                    if (isLockedPack) {
                      onPremiumPackLocked();
                      return;
                    }
                    setSelectedPackId(pack.id);
                    setSelectedPackLevelId(pack.levels[0]?.id || "");
                    setSelectedPackWords({});
                    setIsPackExpanded((current) =>
                      isActivePack ? !current : true,
                    );
                  }}
                >
                  {isLockedPack
                    ? "Premium"
                    : isActivePack && isPackExpanded
                      ? "Скрыть"
                      : "Открыть"}
                </button>
                <div className="pack-card-meta">
                  {pack.difficulty ? (
                    <span className="pack-badge pack-badge-difficulty">
                      {pack.difficulty}
                    </span>
                  ) : null}
                  <span className="pack-badge">
                    {pack.levels.length === 1
                      ? "1 ситуация"
                      : `${pack.levels.length} ситуации`}
                  </span>
                  <span className="pack-badge">{totalWords} слов и фраз</span>
                  {isLockedPack ? (
                    <span className="pack-badge">Premium</span>
                  ) : null}
                  {pack.has_added_words ? (
                    <span className="pack-badge pack-badge-success">Добавлен</span>
                  ) : null}
                </div>
                {isActivePack && isPackExpanded ? (
                  <>
                    {pack.levels.length > 1 ? (
                      <div className="pack-scenario-list">
                        {pack.levels.map((level) => (
                          <button
                            key={level.id}
                            className={
                              selectedPackLevelId === level.id
                                ? "pack-scenario-card active"
                                : "pack-scenario-card"
                            }
                            type="button"
                            onClick={() => {
                              setSelectedPackLevelId(level.id);
                              setSelectedPackWords({});
                            }}
                          >
                            <span className="pack-scenario-head">
                              <strong>{level.title}</strong>
                              {selectedPackLevelId === level.id ? (
                                <span className="pack-scenario-selected">Выбрано</span>
                              ) : null}
                            </span>
                            <small>{level.description}</small>
                            <span className="pack-scenario-meta">
                              {level.difficulty ? `${level.difficulty} · ` : ""}
                              {level.size} слов и фраз
                              {level.has_added_words ? " · Добавлен" : ""}
                            </span>
                          </button>
                        ))}
                      </div>
                    ) : null}
                    <p className="inline-note pack-level-note">
                      {activeScenario?.description}
                    </p>
                    <div className="pack-word-grid">
                      {activeScenario?.items.map((item) => {
                        const checked =
                          selectedPackWords[item.normalized_word]
                          ?? !item.already_added;
                        return (
                          <label
                            key={item.normalized_word}
                            className={
                              item.already_added
                                ? "pack-word-row muted"
                                : "pack-word-row"
                            }
                          >
                            <input
                              type="checkbox"
                              checked={checked}
                              disabled={item.already_added}
                              onChange={(event) =>
                                setSelectedPackWords((current) => ({
                                  ...current,
                                  [item.normalized_word]: event.target.checked,
                                }))
                              }
                            />
                            <span className="pack-word-main">
                              <strong>{item.word}</strong>
                              <small>{item.translation}</small>
                            </span>
                            {item.already_added ? (
                              <em className="pack-word-state">Уже есть</em>
                            ) : null}
                          </label>
                        );
                      })}
                    </div>
                    <div className="button-row batch-save-row">
                      <button
                        className="primary-button"
                        type="button"
                        onClick={onAddSelectedPack}
                        disabled={addBusy || !selectedWordCount}
                      >
                        {addBusy
                          ? "Добавляем..."
                          : `Добавить ${selectedWordCount} слов и фраз`}
                      </button>
                    </div>
                  </>
                ) : null}
              </article>
            );
          })}
        </div>
      ) : packsLoading ? (
        <div className="empty-state">Загружаем наборы...</div>
      ) : null}
    </section>
  );
}
