function GateShell({ isKeyboardOpen, notice, children }) {
  return (
    <div
      className={`app-shell auth-layout${isKeyboardOpen ? " keyboard-open" : ""}`}
    >
      <main className="auth-stage">
        {notice ? <div className="notice">{notice.message}</div> : null}
        {children}
      </main>
    </div>
  );
}

function GeorgianDisplayModeGate({
  busy,
  cancelLabel,
  georgianDisplayModeOptions,
  onCancel,
  onConfirm,
}) {
  return (
    <section className="glass-card compact-section">
      <p className="overline">Грузинский</p>
      <h3>Как показывать грузинский? ✨</h3>
      <p className="lead compact">
        Для старта рекомендуем показывать и грузинское письмо, и латиницу. Позже
        это всегда можно изменить в настройках.
      </p>
      <div className="stack-form">
        {georgianDisplayModeOptions.map((item) => (
          <button
            key={item.code}
            className={item.recommended ? "primary-button" : "secondary-button"}
            type="button"
            onClick={() => onConfirm(item.code)}
            disabled={busy}
          >
            {item.label}
            {item.recommended ? " (Рекомендуется)" : ""}
          </button>
        ))}
        <button
          className="secondary-button"
          type="button"
          onClick={onCancel}
          disabled={busy}
        >
          {cancelLabel}
        </button>
      </div>
    </section>
  );
}

function StudiedLanguageGate({ availableLanguages, busy, onSelectLanguage }) {
  return (
    <section className="glass-card compact-section">
      <p className="overline">Первый запуск</p>
      <h3>Какой язык ты хочешь учить? ✨</h3>
      <p className="lead compact">
        Сначала выбери язык обучения. Прогресс, слова и готовые наборы будут
        храниться отдельно для каждого языка.
      </p>
      <div className="pack-list">
        {availableLanguages.map((item) => (
          <button
            key={item.code}
            className="segment-button active"
            type="button"
            onClick={() => onSelectLanguage(item.code)}
            disabled={busy}
          >
            {busy ? "Сохраняем..." : item.label}
          </button>
        ))}
      </div>
    </section>
  );
}

function OnboardingIntro({ activeLanguageLabel, busy, onComplete, onNext }) {
  return (
    <section className="glass-card compact-section">
      <p className="overline">Старт</p>
      <h3>VocabuMe для жизни после переезда ✨</h3>
      <p className="lead compact">
        Ты выбрал {activeLanguageLabel.toLowerCase()}. Дальше можно быстро
        закрывать реальные задачи: банк, документы, жилье, связь, рынок, почта.
      </p>
      <div className="stack-form">
        <div className="inline-note">
          1. Выбери ситуацию, которая нужна прямо сейчас.
          <br />
          2. Добавь нужные слова и фразы в словарь.
          <br />
          3. Пройди практику перед реальным диалогом.
        </div>
        <button
          className="primary-button"
          type="button"
          onClick={onNext}
          disabled={busy}
        >
          Дальше
        </button>
        <button
          className="secondary-button"
          type="button"
          onClick={onComplete}
          disabled={busy}
        >
          Сразу открыть наборы
        </button>
      </div>
    </section>
  );
}

function OnboardingPremium({
  billing,
  busy,
  checkoutBusyPeriod,
  monetization,
  onBack,
  onComplete,
  onStartCheckout,
}) {
  const monthlyAmount =
    monetization.plans?.premium?.price?.monthly?.amount || "6.99";
  const yearlyAmount =
    monetization.plans?.premium?.price?.yearly?.amount || "39.99";

  return (
    <section className="glass-card compact-section onboarding-premium-card">
      <p className="overline">Premium</p>
      <h3>Полный доступ к сценариям для переезда 🚀</h3>
      <p className="lead compact">
        Для экспатов это не просто слова, а готовые сценарии для жизни после
        переезда: банк, аренда, счета, документы и бытовые вопросы.
      </p>
      <div className="simple-list onboarding-premium-list">
        <div className="simple-row">
          <strong>Полный доступ к сценариям для переезда</strong>
          <small>
            Банк, аренда, счета, документы, магазин, почта и другие
            expat-сценарии.
          </small>
        </div>
        <div className="simple-row">
          <strong>Больше слов и фраз под твою ситуацию</strong>
          <small>Без ограничения free-плана по дневному добавлению.</small>
        </div>
        <div className="simple-row">
          <strong>AI для сложных бытовых диалогов</strong>
          <small>
            Объяснения, дополнительные генерации и будущие диалоги с фидбеком по
            темам переезда.
          </small>
        </div>
      </div>
      <div className="pack-list onboarding-premium-prices">
        <span className="pack-badge">${monthlyAmount} / месяц</span>
        <span className="pack-badge">${yearlyAmount} / год</span>
      </div>
      {!billing.premium_active ? (
        <div className="button-row onboarding-premium-buy-row">
          <button
            className="primary-button"
            type="button"
            onClick={() => onStartCheckout("monthly")}
            disabled={busy || Boolean(checkoutBusyPeriod)}
          >
            {checkoutBusyPeriod === "monthly"
              ? "Открываем оплату..."
              : "Купить доступ · месяц"}
          </button>
          <button
            className="secondary-button"
            type="button"
            onClick={() => onStartCheckout("yearly")}
            disabled={busy || Boolean(checkoutBusyPeriod)}
          >
            {checkoutBusyPeriod === "yearly"
              ? "Открываем оплату..."
              : "Купить доступ · год"}
          </button>
        </div>
      ) : (
        <div className="inline-note">
          Premium уже активен. Можно сразу переходить к наборам.
        </div>
      )}
      <div className="stack-form onboarding-premium-actions">
        <button
          className="primary-button"
          type="button"
          onClick={onComplete}
          disabled={busy}
        >
          Начать бесплатно
        </button>
        <button
          className="secondary-button"
          type="button"
          onClick={onBack}
          disabled={busy}
        >
          Назад
        </button>
      </div>
    </section>
  );
}

export default function OnboardingGate({
  activeLanguageLabel,
  availableLanguages,
  billing,
  busy,
  checkoutBusyPeriod,
  georgianDisplayModeOptions,
  georgianDisplayModePrompt,
  isKeyboardOpen,
  monetization,
  needsOnboarding,
  needsStudiedLanguageSelection,
  notice,
  onboardingStep,
  onCancelGeorgianDisplayMode,
  onCompleteOnboarding,
  onConfirmGeorgianDisplayMode,
  onSelectLanguage,
  onSetOnboardingStep,
  onStartCheckout,
}) {
  if (georgianDisplayModePrompt) {
    return (
      <GateShell isKeyboardOpen={isKeyboardOpen} notice={notice}>
        <GeorgianDisplayModeGate
          busy={busy}
          cancelLabel={needsStudiedLanguageSelection ? "Назад" : "Отмена"}
          georgianDisplayModeOptions={georgianDisplayModeOptions}
          onCancel={onCancelGeorgianDisplayMode}
          onConfirm={onConfirmGeorgianDisplayMode}
        />
      </GateShell>
    );
  }

  if (needsStudiedLanguageSelection) {
    return (
      <GateShell isKeyboardOpen={isKeyboardOpen} notice={notice}>
        <StudiedLanguageGate
          availableLanguages={availableLanguages}
          busy={busy}
          onSelectLanguage={onSelectLanguage}
        />
      </GateShell>
    );
  }

  if (!needsOnboarding) {
    return null;
  }

  const completeOnboarding = () => onCompleteOnboarding({ openPacks: true });
  return (
    <GateShell isKeyboardOpen={isKeyboardOpen} notice={notice}>
      {onboardingStep === "intro" ? (
        <OnboardingIntro
          activeLanguageLabel={activeLanguageLabel}
          busy={busy}
          onComplete={completeOnboarding}
          onNext={() => onSetOnboardingStep("premium")}
        />
      ) : (
        <OnboardingPremium
          billing={billing}
          busy={busy}
          checkoutBusyPeriod={checkoutBusyPeriod}
          monetization={monetization}
          onBack={() => onSetOnboardingStep("intro")}
          onComplete={completeOnboarding}
          onStartCheckout={onStartCheckout}
        />
      )}
    </GateShell>
  );
}
