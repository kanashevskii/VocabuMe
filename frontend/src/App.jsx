import { startTransition, useDeferredValue, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

const PRIMARY_TABS = [
  { id: "today", label: "Сегодня" },
  { id: "learn", label: "Практика" },
  { id: "words", label: "Слова" },
  { id: "progress", label: "Прогресс" },
  { id: "more", label: "Ещё" }
];

const LIBRARY_MODES = [
  { id: "cards", label: "Карточки" },
  { id: "words", label: "Все слова" }
];

const MORE_PANELS = [
  { id: "review", label: "Повтор" },
  { id: "irregular", label: "Глаголы" },
  { id: "settings", label: "Настройки" }
];

const IRREGULAR_MODES = [
  { id: "review", label: "Повторять" },
  { id: "test", label: "Тест" }
];

function getCookie(name) {
  const match = document.cookie.match(new RegExp(`(^| )${name}=([^;]+)`));
  return match ? decodeURIComponent(match[2]) : "";
}

async function reportClientError(payload) {
  try {
    const telegramInitData = window.Telegram?.WebApp?.initData || "";
    const headers = { "Content-Type": "application/json" };
    if (telegramInitData) {
      headers["X-Telegram-Init-Data"] = telegramInitData;
    }
    const csrf = getCookie("csrftoken");
    if (csrf) {
      headers["X-CSRFToken"] = csrf;
    }
    await fetch("/api/client-error", {
      method: "POST",
      credentials: "include",
      headers,
      body: JSON.stringify(payload)
    });
  } catch {
    // best-effort logging only
  }
}

async function api(url, options = {}) {
  const telegramInitData = window.Telegram?.WebApp?.initData || "";
  const isFormData = typeof FormData !== "undefined" && options.body instanceof FormData;
  const headers = {
    ...(isFormData ? {} : { "Content-Type": "application/json" }),
    ...(options.headers || {})
  };
  if (telegramInitData) {
    headers["X-Telegram-Init-Data"] = telegramInitData;
  }
  const method = (options.method || "GET").toUpperCase();
  if (!["GET", "HEAD", "OPTIONS", "TRACE"].includes(method)) {
    headers["X-CSRFToken"] = getCookie("csrftoken");
  }

  let response;
  try {
    response = await fetch(url, { credentials: "include", ...options, headers });
  } catch (error) {
    await reportClientError({
      category: "network",
      message: error?.message || "Network request failed",
      url,
      detail: String(error),
      meta: { method }
    });
    throw error;
  }
  const rawText = await response.text();
  let data = {};
  try {
    data = rawText ? JSON.parse(rawText) : {};
  } catch {
    data = {};
  }
  if (!response.ok) {
    let errorMessage = data.error || rawText || `Request failed (${response.status})`;
    if (response.status === 504) {
      errorMessage = "Сервер отвечал слишком долго. Попробуй ещё раз.";
    } else if (typeof errorMessage === "string" && errorMessage.includes("<html")) {
      errorMessage = `Ошибка сервера (${response.status}).`;
    }
    await reportClientError({
      category: "api",
      status_code: response.status,
      message: errorMessage.slice(0, 4000),
      url,
      detail: rawText.slice(0, 4000),
      meta: { method }
    });
    throw new Error(errorMessage);
  }
  return data;
}

function LogoMark() {
  return (
    <div className="logo-mark" aria-hidden="true">
      <img src="/static/old_logo.jpg" alt="" />
    </div>
  );
}

const MAX_IMAGE_REGENERATIONS = 3;

function AuthPanel({ config, onOpenLogin, loginLink, loginPending }) {
  return (
    <section className="auth-shell">
      <div className="glass-card auth-copy">
        <div className="brand-lockup">
          <LogoMark />
          <div>
            <p className="overline">VocabuMe</p>
            <h1>Один вход. Один словарь. Один общий прогресс.</h1>
          </div>
        </div>
        <p className="lead">Открой в Telegram или подтверди вход через бота. После этого приложение продолжит с того же места.</p>
        <div className="auth-actions">
          <button className="primary-button" type="button" onClick={onOpenLogin} disabled={loginPending}>
            {loginPending ? "Готовим вход..." : "🚀 Войти"}
          </button>
          {config.webapp_url ? (
            <a className="secondary-button" href={config.webapp_url} target="_blank" rel="noreferrer">
              💬 Mini App
            </a>
          ) : null}
        </div>
        {loginLink ? (
          <div className="inline-note">
            <span>Открой бота и нажми Start.</span>
            <a className="primary-link" href={loginLink} target="_blank" rel="noreferrer">
              🤖 Открыть бота
            </a>
          </div>
        ) : null}
      </div>
      <div className="glass-card auth-preview">
        <div className="auth-preview-grid">
          <article className="mini-pane">
            <span>Учить</span>
            <strong>🧠 Карточки, практика, аудирование</strong>
          </article>
          <article className="mini-pane">
            <span>Словарь</span>
            <strong>📚 Быстрый поиск и редактирование</strong>
          </article>
          <article className="mini-pane">
            <span>Синхронизация</span>
            <strong>🔄 Telegram и web используют один профиль</strong>
          </article>
        </div>
      </div>
    </section>
  );
}

function App() {
  const [config, setConfig] = useState({ bot_username: "", webapp_url: "" });
  const [auth, setAuth] = useState({ loading: true, authenticated: false, user: null, progress: null });
  const [notice, setNoticeState] = useState(null);
  const [busy, setBusy] = useState(false);
  const [primaryTab, setPrimaryTab] = useState("today");
  const [libraryMode, setLibraryMode] = useState("cards");
  const [morePanel, setMorePanel] = useState("review");
  const [showLibraryAdd, setShowLibraryAdd] = useState(false);
  const [dashboard, setDashboard] = useState(null);
  const [settings, setSettings] = useState(null);
  const [words, setWords] = useState([]);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [draftTranslation, setDraftTranslation] = useState({});
  const [previewWordId, setPreviewWordId] = useState(null);
  const [expandedWordId, setExpandedWordId] = useState(null);
  const [wordImageVersions, setWordImageVersions] = useState({});
  const [regeneratingWordId, setRegeneratingWordId] = useState(null);
  const [addText, setAddText] = useState("");
  const [addDraft, setAddDraft] = useState(null);
  const [addDrafts, setAddDrafts] = useState([]);
  const [addDraftStep, setAddDraftStep] = useState("input");
  const [addTranslationInput, setAddTranslationInput] = useState("");
  const [batchTranslations, setBatchTranslations] = useState({});
  const [draftImageVersions, setDraftImageVersions] = useState({});
  const [addBusy, setAddBusy] = useState(false);
  const [addBusyLabel, setAddBusyLabel] = useState("");
  const [cardQueue, setCardQueue] = useState([]);
  const [cardIndex, setCardIndex] = useState(0);
  const [cardReveal, setCardReveal] = useState(false);
  const [learnQuestion, setLearnQuestion] = useState(null);
  const [learnResult, setLearnResult] = useState(null);
  const [learnSelection, setLearnSelection] = useState("");
  const [learnTextAnswer, setLearnTextAnswer] = useState("");
  const [learnUsedWordIds, setLearnUsedWordIds] = useState([]);
  const [learnQuestionCount, setLearnQuestionCount] = useState(0);
  const [learnSessionLimit, setLearnSessionLimit] = useState(12);
  const [learnSessionDone, setLearnSessionDone] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [reviewQuestion, setReviewQuestion] = useState(null);
  const [reviewResult, setReviewResult] = useState(null);
  const [reviewSelection, setReviewSelection] = useState("");
  const [irregularPage, setIrregularPage] = useState(0);
  const [irregularList, setIrregularList] = useState(null);
  const [irregularQuestion, setIrregularQuestion] = useState(null);
  const [irregularResult, setIrregularResult] = useState(null);
  const [irregularMode, setIrregularMode] = useState("review");
  const [loginLink, setLoginLink] = useState("");
  const [loginToken, setLoginToken] = useState("");
  const pollRef = useRef(null);
  const stageRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const speakingChunksRef = useRef([]);
  const noticeTimerRef = useRef(null);
  const deferredSearch = useDeferredValue(search);

  const webApp = window.Telegram?.WebApp;
  const isMiniApp = Boolean(webApp?.initData);
  const canRecordSpeech = Boolean(navigator.mediaDevices?.getUserMedia && window.MediaRecorder);

  useEffect(() => {
    const handleWindowError = (event) => {
      void reportClientError({
        category: "frontend",
        level: "error",
        message: event.message || "Unhandled window error",
        url: window.location.pathname,
        detail: event.error?.stack || `${event.filename || ""}:${event.lineno || ""}:${event.colno || ""}`,
        meta: { source: "window.error" }
      });
    };

    const handleUnhandledRejection = (event) => {
      const reason = event.reason;
      void reportClientError({
        category: "frontend",
        level: "error",
        message: reason?.message || "Unhandled promise rejection",
        url: window.location.pathname,
        detail: reason?.stack || String(reason),
        meta: { source: "unhandledrejection" }
      });
    };

    window.addEventListener("error", handleWindowError);
    window.addEventListener("unhandledrejection", handleUnhandledRejection);
    return () => {
      window.removeEventListener("error", handleWindowError);
      window.removeEventListener("unhandledrejection", handleUnhandledRejection);
    };
  }, []);

  useEffect(() => () => {
    if (noticeTimerRef.current) {
      window.clearTimeout(noticeTimerRef.current);
    }
  }, []);

  const stats = useMemo(() => {
    const progress = dashboard?.progress || auth.progress;
    return [
      { label: "📚 Words", value: progress?.total ?? 0 },
      { label: "✅ Learned", value: progress?.learned ?? 0 },
      { label: "🔄 Learning", value: progress?.learning ?? 0 },
      { label: "🔥 Streak", value: progress?.streak_days ?? 0 }
    ];
  }, [auth.progress, dashboard]);

  const todayStats = useMemo(() => stats.slice(0, 3), [stats]);
  const todayAchievements = useMemo(() => {
    const list = auth.progress?.achievements || [];
    return list.slice(-3);
  }, [auth.progress]);
  const hasMoreAchievements = (auth.progress?.achievements?.length || 0) > todayAchievements.length;

  const currentTitle = useMemo(() => {
    if (primaryTab === "today") return "Сегодня";
    if (primaryTab === "learn") return "Практика";
    if (primaryTab === "words") return showLibraryAdd ? "Добавить" : libraryMode === "cards" ? "Карточки" : "Все слова";
    if (primaryTab === "progress") return "Прогресс";
    return MORE_PANELS.find((item) => item.id === morePanel)?.label || "Ещё";
  }, [libraryMode, morePanel, primaryTab, showLibraryAdd]);

  const filteredRecentWords = dashboard?.recent_words || [];
  const nextCards = dashboard?.next_cards || [];
  const currentCard = cardQueue[cardIndex];
  const noticeScope = useMemo(() => {
    if (!auth.authenticated) {
      return "auth";
    }
    return [
      primaryTab,
      libraryMode,
      morePanel,
      showLibraryAdd ? "add" : "main",
      addDraftStep,
    ].join(":");
  }, [auth.authenticated, primaryTab, libraryMode, morePanel, showLibraryAdd, addDraftStep]);

  useEffect(() => {
    setNoticeState((current) => {
      if (!current) {
        return current;
      }
      if (current.scope === "global" || current.scope === noticeScope) {
        return current;
      }
      return null;
    });
  }, [noticeScope]);

  function clearNotice() {
    if (noticeTimerRef.current) {
      window.clearTimeout(noticeTimerRef.current);
      noticeTimerRef.current = null;
    }
    setNoticeState(null);
  }

  function setNotice(message, options = {}) {
    if (!message) {
      clearNotice();
      return;
    }
    const nextNotice = {
      message,
      scope: options.scope || noticeScope,
      sticky: Boolean(options.sticky),
      ttl: options.ttl ?? (options.error ? 7000 : 4500),
    };
    if (noticeTimerRef.current) {
      window.clearTimeout(noticeTimerRef.current);
      noticeTimerRef.current = null;
    }
    setNoticeState(nextNotice);
    if (!nextNotice.sticky) {
      noticeTimerRef.current = window.setTimeout(() => {
        setNoticeState((current) => (current === nextNotice ? null : current));
        noticeTimerRef.current = null;
      }, nextNotice.ttl);
    }
  }

  useEffect(() => {
    const draftIds = [
      ...(addDraft?.id ? [addDraft.id] : []),
      ...addDrafts.map((item) => item.id)
    ];
    if (!draftIds.length) {
      return;
    }

    const shouldPoll = Boolean(addDraft?.image_generation_in_progress) || addDrafts.some((item) => item.image_generation_in_progress);
    if (!shouldPoll) {
      return;
    }

    const intervalId = window.setInterval(async () => {
      try {
        const responses = await Promise.all(draftIds.map((draftId) => api(`/api/words/draft/${draftId}`)));
        const byId = new Map(responses.map((entry) => [entry.draft.id, entry.draft]));
        if (addDraft?.id) {
          setAddDraft((current) => (current ? byId.get(current.id) || current : current));
        }
        if (addDrafts.length) {
          setAddDrafts((current) => current.map((item) => byId.get(item.id) || item));
        }
      } catch {
        // best-effort polling only
      }
    }, 2000);

    return () => window.clearInterval(intervalId);
  }, [addDraft, addDrafts]);

  async function bootstrap() {
    const [cfg, me] = await Promise.all([api("/api/app-config"), api("/api/auth/me")]);
    setConfig(cfg);
    setAuth({
      loading: false,
      authenticated: me.authenticated,
      user: me.user,
      progress: me.progress
    });
  }

  async function loadDashboard() {
    const [dashboardData, settingsData, wordsData, irregularData] = await Promise.all([
      api("/api/dashboard"),
      api("/api/settings"),
      api(`/api/words?status=${statusFilter}&search=${encodeURIComponent(deferredSearch)}`),
      api(`/api/irregular/list?page=${irregularPage}`)
    ]);
    setDashboard(dashboardData);
    setSettings(settingsData.settings);
    setWords(wordsData.items);
    setIrregularList(irregularData);
  }

  async function loadCards() {
    const data = await api("/api/study/cards?scope=all");
    setCardQueue(data.items);
    setCardIndex(0);
    setCardReveal(false);
  }

  async function loadLearningData() {
    setLearnUsedWordIds([]);
    setLearnQuestionCount(0);
    await Promise.all([loadCards(), loadLearnQuestion([], 0), loadReview()]);
  }

  function showPreviousCard() {
    setCardIndex((value) => Math.max(0, value - 1));
    setCardReveal(false);
  }

  function showNextCard() {
    setCardIndex((value) => Math.min(cardQueue.length - 1, value + 1));
    setCardReveal(false);
  }

  async function loadReview() {
    const data = await api("/api/practice/question?mode=review");
    setReviewQuestion(data.empty ? null : data.question);
    setReviewResult(null);
    setReviewSelection("");
  }

  async function loadLearnQuestion(excludeIds = [], questionCount = 0) {
    const ids = excludeIds.filter(Boolean);
    const data = await api(`/api/learn/question?exclude_ids=${encodeURIComponent(ids.join(","))}`);
    setLearnSessionLimit(data.session_limit || 12);
    setLearnSelection("");
    setLearnTextAnswer("");
    setLearnResult(null);
    setIsRecording(false);
    if (questionCount >= (data.session_limit || 12) || data.empty) {
      setLearnQuestion(null);
      setLearnSessionDone(true);
      return;
    }
    setLearnQuestion(data.question);
    setLearnSessionDone(false);
  }

  async function loadIrregularQuestion() {
    const data = await api("/api/irregular/question");
    setIrregularQuestion(data.question);
    setIrregularResult(null);
  }

  function stopPolling() {
    if (pollRef.current) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }

  useEffect(() => () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
    }
  }, []);

  useEffect(() => {
    bootstrap().catch((error) => {
      setNotice(error.message);
      setAuth({ loading: false, authenticated: false, user: null, progress: null });
    });
    return () => stopPolling();
  }, []);

  useEffect(() => {
    if (!isMiniApp || auth.authenticated || !webApp?.initData) {
      return;
    }
    webApp.ready();
    webApp.expand();
    api("/api/auth/telegram/webapp", {
      method: "POST",
      body: JSON.stringify({ init_data: webApp.initData })
    })
      .then((data) => {
        setAuth({ loading: false, authenticated: true, user: data.user, progress: data.progress });
        setNotice("");
      })
      .catch((error) => {
        setNotice(error.message);
        setAuth((current) => ({ ...current, loading: false }));
      });
  }, [auth.authenticated, isMiniApp, webApp]);

  useEffect(() => {
    if (!auth.authenticated) {
      return;
    }
    Promise.all([loadDashboard(), loadLearningData(), loadIrregularQuestion()])
      .catch((error) => setNotice(error.message));
  }, [auth.authenticated, deferredSearch, statusFilter, irregularPage]);

  useEffect(() => {
    if (!auth.authenticated) {
      return;
    }
    const hasPendingWordImages = words.some((item) => item.image_generation_in_progress) || cardQueue.some((item) => item.image_generation_in_progress);
    if (!hasPendingWordImages) {
      return;
    }
    const intervalId = window.setInterval(() => {
      void loadLearningData();
    }, 4000);
    return () => window.clearInterval(intervalId);
  }, [auth.authenticated, words, cardQueue]);

  useEffect(() => {
    if (!loginToken || auth.authenticated) {
      return;
    }
    stopPolling();
    pollRef.current = window.setInterval(async () => {
      try {
        const data = await api(`/api/auth/telegram/poll/${loginToken}`);
        if (data.authenticated) {
          stopPolling();
          setAuth({ loading: false, authenticated: true, user: data.user, progress: data.progress });
          setLoginToken("");
          setLoginLink("");
          setNotice("");
        }
      } catch (error) {
        stopPolling();
        setNotice(error.message);
      }
    }, 2000);
    return () => stopPolling();
  }, [loginToken, auth.authenticated]);

  useLayoutEffect(() => {
    const stage = stageRef.current;
    if (!stage) {
      return;
    }

    stage.scrollTop = 0;
    const frameId = window.requestAnimationFrame(() => {
      stage.scrollTop = 0;
    });

    return () => window.cancelAnimationFrame(frameId);
  }, [primaryTab, libraryMode, morePanel]);

  async function requestLoginLink() {
    setBusy(true);
    try {
      const data = await api("/api/auth/telegram/request-link", {
        method: "POST",
        body: JSON.stringify({})
      });
      setLoginLink(data.deep_link);
      setLoginToken(data.token);
      setNotice("Открой бота и нажми Start.");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  function resetAddFlow() {
    setAddText("");
    setAddDraft(null);
    setAddDrafts([]);
    setAddDraftStep("input");
    setAddTranslationInput("");
    setBatchTranslations({});
    setDraftImageVersions({});
    setAddBusy(false);
    setAddBusyLabel("");
  }

  async function refreshAfterWordMutation() {
    await Promise.all([loadDashboard(), loadLearningData()]);
  }

  async function handleAddWords(event) {
    event.preventDefault();
    if (!addText.trim()) {
      setNotice("Добавь одно слово или фразу.");
      return;
    }
    setAddBusy(true);
    setAddBusyLabel("Понимаем слово...");
    try {
      const data = await api("/api/words/draft", {
        method: "POST",
        body: JSON.stringify({ text: addText })
      });
      if (data.batch_review) {
        setAddDrafts(data.drafts);
        setAddDraftStep("batch_review");
        setBatchTranslations(Object.fromEntries(data.drafts.map((draft) => [draft.id, draft.translation || ""])));
        setNotice(`Проверь ${data.drafts.length} слов. Фото загрузятся автоматически, можно не ждать.`);
        return;
      }
      if (data.auto_saved) {
        setAuth((previous) => ({ ...previous, progress: data.progress }));
        setShowLibraryAdd(false);
        setLibraryMode("cards");
        setNotice(`Слово ${data.item.word} добавлено из общей библиотеки.`);
        resetAddFlow();
        await refreshAfterWordMutation();
        return;
      }
      setAddDraft(data.draft);
      setAddDraftStep(data.step);
      setAddTranslationInput(data.draft.translation || "");
      setNotice(data.step === "confirm_translation" ? "Подтверди перевод. Фото загрузится автоматически." : "Фото загружается автоматически. Можно сохранить слово сразу.");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setAddBusy(false);
      setAddBusyLabel("");
    }
  }

  async function confirmDraftTranslation() {
    if (!addDraft || !addTranslationInput.trim()) {
      setNotice("Подтверди перевод.");
      return;
    }
    setAddBusy(true);
    setAddBusyLabel("Подтверждаем перевод...");
    try {
      const data = await api(`/api/words/draft/${addDraft.id}/translation`, {
        method: "POST",
        body: JSON.stringify({ translation: addTranslationInput.trim() })
      });
      setAddDraft(data.draft);
      setAddDraftStep(data.step);
      setNotice("Фото загружается автоматически. Можно сохранить слово и не ждать.");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setAddBusy(false);
      setAddBusyLabel("");
    }
  }

  async function regenerateDraftImage() {
    if (!addDraft) {
      return;
    }
    setAddBusy(true);
    setAddBusyLabel(addDraft.has_image ? "Готовим другую картинку..." : "Генерируем фото...");
    try {
      const data = await api(`/api/words/draft/${addDraft.id}/image/regenerate`, {
        method: "POST",
        body: JSON.stringify({})
      });
      const version = `${data.draft.updated_at}-${Date.now()}`;
      await preloadDraftImage(addDraft.id, version);
      setAddDraft(data.draft);
      setDraftImageVersions((current) => ({ ...current, [addDraft.id]: version }));
      setAddDraftStep(data.step);
      setNotice("Показали новый вариант.");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setAddBusy(false);
      setAddBusyLabel("");
    }
  }

  async function saveDraft(useImage) {
    if (!addDraft) {
      return;
    }
    setAddBusy(true);
    setAddBusyLabel("Сохраняем слово...");
    try {
      const data = await api(`/api/words/draft/${addDraft.id}/save`, {
        method: "POST",
        body: JSON.stringify({ use_image: useImage })
      });
      setAuth((previous) => ({ ...previous, progress: data.progress }));
      setShowLibraryAdd(false);
      setLibraryMode("cards");
      setNotice(`Слово ${data.item.word} добавлено.`);
      resetAddFlow();
      await refreshAfterWordMutation();
    } catch (error) {
      setNotice(error.message);
    } finally {
      setAddBusy(false);
      setAddBusyLabel("");
    }
  }

  async function closeAddWords() {
    const draftIds = [addDraft?.id, ...addDrafts.map((item) => item.id)].filter(Boolean);
    for (const draftId of draftIds) {
      try {
        await api(`/api/words/draft/${draftId}`, {
          method: "DELETE"
        });
      } catch (error) {
        setNotice(error.message);
      }
    }
    resetAddFlow();
    setShowLibraryAdd(false);
  }

  async function regenerateBatchDraftImage(draftId) {
    const currentDraft = addDrafts.find((item) => item.id === draftId);
    setAddBusy(true);
    setAddBusyLabel(currentDraft?.has_image ? "Готовим другую картинку..." : "Генерируем фото...");
    try {
      const data = await api(`/api/words/draft/${draftId}/image/regenerate`, {
        method: "POST",
        body: JSON.stringify({})
      });
      const version = `${data.draft.updated_at}-${Date.now()}`;
      await preloadDraftImage(draftId, version);
      setAddDrafts((current) => current.map((item) => (item.id === draftId ? data.draft : item)));
      setDraftImageVersions((current) => ({ ...current, [draftId]: version }));
      setNotice("Картинка обновлена.");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setAddBusy(false);
      setAddBusyLabel("");
    }
  }

  async function saveBatchDrafts() {
    if (!addDrafts.length) {
      return;
    }
    setAddBusy(true);
    setAddBusyLabel("Сохраняем слова...");
    try {
      for (const draft of addDrafts) {
        const translation = (batchTranslations[draft.id] || draft.translation || "").trim();
        if (!translation) {
          throw new Error(`Заполни перевод для ${draft.word}.`);
        }
        let currentDraft = draft;
        if (translation !== (draft.translation || "").trim()) {
          const confirmed = await api(`/api/words/draft/${draft.id}/translation`, {
            method: "POST",
            body: JSON.stringify({ translation })
          });
          currentDraft = confirmed.draft;
        }
        await api(`/api/words/draft/${currentDraft.id}/save`, {
          method: "POST",
          body: JSON.stringify({ use_image: true })
        });
      }
      setShowLibraryAdd(false);
      setLibraryMode("cards");
      setNotice(`Добавлено ${addDrafts.length} слов.`);
      resetAddFlow();
      await refreshAfterWordMutation();
    } catch (error) {
      setNotice(error.message);
    } finally {
      setAddBusy(false);
      setAddBusyLabel("");
    }
  }

  async function handleCardAnswer(correct) {
    const current = cardQueue[cardIndex];
    if (!current) {
      return;
    }
    setBusy(true);
    try {
      const data = await api("/api/study/answer", {
        method: "POST",
        body: JSON.stringify({ word_id: current.id, correct })
      });
      setAuth((previous) => ({ ...previous, progress: data.progress }));
      setDashboard((previous) => (previous ? { ...previous, progress: data.progress } : previous));
      setCardQueue((previous) => previous.map((item) => (item.id === current.id ? { ...item, ...data.item } : item)));
      if (cardIndex + 1 < cardQueue.length) {
        setCardIndex((value) => value + 1);
      }
      setCardReveal(false);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function handlePracticeAnswer(answer, mode, question, setter) {
    if (!question) {
      return;
    }
    setBusy(true);
    try {
      if (mode === "review") {
        setReviewSelection(answer);
      } else {
        setPracticeSelection(answer);
      }
      const data = await api("/api/practice/answer", {
        method: "POST",
        body: JSON.stringify({
          word_id: question.item.id,
          answer,
          mode: question.mode
        })
      });
      setter(data);
      setAuth((previous) => ({ ...previous, progress: data.progress }));
      await loadDashboard();
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  function getPracticeExpectedAnswer(question) {
    if (!question) {
      return "";
    }
    if (question.mode === "reverse") {
      return question.item.word;
    }
    return question.item.translation;
  }

  function revealPracticeAnswer(question, mode, resultSetter) {
    const correctAnswer = getPracticeExpectedAnswer(question);
    if (!correctAnswer) {
      return;
    }
    if (mode === "review") {
      setReviewSelection("");
    } else {
      setPracticeSelection("");
    }
    resultSetter({ correct: false, correct_answer: correctAnswer, skipped: true });
  }

  function getOptionState(option, result, selectedAnswer) {
    if (!result) {
      return "";
    }
    if (option === result.correct_answer) {
      return "is-correct";
    }
    if (selectedAnswer && option === selectedAnswer && !result.correct) {
      return "is-wrong";
    }
    return "";
  }

  function getLearnExpectedAnswer(question = learnQuestion) {
    if (!question) {
      return "";
    }
    if (question.exercise_type === "practice_en_ru" || question.exercise_type === "listening_translate") {
      return question.item.translation;
    }
    return question.item.word;
  }

  function revealLearnAnswer() {
    const correctAnswer = getLearnExpectedAnswer();
    if (!correctAnswer || !learnQuestion) {
      return;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.onstop = null;
      mediaRecorderRef.current.stop();
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    setIsRecording(false);
    setLearnSelection("");
    setLearnResult({ correct: false, correct_answer: correctAnswer, skipped: true, exercise_type: learnQuestion.exercise_type });
  }

  async function advanceLearnSession() {
    if (!learnQuestion) {
      return;
    }
    const nextUsedIds = [...learnUsedWordIds, learnQuestion.item.id];
    const nextCount = learnQuestionCount + 1;
    setLearnUsedWordIds(nextUsedIds);
    setLearnQuestionCount(nextCount);
    await loadLearnQuestion(nextUsedIds, nextCount);
  }

  async function handleLearnChoiceAnswer(answer) {
    if (!learnQuestion) {
      return;
    }
    setBusy(true);
    try {
      setLearnSelection(answer);
      const data = await api("/api/learn/answer", {
        method: "POST",
        body: JSON.stringify({
          word_id: learnQuestion.item.id,
          answer,
          exercise_type: learnQuestion.exercise_type
        })
      });
      setLearnResult(data);
      setAuth((previous) => ({ ...previous, progress: data.progress }));
      await loadDashboard();
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleLearnListeningSubmit(event) {
    event.preventDefault();
    if (!learnQuestion) {
      return;
    }
    setBusy(true);
    try {
      const data = await api("/api/learn/answer", {
        method: "POST",
        body: JSON.stringify({
          word_id: learnQuestion.item.id,
          answer: learnTextAnswer,
          exercise_type: learnQuestion.exercise_type
        })
      });
      setLearnResult(data);
      setAuth((previous) => ({ ...previous, progress: data.progress }));
      await loadDashboard();
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function uploadSpeakingAttempt(blob) {
    if (!learnQuestion) {
      return;
    }
    setBusy(true);
    try {
      const extension = blob.type.includes("mp4") ? "mp4" : "webm";
      const formData = new FormData();
      formData.append("word_id", String(learnQuestion.item.id));
      formData.append("audio", new File([blob], `speech.${extension}`, { type: blob.type || "audio/webm" }));
      const data = await api("/api/speaking/answer", {
        method: "POST",
        body: formData
      });
      setLearnResult(data);
      setAuth((previous) => ({ ...previous, progress: data.progress }));
      await loadDashboard();
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function startSpeakingRecording() {
    if (!learnQuestion || isRecording) {
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      speakingChunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          speakingChunksRef.current.push(event.data);
        }
      };
      recorder.onstop = async () => {
        const blob = new Blob(speakingChunksRef.current, { type: recorder.mimeType || "audio/webm" });
        stream.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
        if (blob.size > 0) {
          await uploadSpeakingAttempt(blob);
        }
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setLearnResult(null);
      setIsRecording(true);
    } catch (error) {
      setNotice(error.message || "Не удалось включить микрофон.");
    }
  }

  function stopSpeakingRecording() {
    if (!mediaRecorderRef.current || mediaRecorderRef.current.state === "inactive") {
      return;
    }
    mediaRecorderRef.current.stop();
    setIsRecording(false);
  }

  async function handleIrregularAnswer(answer) {
    if (!irregularQuestion) {
      return;
    }
    setBusy(true);
    try {
      const data = await api("/api/irregular/answer", {
        method: "POST",
        body: JSON.stringify({
          base: irregularQuestion.verb.base,
          answer,
          correct_pair: irregularQuestion.correct_pair
        })
      });
      setIrregularResult(data);
      setAuth((previous) => ({ ...previous, progress: data.progress }));
      await loadDashboard();
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function saveTranslation(wordId) {
    const translation = draftTranslation[wordId];
    if (!translation?.trim()) {
      return;
    }
    setBusy(true);
    try {
      await api(`/api/words/${wordId}`, {
        method: "PATCH",
        body: JSON.stringify({ translation })
      });
      setNotice("Сохранено.");
      await loadDashboard();
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function deleteWord(wordId) {
    setBusy(true);
    try {
      await api(`/api/words/${wordId}`, { method: "DELETE", body: JSON.stringify({}) });
      await refreshAfterWordMutation();
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function preloadWordImage(wordId, version, attempts = 5) {
    const src = `/api/image/${wordId}?v=${version}`;
    for (let index = 0; index < attempts; index += 1) {
      const loaded = await new Promise((resolve) => {
        const image = new window.Image();
        image.onload = () => resolve(true);
        image.onerror = () => resolve(false);
        image.src = src;
      });
      if (loaded) {
        return true;
      }
      await new Promise((resolve) => setTimeout(resolve, 350));
    }
    return false;
  }

  async function preloadDraftImage(draftId, version, attempts = 5) {
    const src = `/api/draft-image/${draftId}?v=${version}`;
    for (let index = 0; index < attempts; index += 1) {
      const loaded = await new Promise((resolve) => {
        const image = new window.Image();
        image.onload = () => resolve(true);
        image.onerror = () => resolve(false);
        image.src = src;
      });
      if (loaded) {
        return true;
      }
      await new Promise((resolve) => setTimeout(resolve, 350));
    }
    return false;
  }

  async function regenerateWordImage(wordId) {
    setRegeneratingWordId(wordId);
    try {
      const data = await api(`/api/words/${wordId}/image/regenerate`, {
        method: "POST",
        body: JSON.stringify({})
      });
      const version = `${data.item.updated_at}-${Date.now()}`;
      await preloadWordImage(wordId, version);
      setWords((current) => current.map((item) => (item.id === wordId ? data.item : item)));
      setCardQueue((current) => current.map((item) => (item.id === wordId ? data.item : item)));
      setWordImageVersions((current) => ({ ...current, [wordId]: version }));
      setPreviewWordId(wordId);
      setNotice("Изображение обновлено.");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setRegeneratingWordId(null);
    }
  }

  async function saveSettings(event) {
    event.preventDefault();
    setBusy(true);
    try {
      await api("/api/settings", {
        method: "POST",
        body: JSON.stringify(settings)
      });
      setNotice("Настройки сохранены.");
      await Promise.all([loadDashboard(), loadLearningData()]);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  function openLearn(mode) {
    if (showLibraryAdd) {
      void closeAddWords();
    }
    startTransition(() => {
      setPrimaryTab("learn");
      setShowLibraryAdd(false);
    });
    if (learnSessionDone || !learnQuestion) {
      void loadLearningData();
    }
  }

  function openMore(panel) {
    if (showLibraryAdd) {
      void closeAddWords();
    }
    startTransition(() => {
      setPrimaryTab("more");
      setMorePanel(panel);
      setShowLibraryAdd(false);
    });
  }

  function renderToday() {
    const studiedToday = Boolean(auth.progress?.studied_today);
    const learnedToday = auth.progress?.learned_today ?? 0;

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
          <button className="primary-button hero-button" type="button" onClick={() => openLearn("practice")}>
            ▶️ Продолжить
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
              <button className="secondary-button" type="button" onClick={() => setPrimaryTab("progress")}>
                В прогресс →
              </button>
            ) : null}
          </div>
          <div className="achievement-grid">
            {todayAchievements.length ? (
              todayAchievements.map((item) => <span key={item} className="achievement-pill">{item}</span>)
            ) : (
              <div className="empty-state">Пока без наград.</div>
            )}
          </div>
        </section>
      </div>
    );
  }

  function renderCards() {
    return (
      <section className="glass-card learn-card">
        <div className="section-head">
          <div>
            <p className="overline">Cards</p>
            <h3>Карточки 🧠</h3>
          </div>
          <div className="button-row card-nav-row">
            <button className="secondary-button nav-arrow" type="button" onClick={showPreviousCard} disabled={cardIndex === 0} aria-label="Предыдущая карточка">
              ←
            </button>
            <button
              className="secondary-button nav-arrow"
              type="button"
              onClick={showNextCard}
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
                  <img src={`/api/image/${currentCard.id}`} alt={currentCard.word} loading="eager" />
                </div>
              ) : null}
              <div className="study-meta">
                <span>{cardIndex + 1} / {cardQueue.length}</span>
                <span>{currentCard.part_of_speech || "word"}</span>
              </div>
              <h2 className="study-word">{currentCard.word}</h2>
              <p className="transcription">/{currentCard.transcription || "no IPA"}/</p>
              <p className="example">{currentCard.example}</p>
              <audio controls src={`/api/audio/${currentCard.id}`} className="audio-player" />
            </div>
            <div className="study-side">
              {cardReveal ? (
                <>
                  <strong>{currentCard.translation}</strong>
                  <span>{currentCard.example_translation}</span>
                  <span className="study-hint">Посмотри перевод и переходи к следующей карточке.</span>
                  <button className="primary-button" type="button" onClick={() => showNextCard()}>
                    Дальше
                  </button>
                </>
              ) : (
                <button className="primary-button" type="button" onClick={() => setCardReveal(true)}>
                  Показать перевод
                </button>
              )}
            </div>
          </div>
        ) : <div className="empty-state">No cards for now.</div>}
      </section>
    );
  }

  function renderPractice(question, result, mode, resultSetter, reload, selectedAnswer) {
    return (
      <section className="glass-card learn-card">
        <div className="section-head section-head-wrap">
          <div>
            <p className="overline">{mode === "review" ? "Review" : "Practice"}</p>
            <h3>{mode === "review" ? "Повтор старых слов 🔁" : "Тест 🎯"}</h3>
          </div>
          {mode === "review" ? (
            <button className="secondary-button" type="button" onClick={reload}>
              Обновить
            </button>
          ) : null}
        </div>
        {question ? (
          <div className="quiz-panel">
            <div className="prompt-card">
              <strong>{question.mode === "reverse" ? question.item.translation : question.item.word}</strong>
              <span>{question.prompt}</span>
            </div>
            <div className="option-grid">
              {question.options.map((option) => (
                <button
                  key={option}
                  className={`option-button ${getOptionState(option, result, selectedAnswer)}`.trim()}
                  type="button"
                  disabled={Boolean(result)}
                  onClick={() => handlePracticeAnswer(option, mode, question, resultSetter)}
                >
                  {option}
                </button>
              ))}
            </div>
            {mode !== "review" ? (
              <button className="secondary-button" type="button" onClick={() => revealPracticeAnswer(question, mode, resultSetter)} disabled={Boolean(result)}>
                Пропустить
              </button>
            ) : null}
            {result ? (
              <div className={result.correct ? "result-box practice-result-box good" : "result-box practice-result-box bad"}>
                  <span>
                    {result.skipped
                      ? `Правильный ответ: ${result.correct_answer}`
                      : result.correct
                        ? "Верно"
                        : `Правильный ответ: ${result.correct_answer}`}
                  </span>
                <button className="secondary-button" type="button" onClick={reload}>
                  Дальше
                </button>
              </div>
            ) : null}
          </div>
        ) : <div className="empty-state">No items for this mode.</div>}
      </section>
    );
  }

  function renderLearn() {
    const statusClass = learnResult?.status === "correct"
      ? "result-box good"
      : learnResult?.status === "close"
        ? "result-box"
        : "result-box bad";

    if (!learnQuestion) {
      return (
        <section className="glass-card learn-card">
          <div className="section-head section-head-wrap">
            <div>
              <p className="overline">Practice</p>
              <h3>Смешанная практика 🎯</h3>
            </div>
          </div>
          <div className="empty-state">
            {learnSessionDone
              ? `Прогон завершён. Выполнено ${learnQuestionCount} из ${learnSessionLimit} заданий.`
              : "Сейчас нет подходящих заданий."}
          </div>
          <button className="primary-button" type="button" onClick={() => void loadLearningData()}>
            Начать новый прогон
          </button>
        </section>
      );
    }

    const isChoice = learnQuestion.kind === "choice";
    const isListening = learnQuestion.kind === "listening";
    const isSpeaking = learnQuestion.kind === "speaking";
    const promptTitle = isChoice
      ? (learnQuestion.exercise_type === "practice_ru_en" ? learnQuestion.item.translation : learnQuestion.item.word)
      : isListening
        ? "Слушай внимательно 🎧"
        : learnQuestion.item.word;

    return (
      <div className="screen-stack">
        <section className="glass-card learn-card">
          <div className="section-head section-head-wrap">
            <div>
              <p className="overline">Practice</p>
              <h3>Смешанная практика 🎯</h3>
            </div>
            <span className="status-tag">{learnQuestionCount + 1} / {learnSessionLimit}</span>
          </div>
          <div className="prompt-card">
            <strong>{promptTitle}</strong>
            <span>{learnQuestion.prompt}</span>
            <span className="study-hint">{learnQuestion.exercise_label}</span>
          </div>
          {isChoice ? (
            <div className="quiz-panel">
              <div className="option-grid">
                {learnQuestion.options.map((option) => (
                  <button
                    key={option}
                    className={`option-button ${getOptionState(option, learnResult, learnSelection)}`.trim()}
                    type="button"
                    disabled={Boolean(learnResult)}
                    onClick={() => void handleLearnChoiceAnswer(option)}
                  >
                    {option}
                  </button>
                ))}
              </div>
              <button className="secondary-button" type="button" onClick={revealLearnAnswer} disabled={Boolean(learnResult)}>
                Пропустить
              </button>
            </div>
          ) : null}
          {isListening ? (
            <form className="stack-form" onSubmit={handleLearnListeningSubmit}>
              <audio controls src={`/api/audio/${learnQuestion.item.id}`} className="audio-player" />
              <input value={learnTextAnswer} onChange={(event) => setLearnTextAnswer(event.target.value)} placeholder="Твой ответ" />
              <div className="button-row">
                <button className="primary-button" type="submit">Проверить</button>
                <button className="secondary-button" type="button" onClick={revealLearnAnswer} disabled={Boolean(learnResult)}>
                  Пропустить
                </button>
              </div>
            </form>
          ) : null}
          {isSpeaking ? (
            <div className="quiz-panel">
              <audio controls src={`/api/audio/${learnQuestion.item.id}`} className="audio-player" />
              <div className="button-row">
                <button
                  className={isRecording ? "secondary-button" : "primary-button"}
                  type="button"
                  onClick={isRecording ? stopSpeakingRecording : startSpeakingRecording}
                  disabled={busy || !canRecordSpeech}
                >
                  {isRecording ? "⏹️ Остановить запись" : "🎙️ Начать запись"}
                </button>
                <button className="secondary-button" type="button" onClick={revealLearnAnswer} disabled={Boolean(learnResult)}>
                  Пропустить
                </button>
              </div>
              {!learnResult ? (
                <div className="empty-state">
                  {!canRecordSpeech
                    ? "В этом браузере запись голоса недоступна."
                    : isRecording
                      ? "Идёт запись. Нажми «Остановить запись» после произношения."
                      : "Нажми на запись и произнеси слово."}
                </div>
              ) : null}
            </div>
          ) : null}
          {learnResult ? (
            <div className={isSpeaking ? statusClass : learnResult.correct ? "result-box good" : "result-box bad"}>
              <span>
                {isSpeaking
                  ? `${learnResult.message} Транскрибация: ${learnResult.transcript || "—"}.`
                  : learnResult.skipped
                    ? `Правильный ответ: ${learnResult.correct_answer}`
                    : learnResult.correct
                      ? "Верно"
                      : `Правильный ответ: ${learnResult.correct_answer}`}
              </span>
              <button className="secondary-button" type="button" onClick={() => void advanceLearnSession()}>
                Дальше
              </button>
            </div>
          ) : null}
        </section>
      </div>
    );
  }

  function renderLibrary() {
    return (
      <section className="screen-stack">
        {showLibraryAdd ? (
          renderAddWords()
        ) : (
          <>
            <div className="segment-wrap main-segment">
              {LIBRARY_MODES.map((item) => (
                <button
                  key={item.id}
                  className={libraryMode === item.id ? "segment-button active" : "segment-button"}
                  type="button"
                  onClick={() => setLibraryMode(item.id)}
                >
                  {item.label}
                </button>
              ))}
            </div>
            {libraryMode === "cards" ? renderCards() : null}
            {libraryMode === "words" ? renderWordsList() : null}
          </>
        )}
      </section>
    );
  }

  function renderWordsList() {
    return (
      <>
        <div className="glass-card compact-section">
          <div className="filters">
            <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search" />
            <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
              <option value="all">Все</option>
              <option value="learning">В процессе</option>
              <option value="learned">Выучено</option>
            </select>
          </div>
        </div>
        <div className="word-list">
          {words.map((item) => (
            <article className="glass-card word-item" key={item.id}>
              <div className="word-item-head">
                <div>
                  <strong>{item.word}</strong>
                  <p className="word-item-example">{item.example}</p>
                </div>
                <span className={item.is_learned ? "status-tag good" : "status-tag"}>
                  {item.is_learned ? "Выучено" : `${item.correct_count}/${settings?.exercise_goal || 4}`}
                </span>
              </div>
              <input
                value={draftTranslation[item.id] ?? item.translation}
                onChange={(event) => setDraftTranslation((current) => ({ ...current, [item.id]: event.target.value }))}
              />
              <div className="button-row word-item-actions word-item-actions-primary">
                <button className="secondary-button" type="button" onClick={() => saveTranslation(item.id)}>
                  Сохранить
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => {
                    setExpandedWordId((current) => current === item.id ? null : item.id);
                    setPreviewWordId((current) => (current === item.id ? null : current));
                  }}
                >
                  {expandedWordId === item.id ? "Скрыть" : "Ещё"}
                </button>
              </div>
              {expandedWordId === item.id ? (
                <div className="word-item-extra">
                  <div className="button-row word-item-actions">
                    <button className="secondary-button" type="button" onClick={() => setPreviewWordId((current) => current === item.id ? null : item.id)}>
                      {previewWordId === item.id ? "🫥 Скрыть фото" : "🖼 Фото"}
                    </button>
                    <button
                      className={regeneratingWordId === item.id ? "secondary-button is-loading" : "secondary-button"}
                      type="button"
                      onClick={() => regenerateWordImage(item.id)}
                      disabled={regeneratingWordId === item.id || (item.image_regeneration_count || 0) >= MAX_IMAGE_REGENERATIONS}
                    >
                      {regeneratingWordId === item.id
                        ? "⏳ Обновляем..."
                        : (item.image_regeneration_count || 0) >= MAX_IMAGE_REGENERATIONS
                          ? "Лимит фото"
                          : "♻️ Обновить фото"}
                    </button>
                    <button className="danger-button" type="button" onClick={() => deleteWord(item.id)}>
                      Удалить
                    </button>
                  </div>
                      {previewWordId === item.id ? (
                    item.has_image ? (
                      <div className="word-image-preview">
                        <img
                          key={wordImageVersions[item.id] || item.updated_at}
                          src={`/api/image/${item.id}?v=${wordImageVersions[item.id] || item.updated_at}`}
                          alt={item.word}
                        />
                      </div>
                    ) : (
                      <div className="empty-card">У этого слова пока нет изображения.</div>
                    )
                  ) : null}
                </div>
              ) : null}
            </article>
          ))}
          {!words.length ? <div className="glass-card empty-card">No words found.</div> : null}
        </div>
      </>
    );
  }

  function renderProgress() {
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
            <h3>{auth.progress?.rank_percent ? `${auth.progress.rank_percent}%` : "—"}</h3>
            <span>🏅 Top range</span>
          </div>
        </section>
        <section className="glass-card compact-section">
          <p className="overline">Achievements</p>
          <div className="achievement-grid">
            {(auth.progress?.achievements || []).length ? (
              auth.progress.achievements.map((item) => <span key={item} className="achievement-pill">{item}</span>)
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
            {(auth.progress?.pending_achievement_highlights || []).length ? (
              auth.progress.pending_achievement_highlights.map((item) => (
                <div key={item.text} className="simple-row">
                  <strong>{item.text}</strong>
                  <span>{item.current} / {item.target}</span>
                </div>
              ))
            ) : (
              <div className="empty-state">Все текущие ачивки уже получены.</div>
            )}
          </div>
        </section>
      </div>
    );
  }

  function renderAddWords() {
    const isTranslationStep = addDraftStep === "confirm_translation";
    const isImageStep = addDraftStep === "confirm_image";
    const isBatchReview = addDraftStep === "batch_review";

    return (
      <section className="glass-card compact-section add-wizard">
        <div className="section-head">
          <div>
            <p className="overline">Add</p>
            <h3>{isBatchReview ? "Проверить слова ✨" : "Добавить слово ✨"}</h3>
            <p className="lead compact">
              {isBatchReview
                ? "Проверь переводы. Фото загружаются автоматически и не тормозят добавление."
                : "Подтверждаем перевод, а фото загружается автоматически в фоне."}
            </p>
          </div>
          <button className="secondary-button" type="button" onClick={closeAddWords} disabled={addBusy}>
            Закрыть
          </button>
        </div>
        <div className="wizard-steps">
          <span className={addDraftStep === "input" ? "mode-pill active-pill" : "mode-pill"}>1. Слово</span>
          <span className={isTranslationStep || isBatchReview ? "mode-pill active-pill" : "mode-pill"}>2. Перевод</span>
          <span className={isImageStep || isBatchReview ? "mode-pill active-pill" : "mode-pill"}>3. Картинка</span>
        </div>

        {addBusyLabel ? <div className="inline-note status-note"><strong>{addBusyLabel}</strong></div> : null}

        {!addDraft && !isBatchReview ? (
          <form className="stack-form" onSubmit={handleAddWords}>
            <textarea
              rows={4}
              value={addText}
              onChange={(event) => setAddText(event.target.value)}
              placeholder={"stare\nfigure out\ntravel - путешествие"}
            />
            <button className="primary-button" type="submit" disabled={addBusy}>
              {addBusy ? "Обрабатываем..." : "Добавить слово"}
            </button>
          </form>
        ) : null}

        {isBatchReview && addDrafts.length ? (
          <div className="word-list batch-draft-list">
            {addDrafts.map((draft) => (
              <article className="glass-card word-item" key={draft.id}>
                <div className="word-item-head">
                  <div>
                    <strong>{draft.word}</strong>
                    <p>{draft.example}</p>
                  </div>
                  <span className="status-tag">{draft.part_of_speech || "word"}</span>
                </div>
                <input
                  value={batchTranslations[draft.id] ?? draft.translation}
                  onChange={(event) => setBatchTranslations((current) => ({ ...current, [draft.id]: event.target.value }))}
                />
                {draft.has_image ? (
                  <div className="word-image-preview">
                    <img
                      key={draftImageVersions[draft.id] || draft.updated_at}
                      src={`/api/draft-image/${draft.id}?v=${draftImageVersions[draft.id] || draft.updated_at}`}
                      alt={draft.word}
                    />
                  </div>
                ) : (
                  <div className="empty-card">
                    {draft.image_generation_in_progress ? "Фото готовится автоматически..." : "Фото появится автоматически позже."}
                  </div>
                )}
                <div className="button-row">
                  {draft.has_image ? (
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() => regenerateBatchDraftImage(draft.id)}
                      disabled={addBusy || (draft.image_regeneration_count || 0) >= MAX_IMAGE_REGENERATIONS}
                    >
                      {(draft.image_regeneration_count || 0) >= MAX_IMAGE_REGENERATIONS ? "Лимит фото" : "♻️ Другое фото"}
                    </button>
                  ) : null}
                </div>
              </article>
            ))}
            <div className="button-row batch-save-row">
              <button className="primary-button" type="button" onClick={saveBatchDrafts} disabled={addBusy}>
                {addBusy ? "Сохраняем..." : "Сохранить всё"}
              </button>
            </div>
          </div>
        ) : null}

        {addDraft && isTranslationStep ? (
          <div className="draft-card">
            <div className="prompt-card">
              <strong>{addDraft.word}</strong>
              <span>{addDraft.part_of_speech || "word"}</span>
            </div>
            <div className="stack-form">
              <label className="stack-label">
                <span>Подтверди перевод</span>
                <input value={addTranslationInput} onChange={(event) => setAddTranslationInput(event.target.value)} placeholder="Перевод" />
              </label>
              <div className="button-row">
                <button className="primary-button" type="button" onClick={confirmDraftTranslation} disabled={addBusy}>
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
                {addDraft.has_image ? (
                  <div className="card-visual">
                    <img
                      key={draftImageVersions[addDraft.id] || addDraft.updated_at}
                      src={`/api/draft-image/${addDraft.id}?v=${draftImageVersions[addDraft.id] || addDraft.updated_at}`}
                      alt={addDraft.word}
                    />
                  </div>
                ) : (
                  <div className="empty-card">
                    {addDraft.image_generation_in_progress
                      ? "Фото готовится автоматически. Можно сохранить слово, оно появится позже."
                      : "Фото появится автоматически позже."}
                  </div>
                )}
              </div>
              <div className="study-side">
                <strong>{addDraft.word}</strong>
                <span>{addTranslationInput || addDraft.translation}</span>
                <span>{addDraft.part_of_speech || "word"}</span>
                {addDraft.example ? <span>{addDraft.example}</span> : null}
                <div className="button-row draft-action-row">
                  <button className="primary-button" type="button" onClick={() => saveDraft(true)} disabled={addBusy}>
                    {addBusy ? "Сохраняем..." : "Сохранить"}
                  </button>
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={regenerateDraftImage}
                    disabled={addBusy || (addDraft.image_regeneration_count || 0) >= MAX_IMAGE_REGENERATIONS}
                  >
                    {(addDraft.image_regeneration_count || 0) >= MAX_IMAGE_REGENERATIONS ? "Лимит фото" : "Другое изображение"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : null}
      </section>
    );
  }

  function renderIrregular() {
    return (
      <div className="screen-stack">
        <div className="segment-wrap main-segment">
          {IRREGULAR_MODES.map((item) => (
            <button
              key={item.id}
              className={irregularMode === item.id ? "segment-button active" : "segment-button"}
              type="button"
              onClick={() => setIrregularMode(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>
        {irregularMode === "review" ? (
          <section className="glass-card compact-section">
            <div className="section-head">
              <div>
                <p className="overline">Irregular</p>
                <h3>Повторять глаголы 📘</h3>
              </div>
              <div className="button-row">
                <button className="secondary-button" type="button" onClick={() => setIrregularPage((value) => Math.max(0, value - 1))} disabled={!irregularList?.has_prev}>
                  Prev
                </button>
                <button className="secondary-button" type="button" onClick={() => setIrregularPage((value) => value + 1)} disabled={!irregularList?.has_next}>
                  Next
                </button>
              </div>
            </div>
            <div className="simple-list">
              {(irregularList?.items || []).map((item) => (
                <div key={item.base} className="simple-row four-cols">
                  <strong>{item.base}</strong>
                  <span>{item.past}</span>
                  <span>{item.participle}</span>
                  <span>{item.translation}</span>
                </div>
              ))}
            </div>
          </section>
        ) : null}
        {irregularMode === "test" ? (
          <section className="glass-card compact-section">
            <div className="section-head">
              <div>
                <p className="overline">Train</p>
                <h3>Тест по глаголам 🧩</h3>
              </div>
              <button className="secondary-button" type="button" onClick={loadIrregularQuestion}>
                Refresh
              </button>
            </div>
            {irregularQuestion ? (
              <div className="quiz-panel">
                <div className="prompt-card">
                  <strong>{irregularQuestion.verb.base}</strong>
                  <span>Pick the correct pair</span>
                </div>
                <div className="option-grid">
                  {irregularQuestion.options.map((option) => (
                    <button key={option} className="option-button" type="button" onClick={() => handleIrregularAnswer(option)}>
                      {option}
                    </button>
                  ))}
                </div>
                {irregularResult ? (
                  <div className={irregularResult.correct ? "result-box good" : "result-box bad"}>
                    <span>{irregularResult.correct ? "Correct" : `Correct answer: ${irregularResult.correct_answer}`}</span>
                    <button className="secondary-button" type="button" onClick={loadIrregularQuestion}>
                      Next
                    </button>
                  </div>
                ) : null}
              </div>
            ) : <div className="empty-state">No verb question right now.</div>}
          </section>
        ) : null}
      </div>
    );
  }

  function renderSettings() {
    if (!settings) {
      return null;
    }
    return (
      <section className="glass-card compact-section">
        <p className="overline">Settings</p>
        <h3>Настройки ⚙️</h3>
        <form className="settings-grid" onSubmit={saveSettings}>
          <label>
            <span>Exercises to learn</span>
            <small>Сколько разных упражнений нужно выполнить, чтобы слово стало выученным.</small>
            <input type="number" min="2" max="5" value={settings.exercise_goal} onChange={(event) => setSettings((current) => ({ ...current, exercise_goal: Number(event.target.value) }))} />
          </label>
          <label>
            <span>Questions per run</span>
            <small>Максимум заданий за один прогон практики без повторов слов.</small>
            <input type="number" min="1" max="50" value={settings.session_question_limit} onChange={(event) => setSettings((current) => ({ ...current, session_question_limit: Number(event.target.value) }))} />
          </label>
          <label>
            <span>Days before review</span>
            <input type="number" min="1" max="365" value={settings.days_before_review} onChange={(event) => setSettings((current) => ({ ...current, days_before_review: Number(event.target.value) }))} />
          </label>
          <label>
            <span>Reminder interval</span>
            <input type="number" min="1" max="30" value={settings.reminder_interval_days} onChange={(event) => setSettings((current) => ({ ...current, reminder_interval_days: Number(event.target.value) }))} />
          </label>
          <label>
            <span>Reminder time</span>
            <input type="time" value={settings.reminder_time} onChange={(event) => setSettings((current) => ({ ...current, reminder_time: event.target.value }))} />
          </label>
          <label>
            <span>Time zone</span>
            <input value={settings.reminder_timezone} onChange={(event) => setSettings((current) => ({ ...current, reminder_timezone: event.target.value }))} />
          </label>
          <label className="toggle-row">
            <input type="checkbox" checked={settings.enable_review_old_words} onChange={(event) => setSettings((current) => ({ ...current, enable_review_old_words: event.target.checked }))} />
            <span>Review old words</span>
          </label>
          <label className="toggle-row">
            <input type="checkbox" checked={settings.reminder_enabled} onChange={(event) => setSettings((current) => ({ ...current, reminder_enabled: event.target.checked }))} />
            <span>Enable reminders</span>
          </label>
          <button className="primary-button" type="submit">Сохранить настройки</button>
        </form>
      </section>
    );
  }

  function renderMore() {
    return (
      <div className="screen-stack">
        <div className="segment-wrap main-segment more-segment">
          {MORE_PANELS.map((item) => (
            <button
              key={item.id}
              className={morePanel === item.id ? "segment-button active" : "segment-button"}
              type="button"
              onClick={() => setMorePanel(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>
        {morePanel === "review" ? renderPractice(reviewQuestion, reviewResult, "review", setReviewResult, loadReview, reviewSelection) : null}
        {morePanel === "irregular" ? renderIrregular() : null}
        {morePanel === "settings" ? renderSettings() : null}
      </div>
    );
  }

  function renderScreen() {
    if (primaryTab === "today") return renderToday();
    if (primaryTab === "learn") return renderLearn();
    if (primaryTab === "words") return renderLibrary();
    if (primaryTab === "progress") return renderProgress();
    return renderMore();
  }

  if (auth.loading) {
    return <div className="boot-screen">Loading VocabuMe...</div>;
  }

  if (!auth.authenticated) {
    return (
      <div className="app-shell auth-layout">
        {notice ? <div className="notice">{notice.message}</div> : null}
        <AuthPanel
          config={config}
          onOpenLogin={requestLoginLink}
          loginLink={loginLink}
          loginPending={busy}
        />
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="glass-card topbar">
        <div className="topbar-brand">
          <LogoMark />
          <div>
            <p className="overline">VocabuMe</p>
            <strong className="app-title">{currentTitle}</strong>
          </div>
        </div>
        {primaryTab === "words" ? (
          <button
            className={showLibraryAdd ? "secondary-button header-action active" : "secondary-button header-action"}
            type="button"
            onClick={() => {
              if (showLibraryAdd) {
                closeAddWords();
                return;
              }
              resetAddFlow();
              setShowLibraryAdd(true);
            }}
            aria-label="Добавить слова"
          >
            <span className="header-action-mark">＋</span>
            <span>Добавить</span>
          </button>
        ) : (
          <span className="mode-pill">{isMiniApp ? "Telegram" : "Web"}</span>
        )}
      </header>

      {notice ? <div className="notice">{notice.message}</div> : null}

      <main className="app-stage" ref={stageRef}>
        {renderScreen()}
      </main>

      <nav className="nav-grid-bottom">
        {PRIMARY_TABS.map((item) => (
          <button
            key={item.id}
            className={item.id === primaryTab ? "nav-pill active" : "nav-pill"}
            type="button"
            onClick={() => startTransition(() => {
              if (item.id !== "words" && showLibraryAdd) {
                void closeAddWords();
              }
              setPrimaryTab(item.id);
              if (item.id !== "words") {
                setShowLibraryAdd(false);
              }
            })}
          >
            <span className="nav-label">{item.label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}

export default App;
