import {
  startTransition,
  useDeferredValue,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import AuthPanel from "./components/AuthPanel";
import AppTopbar from "./components/AppTopbar";
import BottomNav from "./components/BottomNav";
import {
  ALPHABET_MODES,
  IRREGULAR_MODES,
  LIBRARY_MODES,
  MAX_ADD_BATCH_WORDS,
  MAX_IMAGE_REGENERATIONS,
} from "./constants";
import { api, reportClientError } from "./lib/api";
import ProgressScreen from "./screens/ProgressScreen";
import SettingsScreen from "./screens/SettingsScreen";
import TodayScreen from "./screens/TodayScreen";

function getSessionPraise(correct, total) {
  if (!total) {
    return "Сессию можно пройти в своём темпе.";
  }
  const ratio = correct / total;
  if (ratio >= 0.9) {
    return "🔥 Отличный результат. Почти всё выполнено без ошибок.";
  }
  if (ratio >= 0.7) {
    return "👏 Очень хорошо. Большая часть заданий выполнена верно.";
  }
  if (ratio >= 0.45) {
    return `💪 Неплохо. Основа уже есть, можно пройти ещё один круг.`;
  }
  return `🌱 Начало положено. Следующая сессия уже будет увереннее.`;
}

function formatLearnCorrectAnswer(learnQuestion, learnResult) {
  const answer = learnResult?.correct_answer || "";
  if (!learnQuestion || !learnResult) {
    return answer;
  }
  if (learnQuestion.exercise_type === "listening_translate" && learnQuestion.item?.word) {
    return `${answer} (${learnQuestion.item.word})`;
  }
  return answer;
}

function mergeItemsById(current, incoming) {
  if (!incoming?.length) {
    return current;
  }
  const seen = new Set(incoming.map((item) => item.id));
  return [...incoming, ...current.filter((item) => !seen.has(item.id))];
}

const GEORGIAN_TO_LATIN = {
  "ა": "a",
  "ბ": "b",
  "გ": "g",
  "დ": "d",
  "ე": "e",
  "ვ": "v",
  "ზ": "z",
  "თ": "t",
  "ი": "i",
  "კ": "k'",
  "ლ": "l",
  "მ": "m",
  "ნ": "n",
  "ო": "o",
  "პ": "p'",
  "ჟ": "zh",
  "რ": "r",
  "ს": "s",
  "ტ": "t'",
  "უ": "u",
  "ფ": "p",
  "ქ": "k",
  "ღ": "gh",
  "ყ": "q'",
  "შ": "sh",
  "ჩ": "ch",
  "ც": "ts",
  "ძ": "dz",
  "წ": "ts'",
  "ჭ": "ch'",
  "ხ": "kh",
  "ჯ": "j",
  "ჰ": "h",
};

function transliterateGeorgian(text) {
  return Array.from(text || "")
    .map((char) => GEORGIAN_TO_LATIN[char] || char)
    .join("");
}

function hasGeorgianScript(text) {
  return /[\u10A0-\u10FF]/.test(text || "");
}

function App() {
  const [config, setConfig] = useState({ bot_username: "", webapp_url: "" });
  const [auth, setAuth] = useState({ loading: true, authenticated: false, user: null, progress: null });
  const [notice, setNoticeState] = useState(null);
  const [busy, setBusy] = useState(false);
  const [primaryTab, setPrimaryTab] = useState("today");
  const [libraryMode, setLibraryMode] = useState("cards");
  const [learnPanel, setLearnPanel] = useState("mixed");
  const [showLibraryAdd, setShowLibraryAdd] = useState(false);
  const [dashboard, setDashboard] = useState(null);
  const [settings, setSettings] = useState(null);
  const [words, setWords] = useState([]);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [draftTranslation, setDraftTranslation] = useState({});
  const [previewWordId, setPreviewWordId] = useState(null);
  const [expandedWordId, setExpandedWordId] = useState(null);
  const [wordImageVersions] = useState({});
  const [wordImageErrors, setWordImageErrors] = useState({});
  const [regeneratingWordId, setRegeneratingWordId] = useState(null);
  const [addText, setAddText] = useState("");
  const [packs, setPacks] = useState([]);
  const [selectedPackId, setSelectedPackId] = useState("travel");
  const [selectedPackLevelId, setSelectedPackLevelId] = useState("a1_a2");
  const [selectedPackWords, setSelectedPackWords] = useState({});
  const [isPackExpanded, setIsPackExpanded] = useState(false);
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
  const [learnCorrectCount, setLearnCorrectCount] = useState(0);
  const [learnSessionLimit, setLearnSessionLimit] = useState(12);
  const [learnSessionDone, setLearnSessionDone] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [irregularPage, setIrregularPage] = useState(0);
  const [irregularList, setIrregularList] = useState(null);
  const [irregularQuestion, setIrregularQuestion] = useState(null);
  const [irregularResult, setIrregularResult] = useState(null);
  const [irregularMode, setIrregularMode] = useState("review");
  const [irregularQuestionCount, setIrregularQuestionCount] = useState(0);
  const [irregularCorrectCount, setIrregularCorrectCount] = useState(0);
  const [irregularSessionLimit, setIrregularSessionLimit] = useState(12);
  const [irregularSessionDone, setIrregularSessionDone] = useState(false);
  const [alphabetPage, setAlphabetPage] = useState(0);
  const [alphabetList, setAlphabetList] = useState(null);
  const [alphabetQuestion, setAlphabetQuestion] = useState(null);
  const [alphabetResult, setAlphabetResult] = useState(null);
  const [alphabetMode, setAlphabetMode] = useState("review");
  const [alphabetQuestionCount, setAlphabetQuestionCount] = useState(0);
  const [alphabetCorrectCount, setAlphabetCorrectCount] = useState(0);
  const [alphabetSessionLimit, setAlphabetSessionLimit] = useState(12);
  const [alphabetSessionDone, setAlphabetSessionDone] = useState(false);
  const [alphabetAudioLoadingSymbol, setAlphabetAudioLoadingSymbol] = useState("");
  const [georgianDisplayModePrompt, setGeorgianDisplayModePrompt] = useState(null);
  const [loginLink, setLoginLink] = useState("");
  const [loginToken, setLoginToken] = useState("");
  const [isKeyboardOpen, setIsKeyboardOpen] = useState(false);
  const [webAuthMode, setWebAuthMode] = useState("login");
  const [webEmail, setWebEmail] = useState("");
  const [webPassword, setWebPassword] = useState("");
  const pollRef = useRef(null);
  const stageRef = useRef(null);
  const alphabetAudioRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const speakingChunksRef = useRef([]);
  const noticeTimerRef = useRef(null);
  const deferredSearch = useDeferredValue(search);

  const webApp = window.Telegram?.WebApp;
  const needsStudiedLanguageSelection =
    auth.authenticated && auth.user && !auth.user.has_selected_studied_language;
  const isMiniApp = Boolean(webApp?.initData);
  const canRecordSpeech = Boolean(navigator.mediaDevices?.getUserMedia && window.MediaRecorder);
  const activeStudiedLanguage = settings?.active_studied_language || auth.progress?.course_code || "en";
  const supportsIrregularPractice = activeStudiedLanguage === "en";
  const georgianDisplayMode =
    settings?.georgian_display_mode || auth.user?.georgian_display_mode || "both";
  const showGeorgianLatin = activeStudiedLanguage === "ka" && georgianDisplayMode === "both";
  const georgianDisplayModeOptions =
    settings?.georgian_display_mode_options
    || auth.user?.georgian_display_mode_options
    || [
      { code: "both", label: "Грузинский + латиница", recommended: true },
      { code: "native", label: "Только грузинский", recommended: false },
    ];

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
    if (alphabetAudioRef.current) {
      alphabetAudioRef.current.pause();
      alphabetAudioRef.current = null;
    }
    if (noticeTimerRef.current) {
      window.clearTimeout(noticeTimerRef.current);
    }
  }, []);

  useEffect(() => {
    const html = document.documentElement;
    const body = document.body;
    const root = document.getElementById("root");
    const previousHtmlOverflow = html.style.overflow;
    const previousBodyOverflow = body.style.overflow;
    const previousHtmlHeight = html.style.height;
    const previousBodyHeight = body.style.height;
    const previousRootOverflow = root?.style.overflow ?? "";
    const previousRootHeight = root?.style.height ?? "";

    if (auth.authenticated) {
      html.style.overflow = "hidden";
      body.style.overflow = "hidden";
      html.style.height = "100%";
      body.style.height = "100%";
      if (root) {
        root.style.overflow = "hidden";
        root.style.height = "100%";
      }
    } else {
      html.style.overflow = "auto";
      body.style.overflow = "auto";
      html.style.height = "auto";
      body.style.height = "auto";
      if (root) {
        root.style.overflow = "visible";
        root.style.height = "auto";
      }
    }

    return () => {
      html.style.overflow = previousHtmlOverflow;
      body.style.overflow = previousBodyOverflow;
      html.style.height = previousHtmlHeight;
      body.style.height = previousBodyHeight;
      if (root) {
        root.style.overflow = previousRootOverflow;
        root.style.height = previousRootHeight;
      }
    };
  }, [auth.authenticated]);

  useEffect(() => {
    const viewport = window.visualViewport;
    if (!viewport) {
      return;
    }

    const updateKeyboardState = () => {
      const heightDiff = window.innerHeight - viewport.height;
      setIsKeyboardOpen(heightDiff > 140);
    };

    const handleFocusIn = (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement) || !target.matches("input, textarea")) {
        return;
      }
      window.setTimeout(() => {
        target.scrollIntoView({ block: "center", behavior: "smooth" });
      }, 120);
    };

    const handleFocusOut = () => {
      window.setTimeout(updateKeyboardState, 120);
    };

    updateKeyboardState();
    viewport.addEventListener("resize", updateKeyboardState);
    viewport.addEventListener("scroll", updateKeyboardState);
    document.addEventListener("focusin", handleFocusIn);
    document.addEventListener("focusout", handleFocusOut);

    return () => {
      viewport.removeEventListener("resize", updateKeyboardState);
      viewport.removeEventListener("scroll", updateKeyboardState);
      document.removeEventListener("focusin", handleFocusIn);
      document.removeEventListener("focusout", handleFocusOut);
    };
  }, []);

  const stats = useMemo(() => {
    const progress = dashboard?.progress || auth.progress;
    return [
      { label: "📚 Словарь", value: progress?.total ?? 0 },
      { label: "✅ Выучено", value: progress?.learned ?? 0 },
      { label: "🔄 В процессе", value: progress?.learning ?? 0 },
      { label: "🔥 Серия", value: progress?.streak_days ?? 0 }
    ];
  }, [auth.progress, dashboard]);

  const todayStats = useMemo(() => stats.slice(0, 3), [stats]);
  const todayAchievements = useMemo(() => {
    const list = dashboard?.progress?.achievements || auth.progress?.achievements || [];
    return list.slice(-3);
  }, [auth.progress, dashboard]);
  const hasMoreAchievements =
    (dashboard?.progress?.achievements || auth.progress?.achievements || []).length >
    todayAchievements.length;

  const currentTitle = useMemo(() => {
    if (primaryTab === "today") return "Сегодня";
    if (primaryTab === "learn") return "Практика";
    if (primaryTab === "words") return showLibraryAdd ? "Добавить" : libraryMode === "cards" ? "Карточки" : "Список";
    if (primaryTab === "progress") return "Прогресс";
    return primaryTab === "more" ? "Настройки" : "Практика";
  }, [primaryTab, libraryMode, showLibraryAdd]);

  const currentCard = cardQueue[cardIndex];
  const noticeScope = useMemo(() => {
    if (!auth.authenticated) {
      return "auth";
    }
    return [
      primaryTab,
      libraryMode,
      learnPanel,
      showLibraryAdd ? "add" : "main",
      addDraftStep,
    ].join(":");
  }, [auth.authenticated, primaryTab, libraryMode, learnPanel, showLibraryAdd, addDraftStep]);
  const showHeaderBack =
    primaryTab === "learn"
    && (
      (learnPanel === "irregular" && irregularMode === "review")
      || (learnPanel === "alphabet" && alphabetMode === "review")
    );
  const showHeaderClose = (primaryTab === "learn" && Boolean(learnQuestion))
    || (primaryTab === "learn" && learnPanel === "irregular" && irregularMode === "test")
    || (primaryTab === "learn" && learnPanel === "alphabet" && alphabetMode === "test");

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

  function formatDisplayLine(text, courseCode = activeStudiedLanguage) {
    const primary = text || "";
    if (courseCode !== "ka" || !showGeorgianLatin || !primary) {
      return { primary, secondary: "" };
    }
    return {
      primary,
      secondary: transliterateGeorgian(primary),
    };
  }

  function formatDisplayAnswer(text, courseCode = activeStudiedLanguage) {
    const value = text || "";
    if (courseCode !== "ka" || !showGeorgianLatin || !hasGeorgianScript(value)) {
      return value;
    }
    return value.replace(
      /[\u10A0-\u10FF]+(?:\s+[\u10A0-\u10FF]+)*/g,
      (match) => `${match}/${transliterateGeorgian(match)}`,
    );
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

  useEffect(() => {
    let cancelled = false;
    const draftsToSync = [
      ...(addDraft ? [addDraft] : []),
      ...addDrafts,
    ].filter((draft, index, all) => draft?.has_image && all.findIndex((item) => item.id === draft.id) === index);

    async function syncDraftImages() {
      for (const draft of draftsToSync) {
        const token = draft.updated_at;
        if (!token || draftImageVersions[draft.id] === token) {
          continue;
        }
        const loaded = await preloadDraftImage(draft.id, token);
        if (cancelled || !loaded) {
          continue;
        }
        setDraftImageVersions((current) => {
          if (current[draft.id] === token) {
            return current;
          }
          return { ...current, [draft.id]: token };
        });
      }
    }

    if (draftsToSync.length) {
      void syncDraftImages();
    }

    return () => {
      cancelled = true;
    };
  }, [addDraft, addDrafts, draftImageVersions]);

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
    const [dashboardData, settingsData, wordsData, irregularData, alphabetData] = await Promise.all([
      api("/api/dashboard"),
      api("/api/settings"),
      api(`/api/words?status=${statusFilter}&search=${encodeURIComponent(deferredSearch)}`),
      api(`/api/irregular/list?page=${irregularPage}`),
      api(`/api/alphabet/list?page=${alphabetPage}`)
    ]);
    setDashboard(dashboardData);
    setAuth((current) => ({
      ...current,
      user: dashboardData.user || current.user,
      progress: dashboardData.progress || current.progress,
    }));
    setSettings(settingsData.settings);
    setWords(wordsData.items);
    setIrregularList(irregularData);
    setAlphabetList(alphabetData);
  }

  async function loadCards(options = {}) {
    const { reset = true } = options;
    const data = await api("/api/study/cards?scope=all");
    setCardQueue((current) => {
      if (reset) {
        return data.items;
      }
      return data.items;
    });
    if (reset) {
      setCardIndex(0);
      setCardReveal(false);
      return;
    }
    setCardIndex((currentIndex) => {
      const currentItemId = cardQueue[currentIndex]?.id;
      if (!currentItemId) {
        return 0;
      }
      const nextIndex = data.items.findIndex((item) => item.id === currentItemId);
      return nextIndex >= 0 ? nextIndex : 0;
    });
  }

  async function loadLearningData(options = {}) {
    const { resetCards = true, resetLearn = true } = options;
    if (resetLearn) {
      setLearnUsedWordIds([]);
      setLearnQuestionCount(0);
      setLearnCorrectCount(0);
      setLearnSessionDone(false);
    }
    await Promise.all([
      loadCards({ reset: resetCards }),
      resetLearn ? loadLearnQuestion([], 0) : Promise.resolve(),
    ]);
  }

  async function loadStudyCardsOnly(options = {}) {
    await loadCards(options);
  }

  function showPreviousCard() {
    setCardIndex((value) => Math.max(0, value - 1));
    setCardReveal(false);
  }

  function showNextCard() {
    setCardIndex((value) => Math.min(cardQueue.length - 1, value + 1));
    setCardReveal(false);
  }

  async function loadLearnQuestion(excludeIds = [], questionCount = 0) {
    const ids = excludeIds.filter(Boolean);
    const data = await api(`/api/learn/question?exclude_ids=${encodeURIComponent(ids.join(","))}`);
    setLearnSessionLimit(data.session_limit || 12);
    setLearnSelection("");
    setLearnTextAnswer("");
    setLearnResult(null);
    setIsRecording(false);
    if (questionCount >= (data.session_limit || 12) || (data.empty && questionCount > 0)) {
      setLearnQuestion(null);
      setLearnSessionDone(true);
      return;
    }
    if (data.empty) {
      setLearnQuestion(null);
      setLearnSessionDone(false);
      return;
    }
    setLearnQuestion(data.question);
    setLearnSessionDone(false);
  }

  async function loadIrregularQuestion() {
    const data = await api("/api/irregular/question");
    setIrregularQuestion(data.question);
    setIrregularResult(null);
    setIrregularSessionLimit(settings?.session_question_limit || 12);
  }

  async function loadAlphabetQuestion() {
    const data = await api("/api/alphabet/question");
    setAlphabetQuestion(data.question);
    setAlphabetResult(null);
    setAlphabetSessionLimit(settings?.session_question_limit || 12);
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
    Promise.all([loadDashboard(), loadStudyCardsOnly()])
      .catch((error) => setNotice(error.message));
  }, [auth.authenticated, deferredSearch, statusFilter, irregularPage, alphabetPage]);

  useEffect(() => {
    if (supportsIrregularPractice) {
      return;
    }
    if (learnPanel === "irregular") {
      setLearnPanel("mixed");
    }
    setIrregularMode("review");
    setIrregularQuestion(null);
    setIrregularResult(null);
    setIrregularQuestionCount(0);
    setIrregularCorrectCount(0);
    setIrregularSessionDone(false);
  }, [supportsIrregularPractice, learnPanel]);

  useEffect(() => {
    if (!auth.authenticated) {
      return;
    }
    const hasPendingWordImages = words.some((item) => item.image_generation_in_progress) || cardQueue.some((item) => item.image_generation_in_progress);
    if (!hasPendingWordImages) {
      return;
    }
    const intervalId = window.setInterval(() => {
      void Promise.all([
        loadDashboard(),
        loadCards({ reset: false }),
      ]);
    }, 4000);
    return () => window.clearInterval(intervalId);
  }, [auth.authenticated, words, cardQueue]);

  useEffect(() => {
    if (!auth.authenticated || !showLibraryAdd) {
      return;
    }
    void loadPacks();
    void preparePacksInBackground();
  }, [auth.authenticated, showLibraryAdd]);

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
  }, [primaryTab, libraryMode, learnPanel]);

  async function requestLoginLink() {
    setBusy(true);
    try {
      const data = await api("/api/auth/telegram/request-link", {
        method: "POST",
        body: JSON.stringify({})
      });
      setLoginLink(data.deep_link);
      setLoginToken(data.token);
      if (window.Telegram?.WebApp?.openTelegramLink) {
        window.Telegram.WebApp.openTelegramLink(data.deep_link);
      } else {
        window.open(data.deep_link, "_blank", "noopener,noreferrer");
      }
      setNotice("Открыл бота. Нажми Start, затем вернись сюда.");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function logoutWeb() {
    setBusy(true);
    try {
      await api("/api/auth/logout", {
        method: "POST",
        body: JSON.stringify({})
      });
      stopPolling();
      setLoginLink("");
      setLoginToken("");
      setDashboard(null);
      setSettings(null);
      setWords([]);
      setCardQueue([]);
      setLearnQuestion(null);
      setLearnResult(null);
      setIrregularQuestion(null);
      setAlphabetQuestion(null);
      setShowLibraryAdd(false);
      setAuth({ loading: false, authenticated: false, user: null, progress: null });
      setNotice("Выход выполнен.");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function submitWebAuth(event) {
    event.preventDefault();
    if (!webEmail.trim()) {
      setNotice("Укажи email.");
      return;
    }
    if (!webPassword) {
      setNotice("Укажи пароль.");
      return;
    }

    setBusy(true);
    try {
      const endpoint = webAuthMode === "register" ? "/api/auth/web/register" : "/api/auth/web/login";
      const data = await api(endpoint, {
        method: "POST",
        body: JSON.stringify({
          email: webEmail.trim(),
          password: webPassword,
        }),
      });
      stopPolling();
      setLoginLink("");
      setLoginToken("");
      setAuth({ loading: false, authenticated: true, user: data.user, progress: data.progress });
      setNotice(webAuthMode === "register" ? "Web-аккаунт создан." : "Вход выполнен.");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  function resetAddFlow() {
    setAddText("");
    setSelectedPackWords({});
    setIsPackExpanded(false);
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
    await Promise.all([loadDashboard(), loadStudyCardsOnly(), loadPacks()]);
  }

  async function loadPacks() {
    const data = await api("/api/packs");
    const nextPacks = data.packs || [];
    setPacks(nextPacks);
    if (!nextPacks.length) {
      return;
    }
    const nextPack = nextPacks.find((pack) => pack.id === selectedPackId) || nextPacks[0];
    const nextLevel = nextPack.levels.find((level) => level.id === selectedPackLevelId) || nextPack.levels[0];
    setSelectedPackId(nextPack.id);
    setSelectedPackLevelId(nextLevel?.id || "");
  }

  async function handleAddWords(event) {
    event.preventDefault();
    if (!addText.trim()) {
      setNotice("Добавь одно слово или фразу.");
      return;
    }
    const filledLines = addText
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    if (filledLines.length > MAX_ADD_BATCH_WORDS) {
      setNotice(`За один раз можно добавить максимум ${MAX_ADD_BATCH_WORDS} слов или фраз.`);
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

  async function preparePacksInBackground() {
    try {
      await api("/api/packs/prepare", {
        method: "POST",
        body: JSON.stringify({})
      });
    } catch {
      // best effort only
    }
  }

  async function addSelectedPack() {
    const selectedPack = packs.find((pack) => pack.id === selectedPackId) || packs[0] || null;
    const selectedLevel = selectedPack?.levels.find((level) => level.id === selectedPackLevelId) || selectedPack?.levels?.[0] || null;
    const selected = selectedLevel
      ? selectedLevel.items
        .filter((item) => selectedPackWords[item.normalized_word] ?? !item.already_added)
        .map((item) => item.normalized_word)
      : [];
    if (!selected.length) {
      setNotice("Отметь слова для добавления.");
      return;
    }
    setAddBusy(true);
    setAddBusyLabel("Добавляем пакет...");
    try {
      const data = await api("/api/packs/add", {
        method: "POST",
        body: JSON.stringify({
          pack_id: selectedPackId,
          level_id: selectedPackLevelId,
          selected_words: selected,
        }),
      });
      setWords((current) => mergeItemsById(current, data.created || []));
      setCardQueue((current) => mergeItemsById(current, data.created || []));
      setPacks(data.packs || []);
      setAuth((previous) => ({ ...previous, progress: data.progress }));
      setShowLibraryAdd(false);
      setLibraryMode("cards");
      setNotice(`Добавлено ${data.created.length} слов из пака.`);
      resetAddFlow();
      await refreshAfterWordMutation();
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
      if (data.correct) {
        setLearnCorrectCount((current) => current + 1);
      }
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
      if (data.correct) {
        setLearnCorrectCount((current) => current + 1);
      }
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
      if (data.status === "correct") {
        setLearnCorrectCount((current) => current + 1);
      }
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
      if (data.correct) {
        setIrregularCorrectCount((current) => current + 1);
      }
      setAuth((previous) => ({ ...previous, progress: data.progress }));
      await loadDashboard();
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleAlphabetAnswer(answer) {
    if (!alphabetQuestion) {
      return;
    }
    setBusy(true);
    try {
      const data = await api("/api/alphabet/answer", {
        method: "POST",
        body: JSON.stringify({
          symbol: alphabetQuestion.letter.symbol,
          answer,
        })
      });
      setAlphabetResult(data);
      if (data.correct) {
        setAlphabetCorrectCount((current) => current + 1);
      }
      setAuth((previous) => ({ ...previous, progress: data.progress }));
      await loadDashboard();
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function playAlphabetAudio(symbol) {
    if (!symbol) {
      return;
    }
    if (alphabetAudioRef.current) {
      alphabetAudioRef.current.pause();
      alphabetAudioRef.current = null;
    }
    setAlphabetAudioLoadingSymbol(symbol);
    try {
      const audio = new Audio(`/api/alphabet/audio?symbol=${encodeURIComponent(symbol)}`);
      alphabetAudioRef.current = audio;
      audio.onended = () => {
        if (alphabetAudioRef.current === audio) {
          alphabetAudioRef.current = null;
        }
        setAlphabetAudioLoadingSymbol("");
      };
      audio.onerror = () => {
        if (alphabetAudioRef.current === audio) {
          alphabetAudioRef.current = null;
        }
        setAlphabetAudioLoadingSymbol("");
      };
      await audio.play();
    } catch (error) {
      setAlphabetAudioLoadingSymbol("");
      setNotice("Не удалось воспроизвести аудио буквы.");
    }
  }

  function skipAlphabetQuestion() {
    if (!alphabetQuestion || alphabetResult) {
      return;
    }
    setAlphabetResult({
      correct: false,
      skipped: true,
      correct_answer: alphabetQuestion.letter.symbol,
    });
  }

  function skipIrregularQuestion() {
    if (!irregularQuestion || irregularResult) {
      return;
    }
    setIrregularResult({
      correct: false,
      skipped: true,
      correct_answer: irregularQuestion.correct_pair,
    });
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
    setWordImageErrors((current) => ({ ...current, [wordId]: false }));
    try {
      const data = await api(`/api/words/${wordId}/image/regenerate`, {
        method: "POST",
        body: JSON.stringify({})
      });
      setWords((current) => current.map((item) => (item.id === wordId ? data.item : item)));
      setCardQueue((current) => current.map((item) => (item.id === wordId ? data.item : item)));
      setPreviewWordId(wordId);
      setNotice("Генерируем новое фото. Можно не ждать: оно появится автоматически.");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setRegeneratingWordId(null);
    }
  }

  async function saveSettings(event) {
    event.preventDefault();
    if (
      settings?.active_studied_language === "ka"
      && !settings?.has_selected_georgian_display_mode
    ) {
      setGeorgianDisplayModePrompt({
        source: "settings",
        previousCourseCode: auth.user?.active_studied_language || "en",
      });
      return;
    }
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

  async function selectStudiedLanguage(courseCode, options = {}) {
    const hasSelectedGeorgianDisplayMode =
      settings?.has_selected_georgian_display_mode
      ?? auth.user?.has_selected_georgian_display_mode
      ?? false;
    if (
      courseCode === "ka"
      && !hasSelectedGeorgianDisplayMode
      && !options.georgianDisplayMode
    ) {
      setGeorgianDisplayModePrompt({
        source: options.source || "onboarding",
        previousCourseCode:
          settings?.active_studied_language
          || auth.user?.active_studied_language
          || "en",
      });
      return;
    }
    setBusy(true);
    try {
      await api("/api/settings", {
        method: "POST",
        body: JSON.stringify({
          active_studied_language: courseCode,
          ...(options.georgianDisplayMode
            ? { georgian_display_mode: options.georgianDisplayMode }
            : {}),
        }),
      });
      setGeorgianDisplayModePrompt(null);
      setPrimaryTab("today");
      await Promise.all([
        loadDashboard(),
        loadStudyCardsOnly(),
        loadPacks(),
        courseCode === "en" ? loadIrregularQuestion() : Promise.resolve(),
      ]);
      setNotice("Язык обучения сохранен.");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function confirmGeorgianDisplayMode(mode) {
    if (!georgianDisplayModePrompt) {
      return;
    }
    if (georgianDisplayModePrompt.source === "settings") {
      setSettings((current) => ({
        ...current,
        active_studied_language: "ka",
        georgian_display_mode: mode,
        has_selected_georgian_display_mode: true,
      }));
      setGeorgianDisplayModePrompt(null);
      setNotice("Режим отображения сохранится после нажатия «Сохранить настройки».");
      return;
    }
    await selectStudiedLanguage("ka", {
      georgianDisplayMode: mode,
      source: georgianDisplayModePrompt.source,
    });
  }

  function cancelGeorgianDisplayModePrompt() {
    if (georgianDisplayModePrompt?.source === "settings") {
      setSettings((current) => ({
        ...current,
        active_studied_language:
          georgianDisplayModePrompt.previousCourseCode || "en",
      }));
    }
    setGeorgianDisplayModePrompt(null);
  }

  function openLearn() {
    if (showLibraryAdd) {
      void closeAddWords();
    }
    startTransition(() => {
      setPrimaryTab("learn");
      setShowLibraryAdd(false);
    });
  }

  function openAddWords() {
    resetAddFlow();
    startTransition(() => {
      setPrimaryTab("words");
      setLibraryMode("cards");
      setShowLibraryAdd(true);
    });
  }

  function closeLearnSession() {
    const hasProgress = Boolean(learnQuestion || learnResult || learnQuestionCount > 0 || learnSessionDone);
    if (hasProgress && !window.confirm("Закрыть практику? Текущий прогресс в этой сессии сбросится.")) {
      return;
    }
    if (isRecording) {
      stopSpeakingRecording();
    }
    setLearnQuestion(null);
    setLearnResult(null);
    setLearnSelection("");
    setLearnTextAnswer("");
    setLearnUsedWordIds([]);
    setLearnQuestionCount(0);
    setLearnCorrectCount(0);
    setLearnSessionDone(false);
    setLearnPanel("mixed");
  }

  function backFromIrregularReview() {
    setLearnPanel("mixed");
    setIrregularMode("review");
  }

  function backFromAlphabetReview() {
    setLearnPanel("mixed");
    setAlphabetMode("review");
  }

  function closeIrregularTest() {
    const hasProgress = Boolean(irregularQuestion || irregularResult);
    if (hasProgress && !window.confirm("Закрыть тест по глаголам? Текущий прогресс в этом экране сбросится.")) {
      return;
    }
    setIrregularQuestion(null);
    setIrregularResult(null);
    setIrregularQuestionCount(0);
    setIrregularCorrectCount(0);
    setIrregularSessionDone(false);
    setLearnPanel("mixed");
    setIrregularMode("test");
  }

  function closeAlphabetTest() {
    const hasProgress = Boolean(alphabetQuestion || alphabetResult);
    if (!hasProgress) {
      backFromAlphabetReview();
      return;
    }
    setAlphabetQuestion(null);
    setAlphabetResult(null);
    setAlphabetQuestionCount(0);
    setAlphabetCorrectCount(0);
    setAlphabetSessionDone(false);
    setLearnPanel("alphabet");
    setAlphabetMode("test");
  }

  async function startIrregularTest() {
    setIrregularQuestionCount(0);
    setIrregularCorrectCount(0);
    setIrregularSessionDone(false);
    setIrregularSessionLimit(settings?.session_question_limit || 12);
    await loadIrregularQuestion();
  }

  async function startAlphabetTest() {
    setAlphabetQuestionCount(0);
    setAlphabetCorrectCount(0);
    setAlphabetSessionDone(false);
    setAlphabetSessionLimit(settings?.session_question_limit || 12);
    await loadAlphabetQuestion();
  }

  async function advanceIrregularTest() {
    const nextCount = irregularQuestionCount + 1;
    if (nextCount >= irregularSessionLimit) {
      setIrregularQuestion(null);
      setIrregularResult(null);
      setIrregularQuestionCount(nextCount);
      setIrregularSessionDone(true);
      return;
    }
    setIrregularQuestionCount(nextCount);
    await loadIrregularQuestion();
  }

  async function advanceAlphabetTest() {
    const nextCount = alphabetQuestionCount + 1;
    if (nextCount >= alphabetSessionLimit) {
      setAlphabetQuestion(null);
      setAlphabetResult(null);
      setAlphabetQuestionCount(nextCount);
      setAlphabetSessionDone(true);
      return;
    }
    setAlphabetQuestionCount(nextCount);
    await loadAlphabetQuestion();
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
              <h2 className="study-word">{formatDisplayLine(currentCard.word, currentCard.course_code).primary}</h2>
              {formatDisplayLine(currentCard.word, currentCard.course_code).secondary ? (
                <p className="word-romanization">
                  {formatDisplayLine(currentCard.word, currentCard.course_code).secondary}
                </p>
              ) : null}
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
        ) : (
          <div className="stack-form">
            <div className="empty-state">Пока нет карточек. Добавь новые слова для изучения.</div>
            <button className="primary-button add-words-ghost" type="button" onClick={openAddWords}>
              ＋ Добавить слова
            </button>
          </div>
        )}
      </section>
    );
  }

  function renderLearn() {
    const hasWordsToLearn = (auth.progress?.learning ?? 0) > 0;
    const hasActiveIrregularTest = supportsIrregularPractice && learnPanel === "irregular" && irregularMode === "test";
    const hasActiveAlphabetTest = learnPanel === "alphabet" && alphabetMode === "test";
    const showLearnOverview = !["irregular", "alphabet"].includes(learnPanel) && !learnQuestion && !hasActiveIrregularTest && !hasActiveAlphabetTest;

    if (showLearnOverview) {
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
                <strong>Сессия завершена.</strong> Верно: {learnCorrectCount} из {learnQuestionCount || learnSessionLimit}.
              </div>
            ) : null}
            <div className="button-row">
              {hasWordsToLearn ? (
                <button className="primary-button" type="button" onClick={() => void loadLearningData()}>
                  {learnSessionDone ? "Начать новую сессию" : "Начать сессию"}
                </button>
              ) : null}
              <button className="secondary-button" type="button" onClick={openAddWords}>
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
                  <p className="lead compact">Можно быстро повторять формы или пройти отдельный тест.</p>
                </div>
              </div>
              <div className="segment-wrap main-segment">
                {IRREGULAR_MODES.map((item) => (
                  <button
                    key={item.id}
                    className={irregularMode === item.id ? "segment-button active" : "segment-button"}
                    type="button"
                    onClick={() => {
                      setLearnPanel("irregular");
                      setIrregularMode(item.id);
                      if (item.id === "test" && !irregularQuestion) {
                        void startIrregularTest();
                      }
                    }}
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
                <p className="lead compact">Буквы, названия и транскрипция для текущего языка обучения.</p>
              </div>
            </div>
            <div className="segment-wrap main-segment">
              {ALPHABET_MODES.map((item) => (
                <button
                  key={item.id}
                  className={alphabetMode === item.id ? "segment-button active" : "segment-button"}
                  type="button"
                  onClick={() => {
                    setLearnPanel("alphabet");
                    setAlphabetMode(item.id);
                    if (item.id === "test" && !alphabetQuestion) {
                      void startAlphabetTest();
                    }
                  }}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </section>
        </div>
      );
    }

    if (supportsIrregularPractice && learnPanel === "irregular") {
      return (
        <div className="screen-stack">
          {renderIrregular()}
        </div>
      );
    }

    if (learnPanel === "alphabet") {
      return (
        <div className="screen-stack">
          {renderAlphabet()}
        </div>
      );
    }

    const statusClass = learnResult?.status === "correct"
      ? "result-box good"
      : learnResult?.status === "close"
        ? "result-box"
        : "result-box bad";

    const isChoice = learnQuestion.kind === "choice";
    const isListening = learnQuestion.kind === "listening";
    const isSpeaking = learnQuestion.kind === "speaking";
    const promptTitle = isChoice
      ? (learnQuestion.exercise_type === "practice_ru_en" ? learnQuestion.item.translation : learnQuestion.item.word)
      : isListening
        ? (learnQuestion.exercise_type === "listening_word" ? "Введите слово 🎧" : "Введите перевод 🎧")
        : learnQuestion.item.word;
    const promptTitleDisplay = formatDisplayAnswer(
      promptTitle,
      learnQuestion.item?.course_code,
    );
    const learnCorrectAnswerText = formatDisplayAnswer(
      formatLearnCorrectAnswer(learnQuestion, learnResult),
      learnQuestion.item?.course_code,
    );
    const learnResultText = learnResult
      ? (() => {
        if (learnResult.skipped) {
          return `Правильный ответ: ${learnCorrectAnswerText}`;
        }
        if (learnResult.correct && learnResult.accepted_with_typo) {
          return `Верно, засчитано с опечаткой. Правильно пишется: ${learnCorrectAnswerText}`;
        }
        if (learnResult.correct) {
          return "Верно";
        }
        return `Правильный ответ: ${learnCorrectAnswerText}`;
      })()
      : "";

    return (
      <div className="screen-stack">
        <section className="glass-card learn-card">
          <div className="section-head section-head-wrap">
            <div>
              <p className="overline">Practice</p>
              <h3>Учить слова 🎯</h3>
            </div>
            <span className="status-tag">{learnQuestionCount + 1} / {learnSessionLimit}</span>
          </div>
          <div className="prompt-card">
            <strong>{promptTitleDisplay}</strong>
            {isSpeaking && learnQuestion.item?.transcription ? (
              <p className="transcription">/{learnQuestion.item.transcription}/</p>
            ) : null}
            <span>{learnQuestion.prompt}</span>
            <span className="study-hint">{learnQuestion.exercise_label}</span>
          </div>
          {isChoice ? (
            <div className="quiz-panel quiz-panel-tight">
              <div className="option-grid">
                {learnQuestion.options.map((option) => (
                  <button
                    key={option}
                    className={`option-button ${getOptionState(option, learnResult, learnSelection)}`.trim()}
                    type="button"
                    disabled={Boolean(learnResult)}
                    onClick={() => void handleLearnChoiceAnswer(option)}
                  >
                    {formatDisplayAnswer(option, learnQuestion.item?.course_code)}
                  </button>
                ))}
              </div>
              <button className="secondary-button" type="button" onClick={revealLearnAnswer} disabled={Boolean(learnResult)}>
                Пропустить
              </button>
            </div>
          ) : null}
          {isListening ? (
            <form className="stack-form quiz-panel-tight" onSubmit={handleLearnListeningSubmit}>
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
            <div className="quiz-panel quiz-panel-tight">
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
                  ? learnResult.skipped
                    ? `Правильный ответ: ${formatDisplayAnswer(learnResult.correct_answer, learnQuestion.item?.course_code)}`
                    : `${learnResult.message} Транскрибация: ${learnResult.transcript || "—"}.`
                  : learnResultText}
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
                  {formatDisplayLine(item.word, item.course_code).secondary ? (
                    <p className="word-item-romanization">
                      {formatDisplayLine(item.word, item.course_code).secondary}
                    </p>
                  ) : null}
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
                      className={regeneratingWordId === item.id || item.image_generation_in_progress ? "secondary-button is-loading" : "secondary-button"}
                      type="button"
                      onClick={() => regenerateWordImage(item.id)}
                      disabled={regeneratingWordId === item.id || item.image_generation_in_progress || (item.image_regeneration_count || 0) >= MAX_IMAGE_REGENERATIONS}
                    >
                      {regeneratingWordId === item.id || item.image_generation_in_progress
                        ? "⏳ Генерируем..."
                        : (item.image_regeneration_count || 0) >= MAX_IMAGE_REGENERATIONS
                          ? "Лимит фото"
                          : "♻️ Обновить фото"}
                    </button>
                    <button className="danger-button" type="button" onClick={() => deleteWord(item.id)}>
                      Удалить
                    </button>
                  </div>
                  {item.image_generation_in_progress ? (
                    <div className="inline-note status-note">
                      <strong>Генерируем новое фото...</strong> Старое изображение останется до обновления.
                    </div>
                  ) : null}
                  {previewWordId === item.id ? (
                    item.has_image && !wordImageErrors[item.id] ? (
                      <div className="word-image-preview">
                        <img
                          key={wordImageVersions[item.id] || item.updated_at}
                          src={`/api/image/${item.id}?v=${wordImageVersions[item.id] || item.updated_at}`}
                          alt={item.word}
                          onLoad={() => setWordImageErrors((current) => ({ ...current, [item.id]: false }))}
                          onError={() => setWordImageErrors((current) => ({ ...current, [item.id]: true }))}
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
          ))}
          {!words.length ? <div className="glass-card empty-card">No words found.</div> : null}
        </div>
      </>
    );
  }

  function renderAddWords() {
    const isTranslationStep = addDraftStep === "confirm_translation";
    const isImageStep = addDraftStep === "confirm_image";
    const isBatchReview = addDraftStep === "batch_review";
    const addWordPlaceholder =
      activeStudiedLanguage === "ka"
        ? "გამარჯობა\nმადლობა\nგზა - дорога"
        : "stare\nfigure out\ntravel - путешествие";
    const addWordHint =
      activeStudiedLanguage === "ka"
        ? "По одному слову или фразе на строку. Перевод можно указать через дефис."
        : "По одному слову или фразе на строку. Перевод можно указать через дефис.";
    const displayPacks = [...packs].sort((left, right) => {
      const leftPriority = left.id === "travel" ? 2 : 1;
      const rightPriority = right.id === "travel" ? 2 : 1;
      if (leftPriority !== rightPriority) {
        return leftPriority - rightPriority;
      }
      return 0;
    });
    const selectedPack = displayPacks.find((pack) => pack.id === selectedPackId) || displayPacks[0] || null;
    const selectedLevel = selectedPack?.levels.find((level) => level.id === selectedPackLevelId) || selectedPack?.levels?.[0] || null;
    const selectedWordCount = selectedLevel
      ? selectedLevel.items.filter((item) => selectedPackWords[item.normalized_word] ?? !item.already_added).length
      : 0;

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
          <div className="stack-form">
            <form className="stack-form" onSubmit={handleAddWords}>
              <div className="inline-note">
                {addWordHint}
              </div>
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
            <section className="glass-card compact-section pack-section">
              <div className="section-head">
                <div>
                  <p className="overline">Наборы</p>
                  <h3>Готовые наборы ✈️</h3>
                  <p className="lead compact">Выбери готовый набор по ситуации.</p>
                </div>
              </div>
              {displayPacks.length ? (
                <>
                  <div className="pack-card-list">
                    {displayPacks.map((pack) => {
                      const isActivePack = selectedPackId === pack.id;
                      const totalWords = pack.levels.reduce((sum, level) => sum + level.size, 0);

                      return (
                        <article key={pack.id} className={isActivePack ? "pack-card active" : "pack-card"}>
                          <div className="pack-card-header">
                            <div className="pack-card-copy">
                              <div className="pack-card-title-row">
                                <span className="pack-card-emoji">{pack.emoji}</span>
                                <strong>{pack.title}</strong>
                              </div>
                              <p className="pack-card-description">{pack.description}</p>
                            </div>
                            <button
                              className={isActivePack && isPackExpanded ? "secondary-button" : "primary-button"}
                              type="button"
                              onClick={() => {
                                setSelectedPackId(pack.id);
                                setSelectedPackLevelId(pack.levels[0]?.id || "");
                                setSelectedPackWords({});
                                setIsPackExpanded((current) => (isActivePack ? !current : true));
                              }}
                            >
                              {isActivePack && isPackExpanded ? "Скрыть" : "Открыть"}
                            </button>
                          </div>
                          <div className="pack-card-meta">
                            <span className="pack-badge">
                              {pack.levels.length === 1 ? "1 уровень" : `${pack.levels.length} уровней`}
                            </span>
                            <span className="pack-badge">{totalWords} слов и фраз</span>
                          </div>
                          {isActivePack && isPackExpanded ? (
                            <>
                              {pack.levels.length > 1 ? (
                                <div className="pack-list pack-level-list">
                                  {pack.levels.map((level) => (
                                    <button
                                      key={level.id}
                                      className={selectedPackLevelId === level.id ? "segment-button active" : "segment-button"}
                                      type="button"
                                      onClick={() => {
                                        setSelectedPackLevelId(level.id);
                                        setSelectedPackWords({});
                                      }}
                                    >
                                      {level.title} · {level.size}
                                    </button>
                                  ))}
                                </div>
                              ) : null}
                              <p className="inline-note pack-level-note">{selectedLevel?.description}</p>
                              <div className="pack-word-grid">
                                {selectedLevel?.items.map((item) => {
                                  const checked = selectedPackWords[item.normalized_word] ?? !item.already_added;
                                  return (
                                    <label key={item.normalized_word} className={item.already_added ? "pack-word-row muted" : "pack-word-row"}>
                                      <input
                                        type="checkbox"
                                        checked={checked}
                                        disabled={item.already_added}
                                        onChange={(event) => setSelectedPackWords((current) => ({ ...current, [item.normalized_word]: event.target.checked }))}
                                      />
                                      <span className="pack-word-main">
                                        <strong>{item.word}</strong>
                                        <small>{item.translation}</small>
                                      </span>
                                      {item.already_added ? <em className="pack-word-state">Уже есть</em> : null}
                                    </label>
                                  );
                                })}
                              </div>
                              <div className="button-row batch-save-row">
                                <button className="primary-button" type="button" onClick={addSelectedPack} disabled={addBusy || !selectedWordCount}>
                                  {addBusy ? "Добавляем..." : `Добавить ${selectedWordCount} слов и фраз`}
                                </button>
                              </div>
                            </>
                          ) : null}
                        </article>
                      );
                    })}
                  </div>
                </>
              ) : (
                <div className="empty-state">Готовим первый набор...</div>
              )}
            </section>
          </div>
        ) : null}

        {isBatchReview && addDrafts.length ? (
          <div className="word-list batch-draft-list">
            <div className="inline-note status-note">
              <strong>Можно сохранять слова сразу.</strong> Если часть фото ещё не появилась, они догрузятся автоматически позже и подтянутся в карточках.
            </div>
            {addDrafts.map((draft) => (
              <article className="glass-card word-item" key={draft.id}>
                <div className="word-item-head">
                  <div>
                    <strong>{draft.word}</strong>
                    {formatDisplayLine(draft.word, draft.course_code).secondary ? (
                      <p className="word-item-romanization">
                        {formatDisplayLine(draft.word, draft.course_code).secondary}
                      </p>
                    ) : null}
                    <p>{draft.example}</p>
                  </div>
                  <span className="status-tag">{draft.part_of_speech || "word"}</span>
                </div>
                <input
                  value={batchTranslations[draft.id] ?? draft.translation}
                  onChange={(event) => setBatchTranslations((current) => ({ ...current, [draft.id]: event.target.value }))}
                />
                {draft.has_image && draftImageVersions[draft.id] === draft.updated_at ? (
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
              {formatDisplayLine(addDraft.word, addDraft.course_code).secondary ? (
                <span className="word-romanization inline-romanization">
                  {formatDisplayLine(addDraft.word, addDraft.course_code).secondary}
                </span>
              ) : null}
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
                {addDraft.has_image && draftImageVersions[addDraft.id] === addDraft.updated_at ? (
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
                {formatDisplayLine(addDraft.word, addDraft.course_code).secondary ? (
                  <span className="word-romanization inline-romanization">
                    {formatDisplayLine(addDraft.word, addDraft.course_code).secondary}
                  </span>
                ) : null}
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
        {irregularMode === "review" ? (
          <section className="glass-card compact-section">
            <div className="section-head">
              <div>
                <p className="overline">Irregular</p>
                <h3>Повторять глаголы 📘</h3>
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
            <div className="button-row card-nav-row">
              <button className="secondary-button nav-arrow" type="button" onClick={() => setIrregularPage((value) => Math.max(0, value - 1))} disabled={!irregularList?.has_prev} aria-label="Предыдущая страница">
                ←
              </button>
              <button className="secondary-button nav-arrow" type="button" onClick={() => setIrregularPage((value) => value + 1)} disabled={!irregularList?.has_next} aria-label="Следующая страница">
                →
              </button>
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
              <span className="status-tag">{Math.min(irregularQuestionCount + 1, irregularSessionLimit)} / {irregularSessionLimit}</span>
            </div>
            {irregularQuestion ? (
              <div className="quiz-panel">
                <div className="prompt-card">
                  <strong>{irregularQuestion.verb.base}</strong>
                  <span>Выбери правильную форму</span>
                </div>
                <div className="option-grid">
                  {irregularQuestion.options.map((option) => (
                    <button key={option} className="option-button" type="button" onClick={() => handleIrregularAnswer(option)}>
                      {option}
                    </button>
                  ))}
                </div>
                {!irregularResult ? (
                  <button className="secondary-button" type="button" onClick={skipIrregularQuestion}>
                    Пропустить
                  </button>
                ) : null}
                {irregularResult ? (
                  <div className={irregularResult.correct ? "result-box good" : "result-box bad"}>
                    <span>{irregularResult.correct ? "Верно" : `Правильный ответ: ${irregularResult.correct_answer}`}</span>
                    <button className="secondary-button" type="button" onClick={() => void advanceIrregularTest()}>
                      Дальше
                    </button>
                  </div>
                ) : null}
              </div>
            ) : (
                <div className="stack-form">
                  <div className="empty-state">
                    {irregularSessionDone
                    ? `Тест завершён. Верно ${irregularCorrectCount} из ${irregularQuestionCount || irregularSessionLimit}. ${getSessionPraise(irregularCorrectCount, irregularQuestionCount || irregularSessionLimit)}`
                    : "Сейчас нет вопроса по глаголам."}
                  </div>
                {irregularSessionDone ? (
                  <button className="primary-button" type="button" onClick={() => void startIrregularTest()}>
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

  function renderAlphabet() {
    return (
      <div className="screen-stack">
        {alphabetMode === "review" ? (
          <section className="glass-card compact-section">
            <div className="section-head">
              <div>
                <p className="overline">Alphabet</p>
                <h3>Повторять алфавит 🔤</h3>
              </div>
            </div>
            <div className="simple-list">
              {(alphabetList?.items || []).map((item) => (
                <div key={item.symbol} className="simple-row four-cols">
                  <strong>{item.symbol}</strong>
                  <span>{item.name}</span>
                  <span>/{item.transcription}/</span>
                  <div className="alphabet-audio-cell">
                    <span>{item.hint}</span>
                    <button
                      className="secondary-button mini-audio-button"
                      type="button"
                      onClick={() => void playAlphabetAudio(item.symbol)}
                      disabled={alphabetAudioLoadingSymbol === item.symbol}
                      aria-label={`Слушать букву ${item.symbol}`}
                    >
                      {alphabetAudioLoadingSymbol === item.symbol ? "..." : "🔊"}
                    </button>
                  </div>
                </div>
              ))}
            </div>
            <div className="button-row card-nav-row">
              <button className="secondary-button nav-arrow" type="button" onClick={() => setAlphabetPage((value) => Math.max(0, value - 1))} disabled={!alphabetList?.has_prev} aria-label="Предыдущая страница">
                ←
              </button>
              <button className="secondary-button nav-arrow" type="button" onClick={() => setAlphabetPage((value) => value + 1)} disabled={!alphabetList?.has_next} aria-label="Следующая страница">
                →
              </button>
            </div>
          </section>
        ) : null}
        {alphabetMode === "test" ? (
          <section className="glass-card compact-section">
            <div className="section-head">
              <div>
                <p className="overline">Alphabet</p>
                <h3>Тест по алфавиту 🧠</h3>
              </div>
              <span className="status-tag">{Math.min(alphabetQuestionCount + 1, alphabetSessionLimit)} / {alphabetSessionLimit}</span>
            </div>
            {alphabetQuestion ? (
              <div className="quiz-panel">
                <div className="prompt-card">
                  <strong>/{alphabetQuestion.letter.transcription}/</strong>
                  <span>{formatDisplayAnswer(alphabetQuestion.letter.symbol, alphabetQuestion.course_code)}</span>
                  <span>{alphabetQuestion.letter.hint}</span>
                </div>
                <div className="option-grid">
                  {alphabetQuestion.options.map((option) => (
                    <button key={option} className="option-button" type="button" onClick={() => handleAlphabetAnswer(option)}>
                      {formatDisplayAnswer(option, alphabetQuestion.course_code)}
                    </button>
                  ))}
                </div>
                {!alphabetResult ? (
                  <button className="secondary-button" type="button" onClick={skipAlphabetQuestion}>
                    Пропустить
                  </button>
                ) : null}
                {alphabetResult ? (
                  <div className={alphabetResult.correct ? "result-box good" : "result-box bad"}>
                    <span>
                      {alphabetResult.correct
                        ? "Верно"
                        : `Правильный ответ: ${formatDisplayAnswer(
                          alphabetResult.correct_answer,
                          alphabetQuestion.course_code,
                        )}`}
                    </span>
                    <button className="secondary-button" type="button" onClick={() => void advanceAlphabetTest()}>
                      Дальше
                    </button>
                  </div>
                ) : null}
              </div>
            ) : (
              <div className="stack-form">
                <div className="empty-state">
                  {alphabetSessionDone
                    ? `Тест завершён. Верно ${alphabetCorrectCount} из ${alphabetQuestionCount || alphabetSessionLimit}. ${getSessionPraise(alphabetCorrectCount, alphabetQuestionCount || alphabetSessionLimit)}`
                    : "Сейчас нет вопроса по алфавиту."}
                </div>
                {alphabetSessionDone ? (
                  <button className="primary-button" type="button" onClick={() => void startAlphabetTest()}>
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

  function renderMore() {
    return (
      <SettingsScreen
        settings={settings}
        onSave={saveSettings}
        onChange={(field, value) =>
          setSettings((current) => {
            if (
              field === "active_studied_language"
              && value === "ka"
              && !current.has_selected_georgian_display_mode
            ) {
              setGeorgianDisplayModePrompt({
                source: "settings",
                previousCourseCode: current.active_studied_language || "en",
              });
              return {
                ...current,
                active_studied_language: "ka",
                georgian_display_mode: current.georgian_display_mode || "both",
              };
            }
            if (field === "georgian_display_mode") {
              return {
                ...current,
                georgian_display_mode: value,
                has_selected_georgian_display_mode: true,
              };
            }
            return { ...current, [field]: value };
          })
        }
      />
    );
  }

  function renderScreen() {
    if (primaryTab === "today") {
      return (
        <TodayScreen
          progress={auth.progress}
          todayStats={todayStats}
          todayAchievements={todayAchievements}
          hasMoreAchievements={hasMoreAchievements}
          hasWordsToLearn={(auth.progress?.learning ?? 0) > 0}
          onOpenAddWords={openAddWords}
          onOpenLearn={() => openLearn("practice")}
          onOpenProgress={() => setPrimaryTab("progress")}
        />
      );
    }
    if (primaryTab === "learn") return renderLearn();
    if (primaryTab === "words") return renderLibrary();
    if (primaryTab === "progress") {
      return <ProgressScreen progress={auth.progress} stats={stats} />;
    }
    return renderMore();
  }

  if (auth.loading) {
    return <div className="boot-screen">Loading VocabuMe...</div>;
  }

  if (!auth.authenticated) {
    return (
      <div className={`app-shell auth-layout${isKeyboardOpen ? " keyboard-open" : ""}`}>
        <main className="auth-stage">
          {notice ? <div className="notice">{notice.message}</div> : null}
          <AuthPanel
            config={config}
            onOpenLogin={requestLoginLink}
            loginLink={loginLink}
            loginPending={busy}
            webAuthMode={webAuthMode}
            onChangeWebAuthMode={setWebAuthMode}
            onSubmitWebAuth={submitWebAuth}
            webAuthPending={busy}
            webEmail={webEmail}
            webPassword={webPassword}
            onWebEmailChange={setWebEmail}
            onWebPasswordChange={setWebPassword}
          />
        </main>
      </div>
    );
  }

  if (georgianDisplayModePrompt && !needsStudiedLanguageSelection) {
    return (
      <div className={`app-shell auth-layout${isKeyboardOpen ? " keyboard-open" : ""}`}>
        <main className="auth-stage">
          {notice ? <div className="notice">{notice.message}</div> : null}
            <section className="glass-card compact-section">
              <p className="overline">Грузинский</p>
              <h3>Как показывать грузинский? ✨</h3>
              <p className="lead compact">
                Для старта рекомендуем показывать и грузинское письмо, и латиницу. Позже это всегда можно изменить в настройках.
              </p>
            <div className="stack-form">
              {georgianDisplayModeOptions.map((item) => (
                <button
                  key={item.code}
                  className={item.recommended ? "primary-button" : "secondary-button"}
                  type="button"
                  onClick={() => void confirmGeorgianDisplayMode(item.code)}
                  disabled={busy}
                >
                  {item.label}{item.recommended ? " (Рекомендуется)" : ""}
                </button>
              ))}
              <button className="secondary-button" type="button" onClick={cancelGeorgianDisplayModePrompt} disabled={busy}>
                Отмена
              </button>
            </div>
          </section>
        </main>
      </div>
    );
  }

  if (needsStudiedLanguageSelection) {
    if (georgianDisplayModePrompt) {
      return (
        <div className={`app-shell auth-layout${isKeyboardOpen ? " keyboard-open" : ""}`}>
          <main className="auth-stage">
            {notice ? <div className="notice">{notice.message}</div> : null}
            <section className="glass-card compact-section">
              <p className="overline">Грузинский</p>
              <h3>Как показывать грузинский? ✨</h3>
              <p className="lead compact">
                Для старта рекомендуем показывать и грузинское письмо, и латиницу. Позже это всегда можно изменить в настройках.
              </p>
              <div className="stack-form">
                {georgianDisplayModeOptions.map((item) => (
                  <button
                    key={item.code}
                    className={item.recommended ? "primary-button" : "secondary-button"}
                    type="button"
                    onClick={() => void confirmGeorgianDisplayMode(item.code)}
                    disabled={busy}
                  >
                    {item.label}{item.recommended ? " (Рекомендуется)" : ""}
                  </button>
                ))}
                <button className="secondary-button" type="button" onClick={cancelGeorgianDisplayModePrompt} disabled={busy}>
                  Назад
                </button>
              </div>
            </section>
          </main>
        </div>
      );
    }
    return (
      <div className={`app-shell auth-layout${isKeyboardOpen ? " keyboard-open" : ""}`}>
        <main className="auth-stage">
          {notice ? <div className="notice">{notice.message}</div> : null}
          <section className="glass-card compact-section">
            <p className="overline">Первый запуск</p>
            <h3>Какой язык ты хочешь учить? ✨</h3>
            <p className="lead compact">
              Сначала выбери язык обучения. Прогресс, слова и готовые наборы будут
              храниться отдельно для каждого языка.
            </p>
            <div className="pack-list">
              {(auth.user?.available_studied_languages || []).map((item) => (
                <button
                  key={item.code}
                  className="segment-button active"
                  type="button"
                  onClick={() => void selectStudiedLanguage(item.code, { source: "onboarding" })}
                  disabled={busy}
                >
                  {busy ? "Сохраняем..." : item.label}
                </button>
              ))}
            </div>
          </section>
        </main>
      </div>
    );
  }

  return (
    <div className={`app-shell${isKeyboardOpen ? " keyboard-open" : ""}`}>
      <AppTopbar
        busy={busy}
        currentTitle={currentTitle}
        isMiniApp={isMiniApp}
        onBack={learnPanel === "alphabet" ? backFromAlphabetReview : backFromIrregularReview}
        onClose={
          learnPanel === "irregular" && irregularMode === "test"
            ? closeIrregularTest
            : learnPanel === "alphabet" && alphabetMode === "test"
              ? closeAlphabetTest
            : closeLearnSession
        }
        onLogout={logoutWeb}
        onToggleAddWords={() => {
          if (showLibraryAdd) {
            void closeAddWords();
            return;
          }
          openAddWords();
        }}
        primaryTab={primaryTab}
        showHeaderBack={showHeaderBack}
        showHeaderClose={showHeaderClose}
        showLibraryAdd={showLibraryAdd}
      />

      {notice ? <div className="notice">{notice.message}</div> : null}

      <main className="app-stage" ref={stageRef}>
        <AppTopbar
          busy={busy}
          currentTitle={currentTitle}
          extraClass="desktop-scroll-topbar"
          isMiniApp={isMiniApp}
          onBack={learnPanel === "alphabet" ? backFromAlphabetReview : backFromIrregularReview}
          onClose={
            learnPanel === "irregular" && irregularMode === "test"
              ? closeIrregularTest
              : learnPanel === "alphabet" && alphabetMode === "test"
                ? closeAlphabetTest
              : closeLearnSession
          }
          onLogout={logoutWeb}
          onToggleAddWords={() => {
            if (showLibraryAdd) {
              void closeAddWords();
              return;
            }
            openAddWords();
          }}
          primaryTab={primaryTab}
          showHeaderBack={showHeaderBack}
          showHeaderClose={showHeaderClose}
          showLibraryAdd={showLibraryAdd}
        />
        {renderScreen()}
      </main>

      <BottomNav
        isKeyboardOpen={isKeyboardOpen}
        primaryTab={primaryTab}
        onSelectTab={(tabId) =>
          startTransition(() => {
            if (tabId !== "words" && showLibraryAdd) {
              void closeAddWords();
            }
            setPrimaryTab(tabId);
            if (tabId !== "words") {
              setShowLibraryAdd(false);
            }
          })
        }
      />
    </div>
  );
}

export default App;
