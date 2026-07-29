import {
  startTransition,
  useCallback,
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
} from "./constants";
import { api, reportClientError } from "./lib/api";
import ProgressScreen from "./screens/ProgressScreen";
import SettingsScreen from "./screens/SettingsScreen";
import TodayScreen from "./screens/TodayScreen";
import OnboardingGate from "./screens/OnboardingGate";
import CardsScreen from "./screens/CardsScreen";
import PacksScreen from "./screens/PacksScreen";
import WordsListScreen from "./screens/WordsListScreen";
import AddWordsScreen from "./screens/AddWordsScreen";
import {
  AlphabetPracticeScreen,
  IrregularPracticeScreen,
} from "./screens/PracticeSupplementScreens";

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

function formatPointsLabel(points) {
  const value = Math.abs(Number(points) || 0);
  const mod10 = value % 10;
  const mod100 = value % 100;
  if (mod10 === 1 && mod100 !== 11) {
    return "очко";
  }
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) {
    return "очка";
  }
  return "очков";
}

function formatPauseRemaining(untilIso) {
  if (!untilIso) {
    return "";
  }
  const remainingMs = new Date(untilIso).getTime() - Date.now();
  if (remainingMs <= 0) {
    return "";
  }
  const minutes = Math.max(1, Math.ceil(remainingMs / 60000));
  return `${minutes} мин`;
}

function formatLearnCorrectAnswer(learnQuestion, learnResult) {
  const answer = learnResult?.correct_answer || "";
  if (!learnQuestion || !learnResult) {
    return answer;
  }
  if (
    learnQuestion.exercise_type === "listening_translate" &&
    learnQuestion.item?.word
  ) {
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

function withFreshAvatarUrl(payload) {
  if (!payload?.avatar_url) {
    return payload;
  }
  const separator = payload.avatar_url.includes("?") ? "&" : "?";
  return {
    ...payload,
    avatar_url: `${payload.avatar_url}${separator}r=${Date.now()}`,
  };
}

async function preloadImage(src, attempts = 4) {
  if (!src) {
    return false;
  }
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
    await new Promise((resolve) => setTimeout(resolve, 150));
  }
  return false;
}

const GEORGIAN_TO_LATIN = {
  ა: "a",
  ბ: "b",
  გ: "g",
  დ: "d",
  ე: "e",
  ვ: "v",
  ზ: "z",
  თ: "t",
  ი: "i",
  კ: "k'",
  ლ: "l",
  მ: "m",
  ნ: "n",
  ო: "o",
  პ: "p'",
  ჟ: "zh",
  რ: "r",
  ს: "s",
  ტ: "t'",
  უ: "u",
  ფ: "p",
  ქ: "k",
  ღ: "gh",
  ყ: "q'",
  შ: "sh",
  ჩ: "ch",
  ც: "ts",
  ძ: "dz",
  წ: "ts'",
  ჭ: "ch'",
  ხ: "kh",
  ჯ: "j",
  ჰ: "h",
};

function transliterateGeorgian(text) {
  return Array.from(text || "")
    .map((char) => GEORGIAN_TO_LATIN[char] || char)
    .join("");
}

function hasGeorgianScript(text) {
  return /[\u10A0-\u10FF]/.test(text || "");
}

function makeAchievementKey(text, courseCode = "en") {
  return `${courseCode}:${text}`;
}

function waitForAudioPreparation(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function App() {
  const [config, setConfig] = useState({ bot_username: "", webapp_url: "" });
  const [auth, setAuth] = useState({
    loading: true,
    authenticated: false,
    user: null,
    progress: null,
  });
  const [notice, setNoticeState] = useState(null);
  const [achievementToast, setAchievementToast] = useState(null);
  const [achievementQueue, setAchievementQueue] = useState([]);
  const [busy, setBusy] = useState(false);
  const [primaryTab, setPrimaryTab] = useState("today");
  const [libraryMode, setLibraryMode] = useState("cards");
  const [learnPanel, setLearnPanel] = useState("mixed");
  const [showLibraryAdd, setShowLibraryAdd] = useState(false);
  const [dashboard, setDashboard] = useState(null);
  const [settings, setSettings] = useState(null);
  const [uploadingAvatar, setUploadingAvatar] = useState(false);
  const [checkoutBusyPeriod, setCheckoutBusyPeriod] = useState("");
  const [words, setWords] = useState([]);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [draftTranslation, setDraftTranslation] = useState({});
  const [previewWordId, setPreviewWordId] = useState(null);
  const [expandedWordId, setExpandedWordId] = useState(null);
  const [wordImageVersions] = useState({});
  const [wordImageErrors, setWordImageErrors] = useState({});
  const [regeneratingWordId, setRegeneratingWordId] = useState(null);
  const [audioRevision, setAudioRevision] = useState(0);
  const [addText, setAddText] = useState("");
  const [packs, setPacks] = useState([]);
  const [packsLoading, setPacksLoading] = useState(false);
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
  const [alphabetAudioLoadingSymbol, setAlphabetAudioLoadingSymbol] =
    useState("");
  const [georgianDisplayModePrompt, setGeorgianDisplayModePrompt] =
    useState(null);
  const [onboardingStep, setOnboardingStep] = useState("intro");
  const [loginLink, setLoginLink] = useState("");
  const [loginToken, setLoginToken] = useState("");
  const [isKeyboardOpen, setIsKeyboardOpen] = useState(false);
  const pollRef = useRef(null);
  const stageRef = useRef(null);
  const alphabetAudioRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const speakingChunksRef = useRef([]);
  const noticeTimerRef = useRef(null);
  const achievementTimerRef = useRef(null);
  const knownAchievementKeysRef = useRef(null);
  const deferredSearch = useDeferredValue(search);

  const webApp = window.Telegram?.WebApp;
  const needsStudiedLanguageSelection =
    auth.authenticated && auth.user && !auth.user.has_selected_studied_language;
  const needsOnboarding =
    auth.authenticated &&
    auth.user &&
    auth.user.has_selected_studied_language &&
    !auth.user.has_completed_onboarding;
  const isMiniApp = Boolean(webApp?.initData);
  const canRecordSpeech = Boolean(
    navigator.mediaDevices?.getUserMedia && window.MediaRecorder
  );
  const activeStudiedLanguage =
    settings?.active_studied_language || auth.progress?.course_code || "en";
  const supportsIrregularPractice = activeStudiedLanguage === "en";
  const georgianDisplayMode =
    settings?.georgian_display_mode ||
    auth.user?.georgian_display_mode ||
    "both";
  const showGeorgianLatin =
    activeStudiedLanguage === "ka" && georgianDisplayMode === "both";
  const georgianDisplayModeOptions = settings?.georgian_display_mode_options ||
    auth.user?.georgian_display_mode_options || [
      { code: "both", label: "Грузинский + латиница", recommended: true },
      { code: "native", label: "Только грузинский", recommended: false },
    ];
  const monetization = settings?.monetization || {
    plans: {
      premium: {
        price: {
          monthly: { amount: "6.99", currency: "USD" },
          yearly: { amount: "39.99", currency: "USD" },
        },
      },
    },
  };
  const billing = settings?.billing || {
    premium_active: false,
    active_subscription: null,
    plans: [],
  };
  const activeLanguageLabel =
    settings?.available_studied_languages?.find(
      (item) => item.code === activeStudiedLanguage
    )?.label ||
    auth.user?.available_studied_languages?.find(
      (item) => item.code === activeStudiedLanguage
    )?.label ||
    (activeStudiedLanguage === "ka" ? "Грузинский" : "Английский");
  const temporaryPracticeFilters =
    settings || auth.user?.temporary_practice_filters || {};
  const listeningTemporarilyDisabled = Boolean(
    temporaryPracticeFilters.listening_temporarily_disabled
  );
  const speakingTemporarilyDisabled = Boolean(
    temporaryPracticeFilters.speaking_temporarily_disabled
  );
  const listeningPauseLabel = formatPauseRemaining(
    temporaryPracticeFilters.listening_paused_until
  );
  const speakingPauseLabel = formatPauseRemaining(
    temporaryPracticeFilters.speaking_paused_until
  );

  useEffect(() => {
    const handleWindowError = (event) => {
      void reportClientError({
        category: "frontend",
        level: "error",
        message: event.message || "Unhandled window error",
        url: window.location.pathname,
        detail:
          event.error?.stack ||
          `${event.filename || ""}:${event.lineno || ""}:${event.colno || ""}`,
        meta: { source: "window.error" },
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
        meta: { source: "unhandledrejection" },
      });
    };

    window.addEventListener("error", handleWindowError);
    window.addEventListener("unhandledrejection", handleUnhandledRejection);
    return () => {
      window.removeEventListener("error", handleWindowError);
      window.removeEventListener(
        "unhandledrejection",
        handleUnhandledRejection
      );
    };
  }, []);

  useEffect(
    () => () => {
      if (alphabetAudioRef.current) {
        alphabetAudioRef.current.pause();
        alphabetAudioRef.current = null;
      }
      if (noticeTimerRef.current) {
        window.clearTimeout(noticeTimerRef.current);
      }
    },
    []
  );

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
      if (
        !(target instanceof HTMLElement) ||
        !target.matches("input, textarea")
      ) {
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

  const progressStats = useMemo(() => {
    const progress = dashboard?.progress || auth.progress;
    return [
      { label: "📚 Словарь", value: progress?.total ?? 0 },
      { label: "✅ Выучено", value: progress?.learned ?? 0 },
      { label: "🔄 В процессе", value: progress?.learning ?? 0 },
      { label: "Дней подряд", value: progress?.streak_days ?? 0 },
    ];
  }, [auth.progress, dashboard]);

  const todayStreakStat = useMemo(
    () => progressStats[3] || null,
    [progressStats]
  );
  const progressTopStats = useMemo(
    () => progressStats.slice(0, 3),
    [progressStats]
  );
  const progressSecondaryStats = useMemo(
    () => progressStats.slice(3),
    [progressStats]
  );
  const todayAchievements = useMemo(() => {
    const list =
      dashboard?.progress?.achievements || auth.progress?.achievements || [];
    return list.slice(-3);
  }, [auth.progress, dashboard]);
  const hasMoreAchievements =
    (dashboard?.progress?.achievements || auth.progress?.achievements || [])
      .length > todayAchievements.length;
  const currentProgress = dashboard?.progress || auth.progress;

  const currentTitle = useMemo(() => {
    if (primaryTab === "today") return "Сегодня";
    if (primaryTab === "learn") return "Практика";
    if (primaryTab === "words") {
      if (showLibraryAdd) return "Добавить";
      if (libraryMode === "cards") return "Карточки";
      if (libraryMode === "packs") return "Наборы";
      return "Список";
    }
    if (primaryTab === "progress") return "Прогресс";
    if (primaryTab === "more") return "Настройки";
    return "Практика";
  }, [primaryTab, libraryMode, showLibraryAdd]);

  const currentCard = cardQueue[cardIndex];
  const prepareAudio = useCallback(async (endpoint, payload) => {
    for (const delay of [0, 600, 1200, 2400]) {
      if (delay) {
        await waitForAudioPreparation(delay);
      }
      const response = await api(endpoint, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      if (response.ready) {
        setAudioRevision((revision) => revision + 1);
        return true;
      }
    }
    return false;
  }, []);

  useEffect(() => {
    if (currentCard?.id) {
      prepareAudio(`/api/audio/${currentCard.id}/prepare`, {}).catch(() => {});
    }
  }, [currentCard?.id, prepareAudio]);

  useEffect(() => {
    if (learnQuestion?.item?.id) {
      prepareAudio(`/api/audio/${learnQuestion.item.id}/prepare`, {}).catch(
        () => {}
      );
    }
  }, [learnQuestion?.item?.id, prepareAudio]);

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
  }, [
    auth.authenticated,
    primaryTab,
    libraryMode,
    learnPanel,
    showLibraryAdd,
    addDraftStep,
  ]);
  const showHeaderBack =
    primaryTab === "learn" &&
    ((learnPanel === "irregular" && irregularMode === "review") ||
      (learnPanel === "alphabet" && alphabetMode === "review"));
  const showHeaderClose =
    (primaryTab === "learn" && Boolean(learnQuestion)) ||
    (primaryTab === "learn" &&
      learnPanel === "irregular" &&
      irregularMode === "test") ||
    (primaryTab === "learn" &&
      learnPanel === "alphabet" &&
      alphabetMode === "test");

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

  function handleActionError(error) {
    if (error?.code?.startsWith("paywall_")) {
      setPrimaryTab("more");
      setNotice(error.message, { sticky: true, error: true, scope: "global" });
      return;
    }
    setNotice(error.message, { error: true });
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
    if (
      courseCode !== "ka" ||
      !showGeorgianLatin ||
      !hasGeorgianScript(value)
    ) {
      return value;
    }
    return value.replace(
      /[\u10A0-\u10FF]+(?:\s+[\u10A0-\u10FF]+)*/g,
      (match) => `${match}/${transliterateGeorgian(match)}`
    );
  }

  useEffect(() => {
    const draftIds = [
      ...(addDraft?.id ? [addDraft.id] : []),
      ...addDrafts.map((item) => item.id),
    ];
    if (!draftIds.length) {
      return;
    }

    const shouldPoll =
      Boolean(addDraft?.image_generation_in_progress) ||
      addDrafts.some((item) => item.image_generation_in_progress);
    if (!shouldPoll) {
      return;
    }

    const intervalId = window.setInterval(async () => {
      try {
        const responses = await Promise.all(
          draftIds.map((draftId) => api(`/api/words/draft/${draftId}`))
        );
        const byId = new Map(
          responses.map((entry) => [entry.draft.id, entry.draft])
        );
        if (addDraft?.id) {
          setAddDraft((current) =>
            current ? byId.get(current.id) || current : current
          );
        }
        if (addDrafts.length) {
          setAddDrafts((current) =>
            current.map((item) => byId.get(item.id) || item)
          );
        }
      } catch {
        // best-effort polling only
      }
    }, 2000);

    return () => window.clearInterval(intervalId);
  }, [addDraft, addDrafts]);

  useEffect(() => {
    let cancelled = false;
    const draftsToSync = [...(addDraft ? [addDraft] : []), ...addDrafts].filter(
      (draft, index, all) =>
        draft?.has_image &&
        all.findIndex((item) => item.id === draft.id) === index
    );

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
    const [cfg, me] = await Promise.all([
      api("/api/app-config"),
      api("/api/auth/me"),
    ]);
    setConfig(cfg);
    setAuth({
      loading: false,
      authenticated: me.authenticated,
      user: me.user,
      progress: me.progress,
    });
  }

  const refreshProgress = useCallback(async () => {
    const me = await api("/api/auth/me", { cache: "no-store" });
    if (!me.authenticated) {
      return;
    }
    setAuth((current) => ({
      ...current,
      user: me.user || current.user,
      progress: me.progress || current.progress,
    }));
    setDashboard((current) =>
      current
        ? {
            ...current,
            user: me.user || current.user,
            progress: me.progress || current.progress,
          }
        : current
    );
  }, []);

  async function loadDashboard() {
    const [
      dashboardData,
      settingsData,
      wordsData,
      irregularData,
      alphabetData,
    ] = await Promise.all([
      api("/api/dashboard"),
      api("/api/settings"),
      api(
        `/api/words?status=${statusFilter}&search=${encodeURIComponent(deferredSearch)}`
      ),
      api(`/api/irregular/list?page=${irregularPage}`),
      api(`/api/alphabet/list?page=${alphabetPage}`),
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

  async function startPremiumCheckout(billingPeriod, source = "miniapp") {
    try {
      setCheckoutBusyPeriod(billingPeriod);
      const data = await api("/api/billing/checkout", {
        method: "POST",
        body: JSON.stringify({
          plan_code: "premium",
          billing_period: billingPeriod,
          source,
        }),
      });

      if (window.Telegram?.WebApp?.openInvoice) {
        await new Promise((resolve) => {
          window.Telegram.WebApp.openInvoice(
            data.invoice_link,
            async (status) => {
              if (status === "paid") {
                await loadDashboard();
                setNotice("Premium активирован.");
              } else if (status === "cancelled") {
                setNotice("Оплата отменена.");
              } else if (status === "failed") {
                setNotice("Оплата не прошла.");
              }
              resolve();
            }
          );
        });
        return;
      }

      window.location.href = data.invoice_link;
    } catch (error) {
      handleActionError(error);
    } finally {
      setCheckoutBusyPeriod("");
    }
  }

  async function loadCards(options = {}) {
    const { reset = true } = options;
    const data = await api("/api/study/cards?scope=all");
    setCardQueue(data.items);
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
      const nextIndex = data.items.findIndex(
        (item) => item.id === currentItemId
      );
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
    const data = await api("/api/learn/question", {
      method: "POST",
      body: JSON.stringify({ exclude_ids: ids }),
    });
    setLearnSessionLimit(data.session_limit || 12);
    setLearnSelection("");
    setLearnTextAnswer("");
    setLearnResult(null);
    setIsRecording(false);
    if (
      questionCount >= (data.session_limit || 12) ||
      (data.empty && questionCount > 0)
    ) {
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

  useEffect(
    () => () => {
      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state !== "inactive"
      ) {
        mediaRecorderRef.current.stop();
      }
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      }
    },
    []
  );

  useEffect(() => {
    return () => {
      if (achievementTimerRef.current) {
        window.clearTimeout(achievementTimerRef.current);
        achievementTimerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    bootstrap().catch((error) => {
      setNotice(error.message);
      setAuth({
        loading: false,
        authenticated: false,
        user: null,
        progress: null,
      });
    });
    return () => stopPolling();
    // App bootstrap is intentionally a once-per-shell request.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!auth.authenticated) {
      return undefined;
    }

    const refreshWhenVisible = () => {
      if (document.visibilityState === "visible") {
        refreshProgress().catch(() => {});
      }
    };

    document.addEventListener("visibilitychange", refreshWhenVisible);
    window.addEventListener("pageshow", refreshWhenVisible);
    return () => {
      document.removeEventListener("visibilitychange", refreshWhenVisible);
      window.removeEventListener("pageshow", refreshWhenVisible);
    };
  }, [auth.authenticated, refreshProgress]);

  useEffect(() => {
    if (!isMiniApp || auth.authenticated || !webApp?.initData) {
      return;
    }
    webApp.ready();
    webApp.expand();
    api("/api/auth/telegram/webapp", {
      method: "POST",
      body: JSON.stringify({ init_data: webApp.initData }),
    })
      .then((data) => {
        setAuth({
          loading: false,
          authenticated: true,
          user: data.user,
          progress: data.progress,
        });
        setNotice("");
      })
      .catch((error) => {
        setNotice(error.message);
        setAuth((current) => ({ ...current, loading: false }));
      });
    // Auth is re-run only when identity inputs change, not when notice scope changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [auth.authenticated, isMiniApp, webApp]);

  useEffect(() => {
    if (!auth.authenticated) {
      knownAchievementKeysRef.current = null;
      setAchievementToast(null);
      setAchievementQueue([]);
      return;
    }
    Promise.all([loadDashboard(), loadStudyCardsOnly(), loadPacks()]).catch(
      (error) => setNotice(error.message)
    );
    // These callbacks intentionally refresh only for the explicit data filters below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    auth.authenticated,
    deferredSearch,
    statusFilter,
    irregularPage,
    alphabetPage,
  ]);

  useEffect(() => {
    if (!auth.authenticated || !currentProgress) {
      return;
    }
    const currentKeys = (currentProgress.achievements || []).map((item) =>
      makeAchievementKey(item, currentProgress.course_code)
    );
    if (!knownAchievementKeysRef.current) {
      knownAchievementKeysRef.current = currentKeys;
      return;
    }
    const previous = new Set(knownAchievementKeysRef.current);
    const newAchievements = (currentProgress.achievements || []).filter(
      (item) =>
        !previous.has(makeAchievementKey(item, currentProgress.course_code))
    );
    knownAchievementKeysRef.current = currentKeys;
    if (!newAchievements.length) {
      return;
    }
    setAchievementQueue((current) => [
      ...current,
      ...newAchievements.map((item) => ({
        key: makeAchievementKey(item, currentProgress.course_code),
        text: item,
      })),
    ]);
  }, [auth.authenticated, currentProgress]);

  useEffect(() => {
    if (achievementToast || !achievementQueue.length) {
      return;
    }
    const [nextToast, ...rest] = achievementQueue;
    setAchievementQueue(rest);
    setAchievementToast(nextToast);
    if (achievementTimerRef.current) {
      window.clearTimeout(achievementTimerRef.current);
    }
    achievementTimerRef.current = window.setTimeout(() => {
      setAchievementToast(null);
      achievementTimerRef.current = null;
    }, 3200);
  }, [achievementQueue, achievementToast]);

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
    const hasPendingWordImages =
      words.some((item) => item.image_generation_in_progress) ||
      cardQueue.some((item) => item.image_generation_in_progress);
    if (!hasPendingWordImages) {
      return;
    }
    const intervalId = window.setInterval(() => {
      void Promise.all([loadDashboard(), loadCards({ reset: false })]);
    }, 4000);
    return () => window.clearInterval(intervalId);
    // Polling lifecycle is governed by the pending-image collections.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [auth.authenticated, words, cardQueue]);

  useEffect(() => {
    if (!auth.authenticated || (!showLibraryAdd && libraryMode !== "packs")) {
      return;
    }
    setSelectedPackId("");
    setSelectedPackLevelId("");
    setSelectedPackWords({});
    setIsPackExpanded(false);
    void loadPacks();
    void preparePacksInBackground();
    // Pack reloads are deliberately tied to the app-shell navigation state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [auth.authenticated, showLibraryAdd, libraryMode, activeStudiedLanguage]);

  useEffect(() => {
    if (!needsOnboarding) {
      setOnboardingStep("intro");
    }
  }, [needsOnboarding, activeStudiedLanguage]);

  useEffect(() => {
    if (!loginToken || auth.authenticated) {
      return;
    }
    stopPolling();
    pollRef.current = window.setInterval(async () => {
      try {
        const data = await api(`/api/auth/telegram/poll/${loginToken}`, {
          method: "POST",
          body: JSON.stringify({}),
        });
        if (data.authenticated) {
          stopPolling();
          setAuth({
            loading: false,
            authenticated: true,
            user: data.user,
            progress: data.progress,
          });
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
    // Login polling is scoped to the token and authenticated state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
        body: JSON.stringify({}),
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
      handleActionError(error);
    } finally {
      setBusy(false);
    }
  }

  async function logoutWeb() {
    setBusy(true);
    try {
      await api("/api/auth/logout", {
        method: "POST",
        body: JSON.stringify({}),
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
      setAuth({
        loading: false,
        authenticated: false,
        user: null,
        progress: null,
      });
      setNotice("Выход выполнен.");
    } catch (error) {
      handleActionError(error);
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
    setPacksLoading(true);
    try {
      const data = await api("/api/packs");
      const nextPacks = data.packs || [];
      setPacks(nextPacks);
      if (!nextPacks.length) {
        return;
      }
      const nextPack =
        nextPacks.find((pack) => pack.id === selectedPackId) || nextPacks[0];
      const nextLevel =
        nextPack.levels.find((level) => level.id === selectedPackLevelId) ||
        nextPack.levels[0];
      setSelectedPackId(nextPack.id);
      setSelectedPackLevelId(nextLevel?.id || "");
    } finally {
      setPacksLoading(false);
    }
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
      setNotice(
        `За один раз можно добавить максимум ${MAX_ADD_BATCH_WORDS} слов или фраз.`
      );
      return;
    }
    setAddBusy(true);
    setAddBusyLabel("Понимаем слово...");
    try {
      const data = await api("/api/words/draft", {
        method: "POST",
        body: JSON.stringify({ text: addText }),
      });
      if (data.batch_review) {
        setAddDrafts(data.drafts);
        setAddDraftStep("batch_review");
        setBatchTranslations(
          Object.fromEntries(
            data.drafts.map((draft) => [draft.id, draft.translation || ""])
          )
        );
        setNotice(
          `Проверь ${data.drafts.length} слов. Фото загрузятся автоматически, можно не ждать.`
        );
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
      setNotice(
        data.step === "confirm_translation"
          ? "Подтверди перевод. Фото загрузится автоматически."
          : "Фото загружается автоматически. Можно сохранить слово сразу."
      );
    } catch (error) {
      handleActionError(error);
    } finally {
      setAddBusy(false);
      setAddBusyLabel("");
    }
  }

  async function preparePacksInBackground() {
    try {
      await api("/api/packs/prepare", {
        method: "POST",
        body: JSON.stringify({}),
      });
    } catch {
      // best effort only
    }
  }

  async function addSelectedPack() {
    const selectedPack =
      packs.find((pack) => pack.id === selectedPackId) || packs[0] || null;
    const selectedLevel =
      selectedPack?.levels.find((level) => level.id === selectedPackLevelId) ||
      selectedPack?.levels?.[0] ||
      null;
    const selected = selectedLevel
      ? selectedLevel.items
          .filter(
            (item) =>
              selectedPackWords[item.normalized_word] ?? !item.already_added
          )
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
      handleActionError(error);
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
        body: JSON.stringify({ translation: addTranslationInput.trim() }),
      });
      setAddDraft(data.draft);
      setAddDraftStep(data.step);
      setNotice(
        "Фото загружается автоматически. Можно сохранить слово и не ждать."
      );
    } catch (error) {
      handleActionError(error);
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
    setAddBusyLabel(
      addDraft.has_image ? "Готовим другую картинку..." : "Генерируем фото..."
    );
    try {
      const data = await api(
        `/api/words/draft/${addDraft.id}/image/regenerate`,
        {
          method: "POST",
          body: JSON.stringify({}),
        }
      );
      const version = `${data.draft.updated_at}-${Date.now()}`;
      await preloadDraftImage(addDraft.id, version);
      setAddDraft(data.draft);
      setDraftImageVersions((current) => ({
        ...current,
        [addDraft.id]: version,
      }));
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
        body: JSON.stringify({ use_image: useImage }),
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
    const draftIds = [addDraft?.id, ...addDrafts.map((item) => item.id)].filter(
      Boolean
    );
    for (const draftId of draftIds) {
      try {
        await api(`/api/words/draft/${draftId}`, {
          method: "DELETE",
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
    setAddBusyLabel(
      currentDraft?.has_image
        ? "Готовим другую картинку..."
        : "Генерируем фото..."
    );
    try {
      const data = await api(`/api/words/draft/${draftId}/image/regenerate`, {
        method: "POST",
        body: JSON.stringify({}),
      });
      const version = `${data.draft.updated_at}-${Date.now()}`;
      await preloadDraftImage(draftId, version);
      setAddDrafts((current) =>
        current.map((item) => (item.id === draftId ? data.draft : item))
      );
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
        const translation = (
          batchTranslations[draft.id] ||
          draft.translation ||
          ""
        ).trim();
        if (!translation) {
          throw new Error(`Заполни перевод для ${draft.word}.`);
        }
        let currentDraft = draft;
        if (translation !== (draft.translation || "").trim()) {
          const confirmed = await api(
            `/api/words/draft/${draft.id}/translation`,
            {
              method: "POST",
              body: JSON.stringify({ translation }),
            }
          );
          currentDraft = confirmed.draft;
        }
        await api(`/api/words/draft/${currentDraft.id}/save`, {
          method: "POST",
          body: JSON.stringify({ use_image: true }),
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
    if (
      question.exercise_type === "practice_en_ru" ||
      question.exercise_type === "listening_translate"
    ) {
      return question.item.translation;
    }
    return question.item.word;
  }

  function revealLearnAnswer() {
    const correctAnswer = getLearnExpectedAnswer();
    if (!correctAnswer || !learnQuestion) {
      return;
    }
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      mediaRecorderRef.current.onstop = null;
      mediaRecorderRef.current.stop();
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    setIsRecording(false);
    setLearnSelection("");
    setLearnResult({
      correct: false,
      correct_answer: correctAnswer,
      skipped: true,
      exercise_type: learnQuestion.exercise_type,
    });
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
          answer,
          question_id: learnQuestion.question_id,
        }),
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
          answer: learnTextAnswer,
          question_id: learnQuestion.question_id,
        }),
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
      formData.append("question_id", learnQuestion.question_id);
      formData.append(
        "audio",
        new File([blob], `speech.${extension}`, {
          type: blob.type || "audio/webm",
        })
      );
      const data = await api("/api/speaking/answer", {
        method: "POST",
        body: formData,
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
        const blob = new Blob(speakingChunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });
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
    if (
      !mediaRecorderRef.current ||
      mediaRecorderRef.current.state === "inactive"
    ) {
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
          question_id: irregularQuestion.question_id,
          answer,
        }),
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
          question_token: alphabetQuestion.question_token,
          answer,
        }),
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
      const ready = await prepareAudio("/api/alphabet/audio/prepare", {
        symbol,
      });
      if (!ready) {
        setNotice(
          "Аудио ещё готовится. Попробуй снова через несколько секунд."
        );
        return;
      }
      const audio = new Audio(
        `/api/alphabet/audio?symbol=${encodeURIComponent(symbol)}&v=${audioRevision}`
      );
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
    } catch {
      setAlphabetAudioLoadingSymbol("");
      setNotice("Не удалось воспроизвести аудио буквы.");
    }
  }

  async function skipAlphabetQuestion() {
    if (!alphabetQuestion || alphabetResult) {
      return;
    }
    try {
      const data = await api("/api/alphabet/answer", {
        method: "POST",
        body: JSON.stringify({
          question_token: alphabetQuestion.question_token,
          answer: "",
        }),
      });
      setAlphabetResult({ ...data, skipped: true });
    } catch (error) {
      setNotice(error.message);
    }
  }

  async function skipIrregularQuestion() {
    if (!irregularQuestion || irregularResult) {
      return;
    }
    try {
      const data = await api("/api/irregular/answer", {
        method: "POST",
        body: JSON.stringify({
          question_id: irregularQuestion.question_id,
          answer: "",
        }),
      });
      setIrregularResult({ ...data, skipped: true });
    } catch (error) {
      setNotice(error.message);
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
        body: JSON.stringify({ translation }),
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
      await api(`/api/words/${wordId}`, {
        method: "DELETE",
        body: JSON.stringify({}),
      });
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
        body: JSON.stringify({}),
      });
      setWords((current) =>
        current.map((item) => (item.id === wordId ? data.item : item))
      );
      setCardQueue((current) =>
        current.map((item) => (item.id === wordId ? data.item : item))
      );
      setPreviewWordId(wordId);
      setNotice(
        "Генерируем новое фото. Можно не ждать: оно появится автоматически."
      );
    } catch (error) {
      setNotice(error.message);
    } finally {
      setRegeneratingWordId(null);
    }
  }

  async function saveSettings(event) {
    event.preventDefault();
    if (
      settings?.active_studied_language === "ka" &&
      !settings?.has_selected_georgian_display_mode
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
        body: JSON.stringify(settings),
      });
      setNotice("Настройки сохранены.");
      await Promise.all([loadDashboard(), loadLearningData()]);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function setTemporaryPracticePause(kind, enabled) {
    const payload =
      kind === "listening"
        ? { pause_listening_for_minutes: enabled ? 15 : 0 }
        : { pause_speaking_for_minutes: enabled ? 15 : 0 };
    setBusy(true);
    try {
      await api("/api/settings", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      await Promise.all([loadDashboard(), loadLearningData()]);
      setNotice(
        kind === "listening"
          ? enabled
            ? "Следующие 15 минут без аудирования."
            : "Аудирование снова включено."
          : enabled
            ? "Следующие 15 минут без говорения."
            : "Говорение снова включено."
      );
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function uploadAvatar(file) {
    setUploadingAvatar(true);
    try {
      const formData = new FormData();
      formData.append("avatar", file);
      const data = await api("/api/profile/avatar", {
        method: "POST",
        body: formData,
      });
      const nextUser = data.user ? withFreshAvatarUrl(data.user) : data.user;
      const nextSettings = data.settings
        ? {
            ...data.settings,
            avatar_url: data.settings.avatar_url
              ? withFreshAvatarUrl({ avatar_url: data.settings.avatar_url })
                  .avatar_url
              : data.settings.avatar_url,
          }
        : data.settings;
      if (nextUser?.avatar_url) {
        await preloadImage(nextUser.avatar_url);
      }
      setAuth((current) => ({
        ...current,
        user: nextUser || current.user,
      }));
      setSettings((current) => ({
        ...(current || {}),
        ...(nextSettings || {}),
      }));
      setNotice("Аватар обновлён.");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setUploadingAvatar(false);
    }
  }

  async function deleteAvatar() {
    setUploadingAvatar(true);
    try {
      const data = await api("/api/profile/avatar", {
        method: "DELETE",
        body: JSON.stringify({}),
      });
      setAuth((current) => ({
        ...current,
        user: data.user || current.user,
      }));
      setSettings((current) => ({
        ...(current || {}),
        ...(data.settings || {}),
      }));
      setNotice("Аватар удалён.");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setUploadingAvatar(false);
    }
  }

  async function selectStudiedLanguage(courseCode, options = {}) {
    const hasSelectedGeorgianDisplayMode =
      settings?.has_selected_georgian_display_mode ??
      auth.user?.has_selected_georgian_display_mode ??
      false;
    if (
      courseCode === "ka" &&
      !hasSelectedGeorgianDisplayMode &&
      !options.georgianDisplayMode
    ) {
      setGeorgianDisplayModePrompt({
        source: options.source || "onboarding",
        previousCourseCode:
          settings?.active_studied_language ||
          auth.user?.active_studied_language ||
          "en",
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
      setNotice(
        "Режим отображения сохранится после нажатия «Сохранить настройки»."
      );
      return;
    }
    await selectStudiedLanguage("ka", {
      georgianDisplayMode: mode,
      source: georgianDisplayModePrompt.source,
    });
  }

  async function completeOnboarding(options = {}) {
    setBusy(true);
    try {
      await api("/api/settings", {
        method: "POST",
        body: JSON.stringify({ has_completed_onboarding: true }),
      });
      setAuth((current) => ({
        ...current,
        user: current.user
          ? { ...current.user, has_completed_onboarding: true }
          : current.user,
      }));
      setSettings((current) =>
        current ? { ...current, has_completed_onboarding: true } : current
      );
      setOnboardingStep("intro");
      if (options.openPacks) {
        openPacks();
      } else {
        setPrimaryTab("today");
      }
      setNotice("Готово. Можно начинать.");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
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

  function openPacks() {
    resetAddFlow();
    startTransition(() => {
      setPrimaryTab("words");
      setLibraryMode("packs");
      setShowLibraryAdd(false);
    });
  }

  function closeLearnSession() {
    const hasProgress = Boolean(
      learnQuestion || learnResult || learnQuestionCount > 0 || learnSessionDone
    );
    if (
      hasProgress &&
      !window.confirm(
        "Закрыть практику? Текущий прогресс в этой сессии сбросится."
      )
    ) {
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
    if (
      hasProgress &&
      !window.confirm(
        "Закрыть тест по глаголам? Текущий прогресс в этом экране сбросится."
      )
    ) {
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
      <CardsScreen
        audioRevision={audioRevision}
        cardIndex={cardIndex}
        cardQueue={cardQueue}
        cardReveal={cardReveal}
        currentCard={currentCard}
        formatDisplayLine={formatDisplayLine}
        onNext={showNextCard}
        onOpenPacks={openPacks}
        onPrevious={showPreviousCard}
        onReveal={() => setCardReveal(true)}
      />
    );
  }

  function renderLearn() {
    const hasWordsToLearn = (auth.progress?.learning ?? 0) > 0;
    const hasActiveIrregularTest =
      supportsIrregularPractice &&
      learnPanel === "irregular" &&
      irregularMode === "test";
    const hasActiveAlphabetTest =
      learnPanel === "alphabet" && alphabetMode === "test";
    const showLearnOverview =
      !["irregular", "alphabet"].includes(learnPanel) &&
      !learnQuestion &&
      !hasActiveIrregularTest &&
      !hasActiveAlphabetTest;

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
                <strong>Сессия завершена.</strong> Верно: {learnCorrectCount} из{" "}
                {learnQuestionCount || learnSessionLimit}.
              </div>
            ) : null}
            <div className="button-row">
              {hasWordsToLearn ? (
                <button
                  className="primary-button"
                  type="button"
                  onClick={() => void loadLearningData()}
                >
                  {learnSessionDone ? "Начать новую сессию" : "Начать сессию"}
                </button>
              ) : null}
              <button
                className="secondary-button"
                type="button"
                onClick={openPacks}
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
          <IrregularPracticeScreen
            correctCount={irregularCorrectCount}
            formatPointsLabel={formatPointsLabel}
            getSessionPraise={getSessionPraise}
            list={irregularList}
            mode={irregularMode}
            onAdvance={() => void advanceIrregularTest()}
            onAnswer={(answer) => void handleIrregularAnswer(answer)}
            onNextPage={() => setIrregularPage((value) => value + 1)}
            onPreviousPage={() =>
              setIrregularPage((value) => Math.max(0, value - 1))
            }
            onSkip={skipIrregularQuestion}
            onStartNewSession={() => void startIrregularTest()}
            question={irregularQuestion}
            questionCount={irregularQuestionCount}
            result={irregularResult}
            sessionDone={irregularSessionDone}
            sessionLimit={irregularSessionLimit}
          />
        </div>
      );
    }

    if (learnPanel === "alphabet") {
      return (
        <div className="screen-stack">
          <AlphabetPracticeScreen
            audioLoadingSymbol={alphabetAudioLoadingSymbol}
            correctCount={alphabetCorrectCount}
            formatDisplayAnswer={formatDisplayAnswer}
            formatPointsLabel={formatPointsLabel}
            getSessionPraise={getSessionPraise}
            list={alphabetList}
            mode={alphabetMode}
            onAdvance={() => void advanceAlphabetTest()}
            onAnswer={(answer) => void handleAlphabetAnswer(answer)}
            onNextPage={() => setAlphabetPage((value) => value + 1)}
            onPlayAudio={(symbol) => void playAlphabetAudio(symbol)}
            onPreviousPage={() =>
              setAlphabetPage((value) => Math.max(0, value - 1))
            }
            onSkip={skipAlphabetQuestion}
            onStartNewSession={() => void startAlphabetTest()}
            question={alphabetQuestion}
            questionCount={alphabetQuestionCount}
            result={alphabetResult}
            sessionDone={alphabetSessionDone}
            sessionLimit={alphabetSessionLimit}
          />
        </div>
      );
    }

    const statusClass =
      learnResult?.status === "correct"
        ? "result-box good"
        : learnResult?.status === "close"
          ? "result-box"
          : "result-box bad";

    const isChoice = learnQuestion.kind === "choice";
    const isListening = learnQuestion.kind === "listening";
    const isSpeaking = learnQuestion.kind === "speaking";
    const promptTitle = isChoice
      ? learnQuestion.exercise_type === "practice_ru_en"
        ? learnQuestion.item.translation
        : learnQuestion.item.word
      : isListening
        ? learnQuestion.exercise_type === "listening_word"
          ? "Введите слово 🎧"
          : "Введите перевод 🎧"
        : learnQuestion.item.word;
    const promptTitleDisplay = formatDisplayAnswer(
      promptTitle,
      learnQuestion.item?.course_code
    );
    const learnCorrectAnswerText = formatDisplayAnswer(
      formatLearnCorrectAnswer(learnQuestion, learnResult),
      learnQuestion.item?.course_code
    );
    const learnPointsEarned = learnResult?.points_earned || 0;
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
            <span className="status-tag">
              {learnQuestionCount + 1} / {learnSessionLimit}
            </span>
          </div>
          <div className="prompt-card">
            <strong>{promptTitleDisplay}</strong>
            {isSpeaking && learnQuestion.item?.transcription ? (
              <p className="transcription">
                /{learnQuestion.item.transcription}/
              </p>
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
                    {formatDisplayAnswer(
                      option,
                      learnQuestion.item?.course_code
                    )}
                  </button>
                ))}
              </div>
              <button
                className="secondary-button"
                type="button"
                onClick={revealLearnAnswer}
                disabled={Boolean(learnResult)}
              >
                Пропустить
              </button>
            </div>
          ) : null}
          {isListening ? (
            <form
              className="stack-form quiz-panel-tight"
              onSubmit={handleLearnListeningSubmit}
            >
              <audio
                controls
                src={`/api/audio/${learnQuestion.item.id}?v=${audioRevision}`}
                className="audio-player"
              />
              <div className="button-row practice-filter-row">
                <button
                  className={
                    listeningTemporarilyDisabled
                      ? "secondary-button active-toggle-button"
                      : "secondary-button"
                  }
                  type="button"
                  onClick={() =>
                    void setTemporaryPracticePause(
                      "listening",
                      !listeningTemporarilyDisabled
                    )
                  }
                >
                  {listeningTemporarilyDisabled
                    ? `Слушать снова${listeningPauseLabel ? ` · ${listeningPauseLabel}` : ""}`
                    : "Не могу слушать · 15 мин"}
                </button>
              </div>
              <input
                value={learnTextAnswer}
                onChange={(event) => setLearnTextAnswer(event.target.value)}
                placeholder="Твой ответ"
              />
              <div className="button-row">
                <button className="primary-button" type="submit">
                  Проверить
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={revealLearnAnswer}
                  disabled={Boolean(learnResult)}
                >
                  Пропустить
                </button>
              </div>
            </form>
          ) : null}
          {isSpeaking ? (
            <div className="quiz-panel quiz-panel-tight">
              <audio
                controls
                src={`/api/audio/${learnQuestion.item.id}?v=${audioRevision}`}
                className="audio-player"
              />
              <div className="button-row practice-filter-row">
                <button
                  className={
                    speakingTemporarilyDisabled
                      ? "secondary-button active-toggle-button"
                      : "secondary-button"
                  }
                  type="button"
                  onClick={() =>
                    void setTemporaryPracticePause(
                      "speaking",
                      !speakingTemporarilyDisabled
                    )
                  }
                >
                  {speakingTemporarilyDisabled
                    ? `Говорить снова${speakingPauseLabel ? ` · ${speakingPauseLabel}` : ""}`
                    : "Не могу говорить · 15 мин"}
                </button>
              </div>
              <div className="button-row">
                <button
                  className={
                    isRecording ? "secondary-button" : "primary-button"
                  }
                  type="button"
                  onClick={
                    isRecording ? stopSpeakingRecording : startSpeakingRecording
                  }
                  disabled={busy || !canRecordSpeech}
                >
                  {isRecording ? "⏹️ Остановить запись" : "🎙️ Начать запись"}
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={revealLearnAnswer}
                  disabled={Boolean(learnResult)}
                >
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
            <div
              className={
                isSpeaking
                  ? statusClass
                  : learnResult.correct
                    ? "result-box good"
                    : "result-box bad"
              }
            >
              <div className="result-copy">
                <span>
                  {isSpeaking
                    ? learnResult.skipped
                      ? `Правильный ответ: ${formatDisplayAnswer(learnResult.correct_answer, learnQuestion.item?.course_code)}`
                      : `${learnResult.message} Транскрибация: ${learnResult.transcript || "—"}.`
                    : learnResultText}
                </span>
                {learnPointsEarned ? (
                  <span className="points-burst">
                    ✨ +{learnPointsEarned}{" "}
                    {formatPointsLabel(learnPointsEarned)}
                  </span>
                ) : null}
              </div>
              <button
                className="secondary-button"
                type="button"
                onClick={() => void advanceLearnSession()}
              >
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
          <AddWordsScreen
            activeStudiedLanguage={activeStudiedLanguage}
            addBusy={addBusy}
            addBusyLabel={addBusyLabel}
            addDraft={addDraft}
            addDraftStep={addDraftStep}
            addDrafts={addDrafts}
            addText={addText}
            addTranslationInput={addTranslationInput}
            batchTranslations={batchTranslations}
            closeAddWords={closeAddWords}
            confirmDraftTranslation={confirmDraftTranslation}
            draftImageVersions={draftImageVersions}
            formatDisplayLine={formatDisplayLine}
            handleAddWords={handleAddWords}
            openPacks={openPacks}
            regenerateBatchDraftImage={regenerateBatchDraftImage}
            regenerateDraftImage={regenerateDraftImage}
            saveBatchDrafts={saveBatchDrafts}
            saveDraft={saveDraft}
            setAddText={setAddText}
            setAddTranslationInput={setAddTranslationInput}
            setBatchTranslations={setBatchTranslations}
          />
        ) : (
          <>
            <div className="segment-wrap main-segment">
              {LIBRARY_MODES.map((item) => (
                <button
                  key={item.id}
                  className={
                    libraryMode === item.id
                      ? "segment-button active"
                      : "segment-button"
                  }
                  type="button"
                  onClick={() => setLibraryMode(item.id)}
                >
                  {item.label}
                </button>
              ))}
            </div>
            {libraryMode === "cards" ? renderCards() : null}
            {libraryMode === "words" ? (
              <WordsListScreen
                draftTranslation={draftTranslation}
                expandedWordId={expandedWordId}
                formatDisplayLine={formatDisplayLine}
                onDeleteWord={deleteWord}
                onOpenPacks={openPacks}
                onRegenerateWordImage={regenerateWordImage}
                onSaveTranslation={saveTranslation}
                previewWordId={previewWordId}
                regeneratingWordId={regeneratingWordId}
                search={search}
                settings={settings}
                setDraftTranslation={setDraftTranslation}
                setExpandedWordId={setExpandedWordId}
                setPreviewWordId={setPreviewWordId}
                setSearch={setSearch}
                setStatusFilter={setStatusFilter}
                setWordImageErrors={setWordImageErrors}
                statusFilter={statusFilter}
                wordImageErrors={wordImageErrors}
                wordImageVersions={wordImageVersions}
                words={words}
              />
            ) : null}
            {libraryMode === "packs" ? (
              <PacksScreen
                addBusy={addBusy}
                billing={billing}
                isPackExpanded={isPackExpanded}
                onAddSelectedPack={addSelectedPack}
                onOpenAddWords={openAddWords}
                onPremiumPackLocked={() =>
                  handleActionError({
                    code: "paywall_premium_pack_gate",
                    message:
                      "Этот сценарий доступен в Premium. Открой полный доступ к сценариям для переезда.",
                  })
                }
                packs={packs}
                packsLoading={packsLoading}
                selectedPackId={selectedPackId}
                selectedPackLevelId={selectedPackLevelId}
                selectedPackWords={selectedPackWords}
                setIsPackExpanded={setIsPackExpanded}
                setSelectedPackId={setSelectedPackId}
                setSelectedPackLevelId={setSelectedPackLevelId}
                setSelectedPackWords={setSelectedPackWords}
              />
            ) : null}
          </>
        )}
      </section>
    );
  }

  function renderMore() {
    return (
      <SettingsScreen
        billing={billing}
        settings={settings}
        uploadingAvatar={uploadingAvatar}
        onDeleteAvatar={() => void deleteAvatar()}
        onStartCheckout={(period) =>
          void startPremiumCheckout(period, "settings")
        }
        onSave={saveSettings}
        onUploadAvatar={(file) => void uploadAvatar(file)}
        onChange={(field, value) =>
          setSettings((current) => {
            if (
              field === "active_studied_language" &&
              value === "ka" &&
              !current.has_selected_georgian_display_mode
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
          todayStreakStat={todayStreakStat}
          todayAchievements={todayAchievements}
          hasMoreAchievements={hasMoreAchievements}
          hasWordsToLearn={(auth.progress?.learning ?? 0) > 0}
          onOpenAddWords={openPacks}
          onOpenLearn={() => openLearn("practice")}
          onOpenProgress={() => setPrimaryTab("progress")}
        />
      );
    }
    if (primaryTab === "learn") return renderLearn();
    if (primaryTab === "words") return renderLibrary();
    if (primaryTab === "progress") {
      return (
        <ProgressScreen
          progress={auth.progress}
          progressTopStats={progressTopStats}
          progressSecondaryStats={progressSecondaryStats}
        />
      );
    }
    return renderMore();
  }

  if (auth.loading) {
    return <div className="boot-screen">Loading VocabuMe...</div>;
  }

  if (!auth.authenticated) {
    return (
      <div
        className={`app-shell auth-layout${isKeyboardOpen ? " keyboard-open" : ""}`}
      >
        <main className="auth-stage">
          {notice ? <div className="notice">{notice.message}</div> : null}
          <AuthPanel
            config={config}
            onOpenLogin={requestLoginLink}
            loginLink={loginLink}
            loginPending={busy}
          />
        </main>
      </div>
    );
  }

  if (
    georgianDisplayModePrompt ||
    needsStudiedLanguageSelection ||
    needsOnboarding
  ) {
    return (
      <OnboardingGate
        activeLanguageLabel={activeLanguageLabel}
        availableLanguages={auth.user?.available_studied_languages || []}
        billing={billing}
        busy={busy}
        checkoutBusyPeriod={checkoutBusyPeriod}
        georgianDisplayModeOptions={georgianDisplayModeOptions}
        georgianDisplayModePrompt={georgianDisplayModePrompt}
        isKeyboardOpen={isKeyboardOpen}
        monetization={monetization}
        needsOnboarding={needsOnboarding}
        needsStudiedLanguageSelection={needsStudiedLanguageSelection}
        notice={notice}
        onboardingStep={onboardingStep}
        onCancelGeorgianDisplayMode={cancelGeorgianDisplayModePrompt}
        onCompleteOnboarding={(options) => void completeOnboarding(options)}
        onConfirmGeorgianDisplayMode={(mode) =>
          void confirmGeorgianDisplayMode(mode)
        }
        onSelectLanguage={(courseCode) =>
          void selectStudiedLanguage(courseCode, { source: "onboarding" })
        }
        onSetOnboardingStep={setOnboardingStep}
        onStartCheckout={(period) =>
          void startPremiumCheckout(period, "onboarding")
        }
      />
    );
  }

  return (
    <div className={`app-shell${isKeyboardOpen ? " keyboard-open" : ""}`}>
      <AppTopbar
        busy={busy}
        currentTitle={currentTitle}
        isMiniApp={isMiniApp}
        onBack={
          learnPanel === "alphabet"
            ? backFromAlphabetReview
            : backFromIrregularReview
        }
        onClose={
          learnPanel === "irregular" && irregularMode === "test"
            ? closeIrregularTest
            : learnPanel === "alphabet" && alphabetMode === "test"
              ? closeAlphabetTest
              : closeLearnSession
        }
        onLogout={logoutWeb}
        onOpenProfile={() => setPrimaryTab("more")}
        onToggleAddWords={() => {
          if (showLibraryAdd) {
            void closeAddWords();
            return;
          }
          openPacks();
        }}
        primaryTab={primaryTab}
        showHeaderBack={showHeaderBack}
        showHeaderClose={showHeaderClose}
        showLibraryAdd={showLibraryAdd}
        user={auth.user}
      />

      {notice ? <div className="notice">{notice.message}</div> : null}
      {achievementToast ? (
        <div className="achievement-toast" key={achievementToast.key}>
          <div className="achievement-toast-badge">🏆</div>
          <div className="achievement-toast-copy">
            <span className="achievement-toast-overline">Новая ачивка</span>
            <strong>{achievementToast.text}</strong>
          </div>
        </div>
      ) : null}

      <main className="app-stage" ref={stageRef}>
        <AppTopbar
          busy={busy}
          currentTitle={currentTitle}
          extraClass="desktop-scroll-topbar"
          isMiniApp={isMiniApp}
          onBack={
            learnPanel === "alphabet"
              ? backFromAlphabetReview
              : backFromIrregularReview
          }
          onClose={
            learnPanel === "irregular" && irregularMode === "test"
              ? closeIrregularTest
              : learnPanel === "alphabet" && alphabetMode === "test"
                ? closeAlphabetTest
                : closeLearnSession
          }
          onLogout={logoutWeb}
          onOpenProfile={() => setPrimaryTab("more")}
          onToggleAddWords={() => {
            if (showLibraryAdd) {
              void closeAddWords();
              return;
            }
            openPacks();
          }}
          primaryTab={primaryTab}
          showHeaderBack={showHeaderBack}
          showHeaderClose={showHeaderClose}
          showLibraryAdd={showLibraryAdd}
          user={auth.user}
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
