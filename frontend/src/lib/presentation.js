export function getSessionPraise(correct, total) {
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
    return "💪 Неплохо. Основа уже есть, можно пройти ещё один круг.";
  }
  return "🌱 Начало положено. Следующая сессия уже будет увереннее.";
}

export function formatPointsLabel(points) {
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

export function formatPauseRemaining(untilIso) {
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

export function formatLearnCorrectAnswer(learnQuestion, learnResult) {
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

export function mergeItemsById(current, incoming) {
  if (!incoming?.length) {
    return current;
  }
  const seen = new Set(incoming.map((item) => item.id));
  return [...incoming, ...current.filter((item) => !seen.has(item.id))];
}

export function withFreshAvatarUrl(payload) {
  if (!payload?.avatar_url) {
    return payload;
  }
  const separator = payload.avatar_url.includes("?") ? "&" : "?";
  return {
    ...payload,
    avatar_url: `${payload.avatar_url}${separator}r=${Date.now()}`,
  };
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

export function transliterateGeorgian(text) {
  return Array.from(text || "")
    .map((char) => GEORGIAN_TO_LATIN[char] || char)
    .join("");
}

export function hasGeorgianScript(text) {
  return /[\u10A0-\u10FF]/.test(text || "");
}

export function makeAchievementKey(text, courseCode = "en") {
  return `${courseCode}:${text}`;
}

export function waitForAudioPreparation(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}
