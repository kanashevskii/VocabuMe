function getCookie(name) {
  const match = document.cookie.match(new RegExp(`(^| )${name}=([^;]+)`));
  return match ? decodeURIComponent(match[2]) : "";
}

export async function reportClientError(payload) {
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
      body: JSON.stringify(payload),
    });
  } catch {
    // best-effort logging only
  }
}

export async function api(url, options = {}) {
  const telegramInitData = window.Telegram?.WebApp?.initData || "";
  const isFormData =
    typeof FormData !== "undefined" && options.body instanceof FormData;
  const headers = {
    ...(isFormData ? {} : { "Content-Type": "application/json" }),
    ...(options.headers || {}),
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
      meta: { method },
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
    } else if (
      typeof errorMessage === "string" &&
      errorMessage.includes("<html")
    ) {
      errorMessage = `Ошибка сервера (${response.status}).`;
    }
    await reportClientError({
      category: "api",
      status_code: response.status,
      message: errorMessage.slice(0, 4000),
      url,
      detail: rawText.slice(0, 4000),
      meta: { method },
    });
    const error = new Error(errorMessage);
    error.status = response.status;
    error.code = data.code || "";
    error.payload = data;
    throw error;
  }

  return data;
}
