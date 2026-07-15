import { expect, test, type Page, type Route } from "@playwright/test";

const user = {
  id: 101,
  chat_id: 101,
  username: "e2e_user",
  has_selected_studied_language: true,
  has_completed_onboarding: true,
  active_studied_language: "en",
  available_studied_languages: [{ code: "en", label: "Английский" }],
};

const progress = {
  total: 0,
  learned: 0,
  learning: 0,
  streak_days: 0,
  studied_today: false,
  learned_today: 0,
  achievements: [],
  pending_achievement_highlights: [],
  rank_percent: 0,
  course_code: "en",
};

const settings = {
  active_studied_language: "en",
  available_studied_languages: [{ code: "en", label: "Английский" }],
  session_question_limit: 12,
  monetization: { plans: { premium: { telegram_stars_price: {} } } },
  billing: { premium_active: false, plans: [] },
};

async function mockApi(page: Page) {
  await page.route("**/api/**", async (route: Route) => {
    const { pathname } = new URL(route.request().url());
    const responseByPath: Record<string, object> = {
      "/api/app-config": { bot_username: "VocabuMe_bot", webapp_url: "http://127.0.0.1:4173" },
      "/api/auth/me": { ok: true, authenticated: true, user, progress },
      "/api/dashboard": { ok: true, user, progress },
      "/api/settings": { ok: true, settings },
      "/api/words": { ok: true, items: [] },
      "/api/irregular/list": { ok: true, items: [], page: 1, total_pages: 1 },
      "/api/alphabet/list": { ok: true, items: [], page: 1, total_pages: 1 },
      "/api/study/cards": { ok: true, items: [] },
      "/api/packs": { ok: true, packs: [] },
      "/api/packs/prepare": { ok: true },
    };
    const body = responseByPath[pathname] || { ok: true, items: [] };
    await route.fulfill({ contentType: "application/json", json: body });
  });
}

test.describe("VocabuMe mobile app shell", () => {
  test("keeps all primary surfaces reachable on a short mobile viewport", async ({ page }) => {
    await mockApi(page);
    await page.setViewportSize({ width: 412, height: 640 });
    await page.goto("/");

    await expect(page.getByRole("heading", { name: "Продолжай учить слова ✨" })).toBeVisible();

    await page.getByRole("button", { name: "Практика" }).click();
    await expect(page.getByRole("heading", { name: "Учить слова 🎯" })).toBeVisible();

    await page.getByRole("button", { name: "Словарь" }).click();
    await expect(page.getByRole("button", { name: "Карточки" })).toBeVisible();

    await page.getByRole("button", { name: "Прогресс" }).click();
    await expect(page.getByRole("heading", { name: "К чему стремиться ✨" })).toBeVisible();

    await page.getByRole("button", { name: "Ещё" }).click();
    await expect(page.getByRole("heading", { name: "Ещё" })).toBeVisible();

    const nav = page.locator(".nav-grid-bottom");
    await expect(nav).toBeVisible();
    await expect(nav.boundingBox()).resolves.toMatchObject({ y: expect.any(Number) });
  });
});
