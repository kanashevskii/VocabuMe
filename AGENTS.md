# AGENTS.md

## Project focus

This repository is no longer "just a Telegram bot".

VocabuMe is a shared product with:
- Telegram bot
- Telegram Mini App
- website

All product changes should preserve shared progress, shared dictionary state, and Telegram-based identity across all three surfaces.

## Architecture rules

- Backend is Django.
- Frontend is React + Vite in [frontend](/Users/eduard/PycharmProjects/englishbot/frontend).
- Shared business logic should go into [vocab/services.py](/Users/eduard/PycharmProjects/englishbot/vocab/services.py) where possible.
- API and auth flows live in [vocab/views.py](/Users/eduard/PycharmProjects/englishbot/vocab/views.py).
- Telegram auth helpers live in [vocab/telegram_auth.py](/Users/eduard/PycharmProjects/englishbot/vocab/telegram_auth.py).
- Bot interactions live in [vocab/bot.py](/Users/eduard/PycharmProjects/englishbot/vocab/bot.py).

## UX expectations

- Treat the frontend as an app shell, not a landing page.
- Prioritize Telegram WebView and mobile viewport behavior first.
- Bottom navigation must remain visible and usable on small screens.
- Content must not hide behind the bottom tab bar.
- When switching tabs or internal modes, reset the visible scroll context intentionally.
- Avoid decorative text blocks that delay the learning flow.

## Auth expectations

- Telegram Mini App auth must work even when session cookies are unreliable inside Telegram WebView.
- Web auth should continue to support Telegram-based login.
- Do not introduce separate non-Telegram identity systems unless explicitly requested.

## Deployment expectations

- Production domain is `vocabume.k1prod.com`.
- Frontend assets are built from `frontend` and served from `frontend/dist`.
- Verify the live asset hash after deployment when frontend changes are made.
- If UI bugs are reported, reproduce locally first, then deploy.

## Testing expectations

- For frontend/mobile bugs, use Playwright and test real mobile-sized viewports.
- Test at least:
  - `Today`
  - `Learn`
  - `Words`
  - `Progress`
  - `More`
- When touching navigation or layout, validate both normal and shorter mobile heights.

## Documentation expectations

- Keep README aligned with the actual product state.
- Keep README aligned with the actual legal/usage terms as well.
- Do not describe VocabuMe as bot-only anymore.
- Preserve and update the rule that VocabuMe is for personal non-commercial use only unless the author has granted separate written permission for commercial use.
