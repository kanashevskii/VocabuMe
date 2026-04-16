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

## Production safety rules

- Do not probe production with many separate SSH reconnects in a short window.
- For one production investigation, prefer one SSH session or one batched SSH command over many tiny checks.
- Before changing production cron, workers, or background jobs, first identify whether they call OpenAI, Telegram, or other billed/rate-limited services.
- If a background job is suspected of burning money or rate limits, stop the spend first, then debug the root cause.
- Treat production background generation as user-latency sensitive: user-triggered flows must stay higher priority than pack warmup or backfill jobs.

## Debugging and verification rules

- Use structured debugging:
  - what is confirmed from code/logs/DB
  - what is inferred
  - which hypothesis is being tested next
- Do not claim that something is fixed, passed, or deployed without direct evidence.
- Evidence should be concrete:
  - command output
  - test result
  - build result
  - live asset hash
  - production state check
- If a safeguard should hold reliably, prefer code, tests, migrations, or automation over "remembering" the rule in prompt text.

## Dependency and supply-chain rules

- Do not change dependencies or lockfiles unless the task actually requires it.
- Avoid opportunistic package upgrades during unrelated work.
- Prefer existing pinned/locked dependencies over newly published versions.
- If adding or upgrading a dependency is necessary, explain why it is needed and verify that the change is scoped to the task.

## Red lines

- Do not delete user data, datasets, media, or production artifacts unless the user explicitly asked to delete them.
- "Filter", "clean", or "exclude" means move, mark, skip, or hide by default, not delete.
- Do not change production config values just because they look strange; first understand why the current value exists.
- Do not enable new background loops, cron jobs, or auto-generation workers without checking cost and retry behavior.

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
