# VocabuMe Implementation Plan

This file is the execution backlog for bringing VocabuMe to a senior/staff engineering baseline.

Working rule:
- Execute one task at a time.
- Always start with the highest-priority task that is not complete.
- Do not parallelize major refactors until the current priority item is merged and verified.
- Preserve shared product behavior across the Telegram bot, Telegram Mini App, and website.

## Current execution order

| Priority | Status | Task | Why first | Exit criteria |
| --- | --- | --- | --- | --- |
| P0 | `done` | Secure secrets and config loading | Unsafe configuration blocks every other production-quality step. | No hardcoded secrets remain, required env vars are documented, startup fails fast on missing required config, `.env.example` is aligned. |
| P1 | `done` | Refactor business logic from handlers/views into services | Shared logic must be centralized before test coverage and further cleanup make sense. | Telegram handlers and API flows delegate to service-layer functions, with no raw ORM or business rules embedded in handlers where avoidable. |
| P2 | `done` | Add logging and structured error handling | Refactors without observability are risky. | Console logging is configured, critical service and external-call paths log failures, user-facing fallbacks exist for expected failure modes. |
| P3 | `done` | Add backend test coverage for critical flows | Core behavior needs a safety net before broader frontend and CI changes. | Unit tests cover services and validation, integration tests cover major bot/API flows, and the critical path coverage is measurable. |
| P4 | `done` | Enforce backend and frontend code quality tooling | Tooling should stabilize code after the first architectural cleanup. | Formatting, linting, and type-checking commands exist and run cleanly in local development. |
| P5 | `done` | Refactor React app shell and shared frontend logic | Frontend cleanup should follow stabilized backend contracts and test coverage. | Large components are split, mobile navigation/layout rules hold on Telegram-sized viewports, and loading/error states are consistent. |
| P6 | `next` | Add frontend tests and mobile E2E validation | UI changes must be protected with realistic viewport coverage. | Critical tabs and navigation work in mobile-sized Playwright runs, and component-level tests cover validation and failure states. |
| P7 | `queued` | Add CI pipeline | CI is only valuable once the project has stable checks worth enforcing. | CI runs backend/frontend install, lint, tests, and frontend build on every push/PR. |
| P8 | `queued` | Docker and local developer workflow improvements | Packaging should reflect the actual verified app structure. | Docker and local run instructions are consistent with the current stack and can boot the app predictably. |
| P9 | `queued` | Performance review and query/render cleanup | Optimization should follow structural correctness and test protection. | No obvious N+1 queries or avoidable render churn remain in critical user flows. |
| P10 | `queued` | Dead code, magic values, and documentation cleanup | Final cleanup should happen after the architecture settles. | Unused code is removed, constants are named, and README/docs reflect the shipped product. |
| P11 | `queued` | UI polish and non-blocking UX improvements | Cosmetic work should stay last. | Empty states, confirmations, and minor usability issues are resolved without regressing core flows. |

## Task details

### P0. Secure secrets and config loading

Scope:
- Audit Django settings, bot startup, frontend config usage, and scripts for hardcoded secrets or implicit defaults.
- Standardize environment loading and fail-fast behavior for required secrets.
- Align `.env.example`, README, and runtime assumptions.

Deliverables:
- Centralized config access pattern.
- No committed runtime secrets.
- Clear local setup instructions for required variables.

Notes:
- Added [core/env.py](/Users/eduard/PycharmProjects/englishbot/core/env.py) as the shared environment access layer.
- Critical secrets now fail fast with explicit configuration errors instead of scattered import-time `decouple` reads.

### P1. Refactor business logic into services

Scope:
- Move reusable domain logic into [vocab/services.py](/Users/eduard/PycharmProjects/englishbot/vocab/services.py) or small adjacent service modules only if separation becomes necessary.
- Reduce logic in [vocab/bot.py](/Users/eduard/PycharmProjects/englishbot/vocab/bot.py) and [vocab/views.py](/Users/eduard/PycharmProjects/englishbot/vocab/views.py).
- Keep Telegram identity, dictionary state, and learning progress shared across all surfaces.

Deliverables:
- Service functions own validation and state changes.
- Handlers/views orchestrate request parsing and response formatting, not business rules.

Notes:
- Completed. Add-word, draft-creation, settings application, word lookup/update/delete, user/draft lookup, helper selectors, achievements, reminder updates, and bot dictionary/progress wrappers now delegate to service-layer functions.
- `vocab/views.py` and `vocab/bot.py` no longer keep direct ORM/state-change business logic where a reusable service-layer function is appropriate.

### P2. Logging and error handling

Scope:
- Standardize logging configuration.
- Add safe exception handling around external integrations, DB mutations, and user-triggered flows.
- Preserve usable fallbacks in bot and web responses.

Notes:
- Completed. Logging now goes to stdout in a consistent format and avoids duplicate handler buildup.
- Critical speaking/audio/image/reminder failure paths now log exceptions and return safe fallback responses instead of failing silently.

### P3. Backend tests

Scope:
- Add unit tests for services and validators.
- Add integration tests for main Telegram and API flows.
- Focus first on the flows touched by P0-P2.

### P4-P11

These remain queued until the earlier work is completed and verified. The order above is intentional, except where we explicitly pull tests forward to establish a deployment safety net earlier.

P4 notes:
- Completed with a pragmatic active-code scope for backend quality commands: `core/env.py`, `core/logging_config.py`, `core/settings.py`, `core/test_settings.py`, `run.py`, `vocab/services.py`, `vocab/views.py`, `vocab/openai_utils.py`, `vocab/reminders.py`, `vocab/management/commands/send_reminders.py`, and `tests/`.
- `make quality` now runs backend linting, focused mypy, frontend lint, frontend build, and backend tests.
- Frontend lint is non-blocking on existing `App.jsx` warnings for now; those warnings are deferred into P5 while the command itself stays green.

P5 notes:
- Started by extracting shared frontend constants, API client/reporting helpers, and the auth shell component out of `frontend/src/App.jsx`.
- Completed by extracting topbar, bottom navigation, today/progress/settings screens, and pruning unused review-only UI code from `frontend/src/App.jsx`.
- Verified locally with `make quality` and a Playwright mobile pass against the local Django server on `390x844` plus shorter-height checks on `390x700`.
- Mobile tab validation covered `Today`, `Learn`, `Words`, `Progress`, and `More`; bottom navigation remained visible and reachable in the checked states.

## Definition of done for each task

A task moves from `next` or `queued` to `done` only when:
- the code is implemented,
- local verification has been run,
- affected docs are updated,
- and the next priority item can start without reopening the previous one immediately.

## Update protocol

When progress is made:
1. Update the status in this file.
2. Keep exactly one task marked `next`.
3. Add short notes only for decisions that affect later tasks.

Current sequencing note:
- Backend test coverage was intentionally pulled ahead of P1/P2 so the project gains a regression safety net before larger refactors.
- After each subsequent task, run `python -m pytest -q` before considering the change ready for deploy.
- Current baseline is an initial safety net, not final target coverage.
- Follow-up required: current backend coverage baseline is working and green, but still below the desired long-term threshold. Return later to expand coverage in draft/image flows, pack preparation flows, and remaining file-serving / edge-case API branches before treating tests as a full production gate.
