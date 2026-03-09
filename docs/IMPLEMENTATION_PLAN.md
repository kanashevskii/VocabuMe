# VocabuMe Implementation Plan

This file is the execution backlog for bringing VocabuMe to a senior/staff engineering baseline.

Working rule:
- Execute one task at a time.
- Always start with the highest-priority task that is not complete.
- Do not parallelize major refactors until the current priority item is merged and verified.
- Preserve shared product behavior across the Telegram bot, Telegram Mini App, and website.

## Current execution order

This order is now optimized for the fastest path to first paid subscriptions, not for finishing the entire engineering cleanup first.

| Priority | Status | Task | Why first | Exit criteria |
| --- | --- | --- | --- | --- |
| P0 | `done` | Secure secrets and config loading | Unsafe configuration blocks every other production-quality step. | No hardcoded secrets remain, required env vars are documented, startup fails fast on missing required config, `.env.example` is aligned. |
| P1 | `done` | Refactor business logic from handlers/views into services | Shared logic must be centralized before test coverage and further cleanup make sense. | Telegram handlers and API flows delegate to service-layer functions, with no raw ORM or business rules embedded in handlers where avoidable. |
| P2 | `done` | Add logging and structured error handling | Refactors without observability are risky. | Console logging is configured, critical service and external-call paths log failures, user-facing fallbacks exist for expected failure modes. |
| P3 | `done` | Add backend test coverage for critical flows | Core behavior needs a safety net before broader frontend and CI changes. | Unit tests cover services and validation, integration tests cover major bot/API flows, and the critical path coverage is measurable. |
| P4 | `done` | Enforce backend and frontend code quality tooling | Tooling should stabilize code after the first architectural cleanup. | Formatting, linting, and type-checking commands exist and run cleanly in local development. |
| P5 | `done` | Refactor React app shell and shared frontend logic | Frontend cleanup should follow stabilized backend contracts and test coverage. | Large components are split, mobile navigation/layout rules hold on Telegram-sized viewports, and loading/error states are consistent. |
| P18 | `done` | Language expansion decision and rollout | Georgian needs to become a first-class product dimension before premium logic and paid content are layered on top. | English and Georgian are supported as explicit study tracks with the minimal required data-model, service, UI, and test changes to safely extend later. |
| P13 | `done` | Monetization model and paywall design | The fastest route to first subscriptions starts with a concrete paid offer and a clear free-vs-premium boundary, but that offer should sit on the intended language architecture. | Free vs premium limits are approved, paywall triggers are documented, and target pricing is fixed for MVP launch. |
| P22 | `next` | Georgia practical relocation scenarios | Users will not buy a vague relocation product; they need immediate concrete value around real Georgia-first tasks. | First approved scenario packs exist for `work permit`, `bank account`, and related everyday relocation tasks in English and Georgian. |
| P16 | `queued` | Onboarding and premium conversion flow | First-run experience must explain the relocation promise and route the user toward the paid offer quickly. | First-run onboarding exists, teaches the core loop, and leads cleanly into the selected premium offer. |
| P14 | `queued` | Subscription and Telegram payments | After offer and onboarding are defined, payment flow becomes the shortest path to first revenue. | Subscription models, payment flow, entitlement checks, and expiration handling work end to end. |
| P15 | `queued` | Free-tier limits and entitlement enforcement | The paywall only converts if limits are enforced consistently after payment logic exists. | Daily/plan-based limits are enforced consistently across bot, Mini App, and website. |
| P17 | `queued` | Product analytics and conversion tracking | We need to see where users convert or drop before spending more effort on scaling. | Core events, funnel checkpoints, and subscription conversion metrics are implemented and documented. |
| P20 | `queued` | Landing page and acquisition assets | Once the product can sell, it needs a focused acquisition surface aligned to the Georgia relocation wedge. | Marketing landing, screenshots, store-like copy, and CTA flow are aligned with the approved product direction. |
| P21 | `queued` | Referral and growth loops | Telegram-native sharing can lower CAC early, but only after the core product and paywall exist. | At least one approved growth loop is implemented with measurable invite/share conversion, and referral mechanics are consistent with pricing and analytics. |
| P6 | `queued` | Add frontend tests and mobile E2E validation | Useful, but the current local quality baseline is already good enough to begin the first monetization sprint. | Critical tabs and navigation work in mobile-sized Playwright runs, and component-level tests cover validation and failure states. |
| P7 | `queued` | Add CI pipeline | Important for scaling changes safely, but not the immediate bottleneck to first subscriptions. | CI runs backend/frontend install, lint, tests, and frontend build on every push/PR. |
| P8 | `queued` | Docker and local developer workflow improvements | Helpful for repeatability, but not a direct driver of first paid conversions. | Docker and local run instructions are consistent with the current stack and can boot the app predictably. |
| P9 | `queued` | Performance review and query/render cleanup | Should follow once we have real monetized traffic patterns to optimize against. | No obvious N+1 queries or avoidable render churn remain in critical user flows. |
| P10 | `queued` | Dead code, magic values, and documentation cleanup | Cleanup remains valuable, but it does not itself create revenue. | Unused code is removed, constants are named, and README/docs reflect the shipped product. |
| P11 | `queued` | UI polish and non-blocking UX improvements | Cosmetic work is still lower leverage than monetization, content, and acquisition. | Empty states, confirmations, and minor usability issues are resolved without regressing core flows. |
| P23 | `queued` | Interview preparation track (deferred) | Valuable later, but too large to block the first Georgia practical-use launch. | A separate interview-prep roadmap exists, but implementation starts only after the first practical relocation scenarios are validated. |

## Product and monetization backlog

These items start only after the current engineering queue is in an acceptable state or when we explicitly decide to switch focus.

Working rule for P12+:
- Do not start implementation directly from these items.
- Before each of these tasks, first extract unresolved product questions.
- Get explicit approval on those unresolved questions from the product owner.
- Only after agreement, move the task to `next` and implement it.

| Priority | Status | Task | Why later | Exit criteria |
| --- | --- | --- | --- | --- |
| P12 | `done` | Product positioning decision | Monetization and feature roadmap depend on a clear market wedge. | One primary positioning is chosen and documented: `English AI trainer`, `Language for relocation`, or another explicitly approved option. |
| P13 | `done` | Monetization model and paywall design | Payments and premium gates should follow a chosen product narrative, not precede it. | Free vs premium limits are approved, paywall triggers are documented, and target pricing is fixed for MVP launch. |
| P14 | `queued` | Subscription and Telegram payments | Billing should implement an approved offer, not a placeholder one. | Subscription models, payment flow, entitlement checks, and expiration handling work end to end. |
| P15 | `queued` | Free-tier limits and entitlement enforcement | Limits only make sense after the paid offer and onboarding promise are defined. | Daily/plan-based limits are enforced consistently across bot, Mini App, and website. |
| P16 | `queued` | Onboarding and premium conversion flow | Onboarding copy, tutorial, and paywall should reflect the chosen positioning and pricing. | First-run onboarding exists, teaches the core loop, and leads cleanly into the selected premium offer. |
| P17 | `queued` | Product analytics and conversion tracking | Ads and growth work are blind without event tracking. | Core events, funnel checkpoints, and subscription conversion metrics are implemented and documented. |
| P18 | `queued` | Language expansion decision and rollout | Multi-language support changes product scope, content, and UX architecture. | Target languages, audience, and data model impact are approved before implementation starts. |
| P19 | `queued` | Relocation-focused content packs and positioning assets | If relocation becomes the wedge, content and messaging must support it immediately. | Approved starter packs, copy, and UX messaging exist for the chosen relocation scenarios. |
| P20 | `queued` | Landing page and acquisition assets | Ads should point to a coherent value proposition and measurable conversion path. | Marketing landing, screenshots, store-like copy, and CTA flow are aligned with the approved product direction. |
| P21 | `queued` | Referral and growth loops | Telegram-native growth should not rely only on paid acquisition. | At least one approved growth loop is implemented with measurable invite/share conversion, and referral mechanics are consistent with pricing and analytics. |
| P22 | `queued` | Georgia practical relocation scenarios | A concrete Georgia-first content path is needed to turn the positioning into something users can immediately buy and use. | First approved scenario packs exist for `work permit`, `bank account`, and related everyday relocation tasks in English and Georgian. |
| P23 | `queued` | Interview preparation track (deferred) | Interview prep is still valuable, but it is a larger content/problem space and should not block the initial relocation wedge. | A separate interview-prep roadmap exists, but implementation starts only after the first practical relocation scenarios are validated. |

## Launch-first rationale

If the goal is first subscriptions as fast as possible, the product needs this sequence:
1. Define what exactly people are buying.
2. Give them one concrete Georgia-first relocation use case worth paying for.
3. Show that value in onboarding and paywall.
4. Accept payment.
5. Enforce premium gates.
6. Measure conversion.
7. Only then spend effort on broader acquisition, growth loops, and deeper engineering hardening.

This means the fastest revenue path is:
- `P18 -> P13 -> P22 -> P16 -> P14 -> P15 -> P17 -> P20 -> P21`

And not:
- `P6 -> P7 -> P8 -> P9 -> P10 -> P11`

because the latter sequence improves engineering quality but does not directly create an offer users can subscribe to.

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

## Product questions to resolve before P12+

### P12. Product positioning decision

Open questions to agree before work starts:
- Is VocabuMe primarily staying in `English learning`, or are we explicitly moving toward `language for relocation`?
- If we keep English, what is the wedge: `AI trainer`, `simple Telegram-first SRS`, `English for IT`, `English for beginners`, or another niche?
- If we move toward relocation, which first market do we want to own: `Georgian`, `Armenian`, `Turkish`, or a broader relocation-first bundle?
- Are we optimizing for a broad market, or for a smaller but easier-to-convert niche?

Implementation approach after agreement:
- Freeze the positioning in README and product docs.
- Align roadmap ordering, onboarding copy, premium messaging, and acquisition assets to that decision.

Decision:
- Approved direction: `Language for relocation`.
- Product should stay globally useful for relocation language learning, not just generic English study.
- Users must still be able to add arbitrary personal words and phrases outside curated relocation scenarios.

### P13. Monetization model and paywall design

Open questions to agree before work starts:
- What is the free plan exactly: daily new-word cap, AI cap, dictionary size cap, quiz cap, or a combination?
- What is the premium promise: `unlimited words`, `unlimited AI`, `AI explanations`, `AI dialogue`, `advanced stats`, or selected subsets?
- Do we want a soft launch period with relaxed limits before hard paywalls appear?
- What exact monthly and yearly prices are acceptable for the first launch?

Implementation approach after agreement:
- Define entitlement matrix.
- Define paywall triggers and copy for bot, Mini App, and website.
- Only then implement payments and enforcement.

Decision so far:
- Free plan should allow up to `10` new words/phrases per day.
- Primary daily AI-related free limit should be on extra image regeneration, not on the first automatic generation for newly added items.
- Initial pricing approved for now:
  - `$6.99 / month`
  - `$39.99 / year`
- General premium framing remains:
  - unlimited new additions
  - unlimited premium relocation packs
  - expanded AI capabilities
- Additional approved launch assumptions:
  - free users keep unlimited review, existing-card practice, reminders, alphabet practice, and irregular verbs
  - free users get up to `2` relocation starter packs before premium pack gates appear
  - first meaningful premium offer moment is a soft paywall after the first successful scenario/practice value moment
  - hard paywalls are triggered by the daily new-item cap, premium scenario pack access, and extra image regeneration cap

Implementation notes:
- Source-of-truth constants now live in [vocab/monetization.py](/Users/eduard/PycharmProjects/englishbot/vocab/monetization.py).
- Settings payload now exposes the approved monetization catalog so later onboarding/paywall/payment work can reuse one backend contract instead of duplicating pricing or limits in the frontend.

### P14. Subscription and Telegram payments

Open questions to agree before work starts:
- Are we using Telegram payments only, or do we need a web fallback too?
- What plans ship first: monthly only, yearly only, or both?
- How do we handle grace periods, renewal failure, refunds, and subscription expiration UX?
- Do we need trial access, promo codes, or launch discounts in v1?

Implementation approach after agreement:
- Add billing domain models.
- Implement payment webhook / callback handling.
- Implement entitlement checks in shared services.

### P15. Free-tier limits and entitlement enforcement

Open questions to agree before work starts:
- Which actions should remain unlimited in free: review, reminders, existing-card practice, irregular verbs?
- Should limits reset daily by UTC or by user timezone?
- Does the free limit apply equally on bot, Mini App, and website?
- What fallback UX appears when the user hits a cap?

Implementation approach after agreement:
- Add counters and quota checks in shared services.
- Expose plan state consistently to bot and frontend.
- Add tests around limits and resets.

### P16. Onboarding and premium conversion flow

Open questions to agree before work starts:
- What is the first-run promise in one sentence?
- Do we want onboarding optimized for `adding first words`, `trying first lesson`, or `understanding premium value`?
- Where should paywall appear first: after onboarding, after first success, after hitting a cap, or after a trial window?
- Do we want separate onboarding for English users and relocation-language users?

Implementation approach after agreement:
- Design first-session path.
- Add tutorial state.
- Add paywall entry points tied to approved funnel moments.

### P17. Product analytics and conversion tracking

Open questions to agree before work starts:
- What analytics stack do we trust for first launch?
- Which events are mandatory: install, registration, first word, first lesson, streak, paywall open, purchase?
- Do we need user-level attribution for Telegram Ads, and if yes how are we storing campaign context?
- What privacy / retention constraints do we want to enforce?

Implementation approach after agreement:
- Define event schema.
- Add server-side and frontend emission points.
- Add a minimal funnel dashboard or export path.

### P18. Language expansion decision and rollout

Open questions to agree before work starts:
- Which second language ships first, if any?
- Is the product multilingual in one shared account, or does language choice create separate study tracks?
- Do we have enough content quality for Georgian / Armenian / Turkish without creating a poor first impression?
- Do we support RU as the only explanation language, or multiple explanation languages too?

Implementation approach after agreement:
- Adjust data model for course/language dimension.
- Isolate language-specific content generation and packs.
- Update onboarding, filters, and analytics accordingly.

Decision so far:
- Georgian should be added now as the first explicit non-English study track.
- The architecture should be prepared for future additional languages, not hardcoded only for English + Georgian.
- Georgian support should be wired through backend, frontend, and tests before premium work starts.
- This should stay minimal and launch-oriented: enough structure to avoid repainting the house later, but not a giant framework rewrite.
- Course model should follow a Duolingo-style approach:
  - the user selects the studied language in settings
  - each studied language has its own separate, non-shared progress
  - English and Georgian should therefore behave as separate study tracks
- For now, the user interface and explanation language remain Russian only.
- First launch language direction:
  - RU-speaking users study either English or Georgian
  - future interface/explanation languages remain a later expansion, not part of this launch

Implementation outline:
1. Add a course dimension to study data.
   - Store `active_studied_language` on the user.
   - Store `course_code` on course-scoped entities.
   - Keep shared identity, but make study progress independent per course.
2. Move aggregate study counters to course-scoped progress.
   - Introduce `UserCourseProgress(user, course_code, ...)`.
   - Migrate existing English progress into the default `en` course.
3. Make shared services course-aware.
   - Filter word lists, drafts, packs, learning candidates, and achievements by the active course.
   - Ensure settings changes create course progress rows lazily.
4. Expose course selection through settings.
   - Frontend settings should let the user switch between `English` and `Georgian`.
   - The switch must behave like a track change, not a cosmetic filter.
5. Keep launch scope tight.
   - UI and explanations stay Russian-only.
   - Existing data defaults to `en`.
   - English packs remain available first; Georgian pack content is a follow-up under `P22`.
6. Verify isolation explicitly.
   - Add tests for course switching, per-course word isolation, and per-course progress isolation.
   - Run the existing backend test baseline after the migration and service changes.

Implementation status:
- Completed in the minimal launch-safe scope.
- Added `active_studied_language` to the user model and `course_code` to course-scoped study entities.
- Added `UserCourseProgress` and migrated legacy aggregate English progress into the default `en` course.
- Updated shared services so words, drafts, packs, achievements, and learning progress resolve against the active course.
- Added a studied-language switch to settings with Russian-only product copy and separate progress messaging.
- Added backend tests for course switching and per-course progress isolation.
- Verified with:
  - `python manage.py makemigrations --check`
  - `python -m pytest -q`
  - `npm run build`

### P19. Relocation-focused content packs and positioning assets

Open questions to agree before work starts:
- Which relocation scenarios matter first: rent, documents, bank, doctor, taxi, cafe, bureaucracy?
- Do we want to prioritize a narrower but hotter wedge: `job interviews and work-permit language for Georgia` ahead of broader relocation survival content?
- Are these packs expert-curated, AI-generated then reviewed, or mixed?
- Do we want phrase-first learning instead of single-word-first for relocation tracks?
- What tone do we want: survival basics, confident settling-in, or professional relocation assistant?

Implementation approach after agreement:
- Build first approved packs.
- Surface them in onboarding and landing.
- Validate uptake before expanding breadth.

Decision so far:
- Global positioning stays `Language for Relocation`.
- First geography-specific rollout should center on Georgia.
- Interview preparation is not the first content track; it stays deferred until everyday relocation utility is validated.

### P22. Georgia practical relocation scenarios

Open questions to agree before work starts:
- Which first scenarios ship first inside the Georgia relocation wedge: `work permit`, `bank account opening`, `documents`, `rent`, `doctor`, `taxi`, `shop`, `daily communication`?
- How do we want to sequence the first packs: bureaucracy-first, banking-first, or a mixed "first week in Georgia" path?
- Are the first tracks bilingual by design (`English + Georgian`) or split by selected study target?
- Which user segment is the initial default: general relocants, office workers, blue-collar workers, freelancers, or mixed?
- What is the minimum acceptable content format per scenario: vocabulary, ready-made phrases, listening, AI drills, and checklists?

Implementation approach after agreement:
- Define the first 2-3 practical Georgia scenarios.
- Build curated packs around real relocation tasks in English and Georgian.
- Reflect those scenarios in onboarding, paywall copy, landing, and acquisition messaging.

### P23. Interview preparation track (deferred)

Open questions to agree before work starts:
- Is interview prep a separate premium module or just another scenario family?
- Do we want English interview prep only, Georgian HR interactions, or both?
- Which interview audience comes first later: service jobs, office jobs, or IT?
- Do we need scenario simulation with AI dialogue before this track is worth launching?

Implementation approach after agreement:
- Keep this as a later specialized branch of the relocation product.
- Start only after the first practical Georgia scenarios prove demand and retention.

### P20. Landing page and acquisition assets

Open questions to agree before work starts:
- Do we need a real public landing or a simpler promo page first?
- What is the main CTA: open bot, open Mini App, start free, or try premium?
- Are we advertising English, relocation language learning, or both?
- Which screenshots and product stories best match the final positioning?

Implementation approach after agreement:
- Build conversion page.
- Align copy with onboarding and paywall.
- Prepare ad assets and verify event tracking before spend.

### P21. Referral and growth loops

Open questions to agree before work starts:
- What reward is economically safe: extra AI credits, premium days, extra packs?
- Should referrals reward only the inviter, or both inviter and invited user?
- What counts as a successful referral: install, registration, first word added, purchase?
- What abuse protections do we need from day one?
- Which loop is first: `share progress`, `share word`, `challenge invite`, or a combination?
- Do we want growth to come primarily from private sharing, group chats, or channel reposts?
- What exact share payload should be generated: static card, dynamic image, deep link, or mixed?
- Which metric defines loop viability: invite CTR, activation of referred users, share rate per active user, or paid conversion from referred traffic?

Implementation approach after agreement:
- Define referral model and reward policy.
- Add fraud-safe attribution rules.
- Track referral funnel separately in analytics.

Growth-loop framing:
- Treat referral as only one part of the growth system, not the whole system.
- The target is a true loop: `User -> Action -> Exposure -> New Users -> Repeat`.
- Telegram-native candidates for VocabuMe:
  - `share progress`
  - `share word`
  - `challenge invite`
  - later, possibly `group learning`

Recommended rollout order:
1. `share word`
2. `share progress`
3. `referral bonus`
4. later: `challenge loop` and `group learning`

Minimum viable loop set:
- Shareable word card with deep link back to VocabuMe.
- Shareable progress card with streak / learned words / CTA.
- Referral bonus tied to a clear activation event.
