# ТЗ: переработка freemium-модели и Telegram Stars billing

- **Дата:** 2026-07-29
- **Статус:** proposal / ready for implementation planning
- **Продукт:** VocabuMe — Telegram-бот, Telegram Mini App и website
- **Основной рынок MVP:** релоканты и экспаты в Грузии

## 1. Цель

Переработать существующую платежную модель VocabuMe так, чтобы:

1. бесплатная версия оставалась полноценным способом познакомиться с продуктом и сформировать учебную привычку;
2. Premium продавался после подтвержденного момента ценности и в контексте конкретной задачи пользователя;
3. цена, показываемая пользователю, совпадала с суммой реального Telegram Stars invoice;
4. месячный план поддерживал настоящее автоматическое продление;
5. годовой план оставался прозрачным предоплаченным доступом без скрытого автопродления;
6. платные AI-возможности имели конечные cost-guard лимиты и положительную юнит-экономику;
7. вся воронка от первого полезного действия до оплаты, продления, отмены и возврата измерялась серверной аналитикой;
8. Telegram-бот, Mini App и website использовали один entitlement и одну платежную историю.

Изменение считается успешным не после появления нового paywall, а после прохождения полного production quality gate и получения достоверной воронки на реальных пользователях.

## 2. Подтвержденное исходное состояние

На момент подготовки ТЗ:

- тарифы описаны в `vocab/monetization.py`;
- UI показывает `$6.99 / month` и `$39.99 / year`;
- фактические Telegram invoice используют `199 XTR` и `999 XTR`;
- месячная и годовая покупки дают доступ на 30 и 365 дней соответственно;
- `subscription_period` в Telegram invoice не передается, поэтому обе покупки являются разовыми;
- при новой успешной покупке предыдущая активная подписка переводится в `expired`;
- Free получает starter-паки, до 10 новых элементов в день, до 3 дополнительных регенераций изображения и неограниченную практику по уже сохраненному материалу;
- Premium снимает дневные лимиты полностью;
- `ai_explanations` и `ai_dialogue` объявлены entitlement-флагами, но законченный пользовательский сценарий этих возможностей отсутствует;
- клик по закрытому паку переносит пользователя в `More / Settings`, а не открывает контекстный paywall;
- soft trigger `post_first_value_moment` описан, но не реализован;
- `paywall_opened` разрешен аналитикой, но не вызывается из frontend;
- `practice_completed` фактически записывается после каждого ответа, а не после завершенной учебной сессии;
- установленная версия `python-telegram-bot` уже поддерживает:
  - `subscription_period` для `create_invoice_link`;
  - `SuccessfulPayment.subscription_expiration_date`;
  - `SuccessfulPayment.is_recurring`;
  - `SuccessfulPayment.is_first_recurring`;
  - `Bot.edit_user_star_subscription`;
  - `Bot.refund_star_payment`.

Обновление Telegram SDK для этой задачи не требуется.

## 3. Зафиксированные продуктовые решения

### 3.1. Каталог запуска

После запуска Monetization V2 должны существовать два платных предложения.

#### `premium_monthly_recurring_v2`

- Название в UI: `Premium на месяц`;
- цена: `399 XTR`;
- тип покупки: recurring;
- период Telegram: ровно `30 * 24 * 60 * 60` секунд;
- автопродление: включено Telegram;
- entitlement: Premium;
- доступ действует до `subscription_expiration_date`, полученной от Telegram;
- пользователь может отключить и повторно включить автопродление до окончания текущего периода.

Ориентировочная reward-value разработчика при текущем коэффициенте Telegram: `399 * $0.013 = $5.187`.

#### `premium_yearly_prepaid_v2`

- Название в UI: `Premium на год`;
- цена: `2 999 XTR`;
- тип покупки: prepaid;
- период доступа: 365 дней;
- `subscription_period` в invoice отсутствует;
- автоматического продления нет;
- entitlement: Premium;
- в UI явно написано `разовая оплата за 365 дней`.

Ориентировочная reward-value разработчика при текущем коэффициенте Telegram: `2 999 * $0.013 = $38.987`.

Эффективная скидка годового плана относительно 12 месячных периодов составляет примерно 37%.

### 3.2. Отображение цены

В Telegram-боте и Telegram Mini App:

- основной и обязательный формат цены — Telegram Stars;
- запрещено показывать `$6.99`, `$39.99` или иной USD-эквивалент как цену покупки;
- месячная кнопка: `399 ⭐ / месяц`;
- годовая кнопка: `2 999 ⭐ / год`;
- под годовой кнопкой: `≈ 250 ⭐ / месяц · разовая оплата`;
- UI не должен самостоятельно пересчитывать Stars в USD, GEL, RUB или другую валюту.

На website без Telegram WebView:

- до появления отдельного web-провайдера используется тот же Telegram Stars checkout;
- отображается та же цена в Stars;
- вход и entitlement остаются Telegram-based;
- подключение Stripe, карт или отдельной web-идентичности в это ТЗ не входит.

### 3.3. Free

Free должен оставаться полезным и не блокировать уже созданный учебный прогресс.

В Free остаются:

- все starter-паки, помеченные `starter_pack=True`;
- превью из 3–5 фраз каждого закрытого relocation-пака;
- просмотр и практика по уже добавленным словам;
- карточки, повторение, напоминания, алфавит и неправильные глаголы;
- сохраненный словарь и прогресс без удаления после окончания Premium;
- до 10 новых слов/фраз в день на первом этапе rollout;
- первая автоматическая генерация изображения для нового элемента;
- до 3 дополнительных регенераций изображения в день на первом этапе rollout.

Важно:

- лимит 10 новых элементов пока не уменьшается без production-данных;
- starter-паки не должны становиться платными;
- окончание Premium никогда не удаляет слова, картинки, историю практики или прогресс;
- Premium-expiration ограничивает только будущие платные действия и доступ к закрытым пакам.

### 3.4. Premium entitlements

Premium дает:

- доступ ко всем relocation-пакам обоих изучаемых языков;
- увеличенный лимит новых персональных элементов;
- увеличенный лимит регенераций изображений;
- доступ к будущим AI-функциям только после их фактического production-релиза.

На старте V2:

- `max_new_items_per_day = 50`;
- `max_extra_image_regenerations_per_day = 10`;
- `premium_relocation_packs = true`;
- `ai_explanations = false`;
- `ai_dialogue = false`.

Значение `None` / unlimited запрещено для платных или потенциально платных OpenAI/TTS/STT операций.

В маркетинговом тексте запрещены формулировки:

- `безлимитный AI`;
- `безлимитная генерация`;
- `AI-диалоги`, пока диалоги не прошли production quality gate;
- `AI-объяснения`, пока отдельный пользовательский сценарий объяснений не реализован.

Разрешенная формулировка: `расширенные AI-лимиты`.

### 3.5. Альтернативный 90-дневный offer

`Relocation Pass на 90 дней` не входит в первый production rollout.

Его разрешено добавить отдельным экспериментом только если одновременно выполнены условия:

- накоплено не менее 500 уникальных показов paywall;
- D30 retention активированных пользователей ниже 20%;
- годовой план выбирают менее 15% плательщиков;
- есть подтверждение, что значимая доля пользователей воспринимает VocabuMe как временный инструмент подготовки к переезду.

До выполнения условий третий план не показывать, чтобы не увеличивать выбор на paywall.

## 4. Целевой пользовательский сценарий

### 4.1. Новый пользователь

1. Пользователь авторизуется через Telegram.
2. Выбирает изучаемый язык.
3. Выбирает актуальную задачу:
   - банк;
   - аренда;
   - документы / ВНЖ;
   - врач / аптека;
   - работа;
   - первые дни;
   - свои слова.
4. Открывает доступный starter-сценарий или превью выбранного закрытого сценария.
5. Добавляет фразы или проходит не менее трех вопросов практики.
6. Backend/frontend фиксируют `first_value_completed`.
7. После успешного результата разрешен soft paywall.
8. Пользователь может купить Premium или продолжить Free без потери созданного результата.

До шага 5 полноэкранный Premium purchase screen не показывать.

Допускается ненавязчивый Premium badge и отображение закрытых сценариев.

### 4.2. Клик по закрытому сценарию

1. Пользователь видит название, описание, количество ситуаций и 3–5 preview-фраз.
2. Клик `Открыть весь сценарий` вызывает backend endpoint регистрации paywall impression.
3. Backend возвращает актуальные планы и создает идемпотентное событие `paywall_shown`.
4. Frontend открывает `PremiumPaywall` поверх текущего экрана.
5. Контекст сценария и scroll position сохраняются.
6. Закрытие paywall возвращает пользователя туда же, а не в `Settings`.

Пример заголовка:

> Подготовься к разговору с арендодателем

Пример текста:

> Открой фразы для просмотра квартиры, договора, депозита, коммуналки и передачи ключей. После добавления пройди короткую практику.

### 4.3. Достижение дневного лимита

Backend возвращает `409` или `403` с machine-readable кодом и структурированным paywall payload.

Frontend:

- не теряет введенные слова;
- сохраняет draft локально или на backend;
- показывает оставшийся лимит и время следующего reset;
- открывает контекстный paywall;
- после успешной покупки повторяет исходное действие только по явному нажатию пользователя, а не автоматически.

### 4.4. Оплата

1. Пользователь выбирает plan на paywall.
2. Frontend отправляет `plan_code`, `source` и `paywall_impression_id`.
3. Backend повторно проверяет активность plan и цену.
4. Backend создает `PaymentAttempt`.
5. Backend создает Telegram invoice:
   - с `subscription_period=2_592_000` для monthly recurring;
   - без `subscription_period` для yearly prepaid.
6. Mini App вызывает `Telegram.WebApp.openInvoice`.
7. Entitlement активируется только после server-side `SUCCESSFUL_PAYMENT`.
8. Client callback `paid` используется только для обновления UI и повторного GET `/api/billing`.

Нельзя выдавать Premium только на основании client callback.

### 4.5. Отмена месячного автопродления

1. Пользователь открывает Premium management screen.
2. Нажимает `Отключить автопродление`.
3. UI показывает точную дату, до которой доступ сохранится.
4. После подтверждения backend вызывает:
   `Bot.edit_user_star_subscription(user_id, first_charge_id, is_canceled=True)`.
5. `cancel_at_period_end` становится `true`.
6. Entitlement остается активным до `current_period_end`.
7. Событие `subscription_cancel_scheduled` записывается один раз.

Повторное включение до окончания периода:

- вызывает тот же Telegram API с `is_canceled=False`;
- сбрасывает `cancel_at_period_end`;
- пишет `subscription_resumed`.

### 4.6. Продление

Каждый recurring charge приходит как новый `SuccessfulPayment`.

Обработчик:

- использует `telegram_payment_charge_id` как idempotency key транзакции;
- проверяет `is_recurring`;
- связывает charge с существующим monthly subscription;
- обновляет `current_period_start`, `current_period_end`, `last_payment_at`;
- использует `subscription_expiration_date` Telegram как приоритетный источник даты;
- создает `payment_succeeded` и `subscription_renewed`;
- не создает второй параллельный active entitlement.

Если `subscription_expiration_date` отсутствует:

- разрешен fallback `now + 30 days`;
- создается alert/error event;
- кейс покрывается отдельным тестом;
- отсутствие даты не должно приводить к двойному начислению периода.

### 4.7. Годовой prepaid

- один успешный charge дает 365 дней доступа;
- повторная покупка годового плана добавляет 365 дней к более поздней из дат `now` и текущего `current_period_end`;
- годовой план не маскируется под recurring;
- кнопок `отменить автопродление` и `возобновить` для него нет.

### 4.8. Конфликт планов

В V2 не реализуется автоматический upgrade/downgrade между recurring monthly и prepaid yearly.

Правила:

- при активном monthly recurring нельзя купить yearly, пока автопродление не отключено;
- после отключения разрешено купить yearly;
- купленный yearly создается со `status=pending`, если monthly-период еще не
  закончился; его `current_period_start` равен окончанию monthly, а
  `current_period_end = current_period_start + 365 days`;
- `reconcile_user_subscriptions(user, now)` атомарно завершает истекший monthly и
  активирует scheduled yearly; use case вызывается перед каждым чтением billing или
  entitlement и не требует нового фонового цикла;
- при активном yearly разрешено купить еще один yearly и продлить срок;
- при активном yearly monthly recurring checkout не предлагается;
- API при недопустимой комбинации возвращает `409 active_plan_conflict`.

Правила должны быть одинаковыми в bot, Mini App и website.

## 5. Архитектура и границы ответственности

### 5.1. Источники истины

- `SubscriptionPlan` — версия продаваемого предложения;
- `PaymentAttempt` — попытка открыть конкретный invoice;
- `PaymentTransaction` — неизменяемая запись каждого успешного charge/refund;
- `UserSubscription` — текущий billing contract и период entitlement;
- `vocab/application/billing.py` — все billing transitions;
- `vocab/application/entitlements.py` — все проверки Free/Premium;
- HTTP и Telegram handlers — только transport adapters;
- frontend никогда не вычисляет entitlement самостоятельно.

### 5.2. Запрещенные архитектурные решения

- хранить Premium только в frontend state;
- считать callback `openInvoice(status="paid")` доказательством оплаты;
- менять историческую цену старого плана через `update_or_create`;
- использовать один `PaymentAttempt` как ledger нескольких recurring charges;
- продлевать entitlement без уникального Telegram charge id;
- обнулять или удалять пользовательский прогресс после expiry;
- размещать разные лимиты в bot и frontend;
- дублировать каталог цен в React.

## 6. Изменения данных

### 6.1. `SubscriptionPlan`

Добавить поля:

```python
product_code = models.CharField(max_length=50, default="premium")
catalog_version = models.PositiveIntegerField(default=1)
price_stars = models.PositiveIntegerField(null=True)
purchase_mode = models.CharField(
    max_length=20,
    choices=(("recurring", "Recurring"), ("prepaid", "Prepaid")),
    default="prepaid",
)
subscription_period_seconds = models.PositiveIntegerField(null=True, blank=True)
display_order = models.PositiveSmallIntegerField(default=100)
is_featured = models.BooleanField(default=False)
```

Ограничения:

- `price_stars > 0` для active plan;
- recurring plan обязан иметь `subscription_period_seconds=2_592_000`;
- prepaid plan обязан иметь `subscription_period_seconds=NULL`;
- `code` остается immutable unique versioned code;
- новые plan codes не переиспользуют legacy-коды.

Текущие `currency` и `price_amount` оставить на время миграции для чтения старых записей, но:

- новые V2 plans хранят `currency="XTR"`;
- `price_stars` является источником фактической invoice price;
- `price_amount` не передается в клиент как purchase price;
- удалить legacy-поля можно отдельной задачей после завершения rollback window.

### 6.2. `PaymentAttempt`

Добавить:

```python
paywall_impression_id = models.UUIDField(null=True, blank=True)
idempotency_key = models.UUIDField(default=uuid.uuid4, unique=True)
expires_at = models.DateTimeField(null=True, blank=True)
purchase_mode = models.CharField(max_length=20, default="prepaid")
```

Изменить семантику:

- `amount_minor` для `XTR` означает целое количество Stars;
- pending invoice имеет TTL 30 минут;
- повторный checkout с тем же `idempotency_key` возвращает существующий invoice;
- новый checkout после cancelled/failed attempt получает новый UUID;
- после TTL attempt переводится в `cancelled` с metadata reason `expired`;
- старые pending attempts при price cutover переводятся в `cancelled` с reason `catalog_v2_cutover`.

### 6.3. Новый `PaymentTransaction`

```python
class PaymentTransaction(models.Model):
    user = models.ForeignKey(TelegramUser, on_delete=models.PROTECT)
    subscription = models.ForeignKey(
        UserSubscription,
        on_delete=models.PROTECT,
        null=True,
        blank=True,
    )
    attempt = models.ForeignKey(
        PaymentAttempt,
        on_delete=models.PROTECT,
        null=True,
        blank=True,
    )
    plan = models.ForeignKey(SubscriptionPlan, on_delete=models.PROTECT)
    kind = models.CharField(
        max_length=20,
        choices=(
            ("initial", "Initial"),
            ("renewal", "Renewal"),
        ),
    )
    status = models.CharField(
        max_length=20,
        choices=(("succeeded", "Succeeded"), ("refunded", "Refunded")),
    )
    currency = models.CharField(max_length=10, default="XTR")
    amount_stars = models.PositiveIntegerField()
    telegram_payment_charge_id = models.CharField(max_length=255, unique=True)
    provider_payment_charge_id = models.CharField(max_length=255, blank=True, default="")
    subscription_expiration_date = models.DateTimeField(null=True, blank=True)
    is_recurring = models.BooleanField(default=False)
    is_first_recurring = models.BooleanField(default=False)
    occurred_at = models.DateTimeField()
    refunded_at = models.DateTimeField(null=True, blank=True)
    refund_reason = models.CharField(max_length=255, blank=True, default="")
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
```

Требования:

- ledger append-only, кроме перехода `succeeded -> refunded`;
- повторная доставка одного charge не создает новую строку;
- raw Telegram Update и персональные данные в metadata не сохранять;
- refund обновляет исходную transaction до `status=refunded`, не создает вторую
  строку с тем же charge id и не меняет `amount_stars`.

### 6.4. `UserSubscription`

Добавить:

```python
purchase_mode = models.CharField(max_length=20, default="prepaid")
auto_renew = models.BooleanField(default=False)
cancel_at_period_end = models.BooleanField(default=False)
current_period_start = models.DateTimeField(null=True, blank=True)
current_period_end = models.DateTimeField(null=True, blank=True)
first_telegram_payment_charge_id = models.CharField(
    max_length=255,
    blank=True,
    default="",
)
last_payment_at = models.DateTimeField(null=True, blank=True)
```

Переходный период:

- `expires_at` синхронизируется с `current_period_end`;
- entitlement читает `current_period_end`, а при `NULL` fallback на legacy `expires_at`;
- после rollback window `expires_at` может стать compatibility-only полем.

Допустимые состояния:

- `active`;
- `expired`;
- `refunded`.

Legacy-значение `cancelled` остается читаемым для старых данных, но новая recurring
подписка после отключения автопродления сохраняет `status=active` и получает
`cancel_at_period_end=True`. В `expired` она переходит только после окончания
оплаченного периода. Нужно расширить choices значением `refunded` и не использовать
`expired` или `cancelled` как замену отключению автопродления.

### 6.5. `ProductEvent`

Добавить:

```python
event_id = models.UUIDField(default=uuid.uuid4, unique=True)
```

Добавить event names:

- `onboarding_completed`;
- `first_value_completed`;
- `paywall_shown`;
- `paywall_plan_selected`;
- `checkout_started`;
- `checkout_cancelled`;
- `checkout_failed`;
- `payment_succeeded`;
- `subscription_activated`;
- `subscription_renewed`;
- `subscription_cancel_scheduled`;
- `subscription_resumed`;
- `subscription_expired`;
- `payment_refunded`.

Legacy `paywall_opened` оставить читаемым, но перестать писать после cutover.

`practice_completed` не переиспользовать для отдельных ответов. До появления server-side learning session:

- текущий per-answer event переименовать в `practice_answered`;
- завершение frontend-сессии отправляет bounded client event `practice_session_completed`;
- `first_value_completed` допускается после первого pack add или первой завершенной практики.

## 7. Каталог и конфигурация

### 7.1. `vocab/monetization.py`

Сделать source of truth versioned:

```python
MONETIZATION_CATALOG_VERSION = 2

PLAN_DEFINITIONS = {
    "free": {...},
    "premium_monthly_recurring_v2": {
        "product_code": "premium",
        "price_stars": 399,
        "purchase_mode": "recurring",
        "duration_days": 30,
        "subscription_period_seconds": 2_592_000,
        "is_featured": False,
    },
    "premium_yearly_prepaid_v2": {
        "product_code": "premium",
        "price_stars": 2_999,
        "purchase_mode": "prepaid",
        "duration_days": 365,
        "subscription_period_seconds": None,
        "is_featured": True,
    },
}
```

`get_telegram_stars_prices_for_user`:

- продолжает поддерживать test-user override;
- override применяется по plan code, а не по `monthly/yearly`;
- production fallback price отсутствует: неизвестный plan должен завершаться ошибкой, а не ценой по умолчанию.

### 7.2. Feature flags

Добавить:

- `MONETIZATION_V2_ENABLED`;
- `PAYWALL_V2_ENABLED`;
- `TELEGRAM_RECURRING_ENABLED`;
- `MONETIZATION_V2_TEST_USER_IDS`.

Поведение:

- test users могут получать V2 до общего rollout;
- production rollout включается одной конфигурацией без новой миграции;
- выключение recurring не отзывает уже купленные recurring subscriptions;
- при rollback обработчик renewal обязан продолжать принимать уже оплаченные recurring charges.

## 8. API-контракты

### 8.1. `GET /api/billing`

Response:

```json
{
  "ok": true,
  "billing": {
    "premium_active": false,
    "entitlement_code": "free",
    "active_subscription": null,
    "plans": [
      {
        "code": "premium_monthly_recurring_v2",
        "name": "Premium на месяц",
        "price": {
          "amount": 399,
          "currency": "XTR",
          "display": "399 ⭐"
        },
        "purchase_mode": "recurring",
        "duration_days": 30,
        "auto_renews": true,
        "subscription_period_seconds": 2592000,
        "is_featured": false
      },
      {
        "code": "premium_yearly_prepaid_v2",
        "name": "Premium на год",
        "price": {
          "amount": 2999,
          "currency": "XTR",
          "display": "2 999 ⭐"
        },
        "purchase_mode": "prepaid",
        "duration_days": 365,
        "auto_renews": false,
        "monthly_equivalent_stars": 250,
        "is_featured": true
      }
    ]
  }
}
```

Для active subscription:

```json
{
  "plan_code": "premium_monthly_recurring_v2",
  "status": "active",
  "purchase_mode": "recurring",
  "current_period_end": "2026-08-28T12:00:00Z",
  "auto_renew": true,
  "cancel_at_period_end": false,
  "can_cancel": true,
  "can_resume": false
}
```

USD price в response отсутствует.

### 8.2. `POST /api/monetization/paywall-impressions`

Request:

```json
{
  "impression_id": "uuid",
  "trigger": "premium_pack_gate",
  "source": "packs",
  "course_code": "ka",
  "pack_id": "georgia_rent_utilities_ka",
  "level_id": "viewing_and_contract"
}
```

Response:

```json
{
  "ok": true,
  "paywall": {
    "impression_id": "uuid",
    "trigger": "premium_pack_gate",
    "title": "Подготовься к разговору с арендодателем",
    "body": "Открой весь сценарий аренды и короткую практику.",
    "plans": []
  }
}
```

Требования:

- authenticated only;
- `impression_id` идемпотентен;
- `impression_id` используется как `ProductEvent.event_id`, отдельная таблица
  impression на MVP не создается;
- trigger входит в allowlist;
- свойства bounded и не содержат пользовательский текст;
- повторный request с тем же UUID возвращает тот же response и не дублирует событие.

### 8.3. `POST /api/billing/checkout`

Request V2:

```json
{
  "checkout_id": "uuid",
  "plan_code": "premium_yearly_prepaid_v2",
  "source": "paywall",
  "paywall_impression_id": "uuid"
}
```

Поле `billing_period` deprecated и не принимается после удаления compatibility window.

`checkout_id` обязателен для Mini App/website и используется как
`PaymentAttempt.idempotency_key`. Bot transport генерирует UUID на backend.

Response:

```json
{
  "ok": true,
  "attempt_id": 123,
  "invoice_link": "https://t.me/$...",
  "plan": {
    "code": "premium_yearly_prepaid_v2",
    "price": {"amount": 2999, "currency": "XTR"},
    "purchase_mode": "prepaid",
    "auto_renews": false
  }
}
```

Ошибки:

- `400 unknown_plan`;
- `409 active_plan_conflict`;
- `409 checkout_already_paid`;
- `429 checkout_rate_limited`;
- `503 billing_temporarily_unavailable`.

### 8.4. `POST /api/billing/subscription/cancel`

Request:

```json
{"subscription_id": 123}
```

Только recurring.

Response:

```json
{
  "ok": true,
  "subscription": {
    "status": "active",
    "cancel_at_period_end": true,
    "current_period_end": "2026-08-28T12:00:00Z"
  }
}
```

### 8.5. `POST /api/billing/subscription/resume`

Контракт аналогичен cancel.

Ошибки cancel/resume:

- `404 subscription_not_found`;
- `409 subscription_not_recurring`;
- `409 subscription_already_cancelled`;
- `409 subscription_not_cancelled`;
- `503 telegram_billing_unavailable`.

Операции идемпотентны.

### 8.6. Entitlement error

Единый формат:

```json
{
  "ok": false,
  "code": "paywall_daily_new_items_limit",
  "message": "На сегодня использован лимит новых AI-карточек.",
  "paywall": {
    "trigger": "daily_new_items_limit",
    "remaining": 0,
    "reset_at": "2026-07-30T00:00:00+04:00",
    "plans": []
  }
}
```

Цены всегда берутся из backend catalog.

### 8.7. OpenAPI

Обновить `/api/openapi.json`:

- новые endpoints;
- новые request/response schemas;
- error codes;
- recurring/prepaid semantics;
- XTR integer amount;
- отсутствие USD purchase price;
- cancel/resume behavior.

## 9. Billing application service

В `vocab/application/billing.py` выделить use cases:

```python
sync_subscription_plans()
list_sellable_plans(user)
create_checkout(user, plan_code, source, impression_id, idempotency_key)
handle_successful_payment(successful_payment)
cancel_recurring_subscription(user, subscription_id)
resume_recurring_subscription(user, subscription_id)
refund_payment(admin_actor, transaction_id, reason)
reconcile_user_subscriptions(user, now)
expire_due_subscriptions(now)
get_billing_payload(user)
```

`handle_successful_payment` должен:

1. начать DB transaction;
2. получить attempt по payload с row lock;
3. проверить currency и amount против immutable plan/attempt;
4. проверить уникальность charge id;
5. создать `PaymentTransaction`;
6. определить initial/renewal;
7. активировать или продлить `UserSubscription`;
8. синхронизировать `current_period_end` и `expires_at`;
9. зафиксировать ProductEvent через `transaction.on_commit`;
10. вернуть сериализованный subscription.

Запрещено автоматически переводить любой предыдущий active subscription в `expired` без учета правил конфликта планов.

## 10. Telegram handlers

### 10.1. `/subscribe`

Показывает актуальный backend catalog:

- `399 ⭐ / месяц · автопродление`;
- `2 999 ⭐ / год · разовая оплата`.

Текст не содержит `безлимитный AI`.

### 10.2. Invoice

Monthly:

```python
subscription_period=timedelta(days=30)
```

Yearly:

```python
subscription_period=None
```

Одинаковое правило применяется к `send_invoice` и `create_invoice_link`.

### 10.3. Successful payment

Передавать в application service:

- `currency`;
- `total_amount`;
- `invoice_payload`;
- `telegram_payment_charge_id`;
- `provider_payment_charge_id`;
- `subscription_expiration_date`;
- `is_recurring`;
- `is_first_recurring`.

Пользовательское сообщение:

- initial monthly: `Premium активирован. Следующее продление — <date>.`;
- renewal: `Premium продлен до <date>.`;
- yearly: `Premium активирован до <date>. Это разовая покупка без автопродления.`;

Повторная доставка Update не должна повторно отправлять сообщение об активации.

### 10.4. Support и refund

Сохранить `/paysupport`.

Добавить staff-only service/admin action:

1. проверить transaction;
2. вызвать `refund_star_payment`;
3. пометить transaction refunded;
4. пересчитать entitlement;
5. записать `payment_refunded`;
6. сохранить reason без персональных данных.

Публичный автоматический refund endpoint не создавать.

## 11. Frontend

### 11.1. Новый `PremiumPaywall`

Создать отдельный компонент, не встраивать purchase UI в `SettingsScreen`.

Props:

```ts
type PremiumPaywallProps = {
  open: boolean;
  impressionId: string;
  trigger: PaywallTrigger;
  context: PaywallContext;
  plans: BillingPlan[];
  activePlanCode?: string;
  checkoutBusyPlanCode?: string;
  onSelectPlan(planCode: string): void;
  onClose(): void;
};
```

Требования:

- bottom sheet на коротких мобильных экранах;
- dialog на широких экранах;
- safe-area padding;
- CTA виден без перекрытия bottom navigation;
- внутренний scroll при недостаточной высоте;
- закрытие доступно без dark pattern;
- цена, recurring и prepaid условия видны до клика;
- фокус и `aria-modal` корректны;
- после закрытия восстанавливаются tab, internal mode и scroll context.

### 11.2. Onboarding

Удалить текущий обязательный Premium screen до первого использования.

Новый flow:

- language;
- актуальная задача;
- starter/preview;
- первое полезное действие;
- soft paywall или переход в основное приложение.

Кнопка `Начать бесплатно` не должна конкурировать с двумя другими primary CTA на одном экране.

### 11.3. Packs

Для locked pack:

- карточка открывает preview;
- preview показывает 3–5 первых элементов;
- полные данные не добавляются в словарь без entitlement;
- CTA `Открыть весь сценарий`;
- paywall открывается локально;
- переход в `More` удаляется.

### 11.4. Settings / Premium management

Settings показывает:

- активный plan;
- дату окончания периода;
- `автопродление включено/отключено`;
- кнопку cancel/resume для recurring;
- текст `разовая оплата` для prepaid;
- ссылку/команду поддержки по платежам.

Settings не является основным acquisition paywall.

### 11.5. Invoice callback

Обработать:

- `paid`;
- `cancelled`;
- `failed`;
- `pending`, если поддерживается клиентом.

После любого terminal status:

- вызвать GET `/api/billing`;
- не активировать Premium локально;
- записать bounded analytics event для cancelled/failed;
- сохранить paywall context, чтобы пользователь мог повторить checkout.

## 12. Аналитика и отчеты

### 12.1. Обязательные свойства

Для monetization events:

- `source`;
- `trigger`;
- `course_code`;
- `pack_id`;
- `level_id`;
- `plan_code`;
- `purchase_mode`;
- `stars_amount`;
- `is_recurring`;
- `impression_id`.

Не отправлять:

- Telegram initData;
- invoice link;
- charge id;
- username;
- пользовательский текст;
- аудио;
- prompt;
- полный Telegram Update.

### 12.2. Воронка

Management command/report должен выдавать за период и по cohort:

1. authenticated users;
2. onboarding completed;
3. first value completed;
4. unique paywall shown;
5. unique checkout started;
6. unique payment succeeded;
7. subscription renewed;
8. cancellation scheduled;
9. refund;
10. D1/D7/D30 retained users.

Обязательные коэффициенты:

- activation = `first_value_completed / authenticated`;
- paywall reach = `paywall_shown / first_value_completed`;
- paywall-to-checkout = `checkout_started / paywall_shown`;
- checkout-to-paid = `payment_succeeded / checkout_started`;
- activated-to-paid = `payment_succeeded / first_value_completed`;
- first renewal rate;
- cancellation rate;
- refund rate.

### 12.3. Юнит-экономика

Добавить management report, объединяющий:

- `PaymentTransaction.amount_stars`;
- configurable `TELEGRAM_STAR_REWARD_USD`, default `0.013`;
- `OpenAIUsageEvent.cost_microusd`;
- TTS cost, если появится платный TTS provider;
- refunds.

Минимальный output:

```text
gross_reward_usd
openai_cost_usd
refund_reward_usd
estimated_gross_margin_usd
estimated_gross_margin_percent
paying_users
ai_cost_per_payer
ai_cost_per_free_active_user
```

Не называть показатель бухгалтерской выручкой: это оценка reward-value до налогов, TON conversion и прочих расходов.

## 13. Миграция и обратная совместимость

### 13.1. Legacy plans

Существующие:

- `premium_monthly`;
- `premium_yearly`.

После включения V2:

- перестают продаваться новым пользователям;
- остаются в БД;
- `is_active=False`;
- исторические подписки продолжают ссылаться на них;
- цена и metadata исторических plans не перезаписываются.

### 13.2. Активные legacy subscriptions

- сохраняют текущий `expires_at`;
- не сокращаются;
- не переводятся автоматически в recurring;
- пользователь видит `предоплаченный доступ до <date>`;
- после expiry ему предлагается V2 catalog.

### 13.3. Pending attempts

В момент cutover:

- pending legacy attempts переводятся в `cancelled`;
- `cancelled_at=now`;
- metadata reason: `catalog_v2_cutover`;
- pre-checkout для старого invoice отклоняется с понятным сообщением открыть оплату заново.

Paid attempts и subscriptions не изменяются.

### 13.4. Rollback

Rollback V2:

- скрывает V2 plans из новых checkout;
- не отменяет уже активные recurring subscriptions;
- successful renewal handler остается включенным;
- cancel/resume остаются доступны;
- legacy entitlement fallback продолжает работать;
- migrations назад не откатываются в production;
- данные V2 не удаляются.

## 14. Безопасность и надежность

Обязательно:

- server-authoritative price;
- exact amount/currency validation до pre-checkout и activation;
- unique charge id;
- row locks на attempt, transaction и subscription transitions;
- idempotency для checkout, successful payment, renewal, cancel, resume и refund;
- rate limit checkout не менее текущего уровня;
- invoice TTL;
- никакого Premium по query param, cookie или frontend flag;
- audit trail для staff refund;
- alert при успешной оплате, которую не удалось связать с entitlement;
- alert при recurring payment без expiration date;
- логирование без invoice links, tokens и raw payment payload.

## 15. Тестирование

### 15.1. Backend unit/integration

Обязательные тесты:

- sync создает V2 plans и не изменяет legacy plans;
- monthly invoice содержит `subscription_period=2_592_000`;
- yearly invoice не содержит subscription period;
- invoice amount точно `399/2 999 XTR`;
- неизвестный/неактивный plan отклоняется;
- client не может подменить цену;
- initial recurring charge создает одну transaction и один subscription;
- duplicate initial update идемпотентен;
- renewal создает новую transaction и продлевает существующий subscription;
- duplicate renewal не продлевает второй раз;
- renewal с неверной суммой отклоняется;
- expiration date Telegram имеет приоритет;
- fallback expiration пишет alert;
- yearly repurchase добавляет 365 дней;
- yearly, купленный после scheduled cancel monthly, не отнимает остаток monthly и
  атомарно активируется после его окончания;
- конфликт планов возвращает 409;
- cancel сохраняет entitlement до period end;
- duplicate cancel идемпотентен;
- resume восстанавливает auto-renew;
- prepaid cancel/resume отклоняется;
- refund вызывает Telegram API один раз;
- refund повторно идемпотентен;
- refund пересчитывает entitlement;
- legacy active subscription не меняет срок;
- legacy pending invoice после cutover отклоняется;
- expiry не удаляет словарь и прогресс;
- Free и Premium лимиты применяются одинаково через bot/API;
- Premium AI operations имеют конечные лимиты;
- paywall impression UUID дедуплицируется;
- analytics properties очищаются от запрещенных ключей;
- funnel считает unique users, а не количество ответов.

Для конкурентных переходов использовать PostgreSQL tests с реальными row locks.

### 15.2. Frontend unit/component

- Stars price render;
- отсутствие USD price;
- recurring/prepaid disclosure;
- contextual copy для pack/limit triggers;
- paywall close восстанавливает context;
- checkout busy state scoped по plan code;
- paid callback делает billing refetch;
- cancelled/failed не активируют Premium;
- settings cancel/resume;
- locked pack preview;
- onboarding не показывает purchase до first value.

### 15.3. Playwright mobile E2E

Минимальные viewport:

- `390x844`;
- `360x640`;
- дополнительный короткий Telegram-like viewport.

Проверить:

- Today;
- Learn;
- Words;
- Progress;
- More;
- locked pack preview;
- paywall;
- invoice-open stub;
- paid/cancelled/failed callbacks;
- bottom navigation;
- safe areas;
- keyboard + paywall;
- scroll restoration.

### 15.4. Telegram sandbox/test-user

Через `TELEGRAM_STARS_TEST_USER_IDS` и цену `1 XTR` проверить:

- Mini App initial monthly recurring;
- bot `/subscribe` monthly recurring;
- yearly prepaid;
- successful activation;
- simulated/real renewal, если Telegram test flow позволяет;
- cancel;
- resume;
- refund;
- duplicate Update delivery.

## 16. Production rollout

### Этап 0. Baseline

До изменения цены:

- снять текущую production funnel за 30/90 дней;
- зафиксировать число active legacy subscriptions;
- зафиксировать pending attempts;
- посчитать фактические Stars и OpenAI cost;
- не менять production background jobs.

Если production DB access отсутствует, rollout блокируется до получения baseline владельцем системы.

### Этап 1. Analytics и paywall

- новые события;
- paywall impression endpoint;
- contextual paywall;
- исправленная семантика practice events;
- старые цены пока не меняются;
- production smoke и mobile flow.

### Этап 2. Cost guards

- конечные Premium limits;
- usage reporting;
- AI copy cleanup;
- проверка gross-margin report;
- production smoke с реальным созданием данных и non-empty media bytes.

### Этап 3. Catalog V2 для test users

- миграции;
- 399/2 999 plans;
- recurring invoice;
- prepaid invoice;
- cancel/resume/refund;
- полный 1 XTR smoke.

### Этап 4. General availability

- отменить pending legacy attempts;
- выключить legacy plans для продажи;
- включить V2 catalog;
- проверить live frontend asset hash;
- провести реальную покупку production plan контролируемым аккаунтом;
- проверить entitlement во всех трех поверхностях;
- выполнить refund тестовой production покупки, если это согласовано владельцем.

### Этап 5. Наблюдение

Первые 72 часа:

- payment activation errors;
- duplicate charge conflicts;
- checkout-to-paid;
- refunds;
- OpenAI cost per payer;
- entitlement complaints;
- recurring status.

Нельзя повышать AI budget только для скрытия ошибок monetization rollout.

## 17. Метрики принятия решения

Первые целевые ориентиры:

- activation: не ниже 35%;
- paywall reach: не ниже 25% активированных;
- paywall-to-checkout: не ниже 10%;
- checkout-to-paid: не ниже 40%;
- activated-to-paid: 2–5%;
- estimated gross margin: не ниже 70%;
- payment activation failure: ниже 0.5%;
- duplicate entitlement incidents: 0;
- пользовательские потери слов/прогресса: 0.

Это диагностические ориентиры, не обещание выручки.

Изменение цены не откатывать только из-за низкого числа покупок, пока не разделены:

- недостаток входящего трафика;
- низкая activation;
- низкий paywall reach;
- низкий checkout conversion;
- платежная техническая ошибка;
- слабая ценность конкретного сценария.

## 18. Definition of Done

Задача завершена только если:

1. V2 plans созданы как новые immutable catalog entries.
2. В bot, Mini App и website отображаются одинаковые Stars prices.
3. Monthly invoice реально recurring.
4. Yearly invoice реально prepaid и явно так обозначен.
5. Initial charge, renewal, cancel, resume и refund имеют идемпотентные server-side transitions.
6. Legacy users не потеряли оплаченный срок.
7. Premium не обещает отсутствующие AI-функции.
8. Платные AI-операции имеют конечные лимиты.
9. Контекстный paywall не переносит пользователя в Settings.
10. Аналитика строит достоверную уникальную воронку.
11. Unit-economics report объединяет Stars reward estimate и AI cost.
12. Все relevant backend/frontend tests прошли.
13. Frontend build и Django deploy checks прошли.
14. Production smoke выполнен на `vocabume.k1prod.com`.
15. Проверены реальные mobile viewports.
16. Выполнен хотя бы один реалистичный payment flow с созданием данных.
17. Проверены media/audio bytes, Premium gates и отсутствие неожиданных background generation jobs.
18. После deploy подтвержден live asset hash.

## 19. Вне scope

В это ТЗ не входят:

- Stripe или карточные web-платежи;
- Apple/Google native applications;
- отдельная email/password identity;
- семейный или командный план;
- promo codes;
- affiliate/referral program;
- 90-day Relocation Pass до выполнения критериев раздела 3.5;
- динамическое региональное ценообразование;
- AI-диалоги как пользовательская функция;
- новый платный TTS provider;
- удаление legacy billing данных;
- автоматический upgrade/downgrade с prorating.

## 20. Рекомендуемая декомпозиция

1. `M2-01` — analytics schema и baseline reports.
2. `M2-02` — contextual paywall backend contract.
3. `M2-03` — `PremiumPaywall` и новый onboarding moment.
4. `M2-04` — finite entitlement/cost guards.
5. `M2-05` — immutable V2 catalog и data migrations.
6. `M2-06` — PaymentTransaction ledger.
7. `M2-07` — recurring initial/renewal handling.
8. `M2-08` — cancel/resume/refund.
9. `M2-09` — bot `/subscribe` V2.
10. `M2-10` — frontend billing management.
11. `M2-11` — OpenAPI и documentation update.
12. `M2-12` — backend/frontend/E2E quality gate.
13. `M2-13` — test-user production smoke.
14. `M2-14` — general rollout и 72-hour monitoring.

Задачи `M2-05`–`M2-08` нельзя выкатывать частично без backward-compatible renewal handler. Задача `M2-14` запрещена до прохождения `M2-12` и `M2-13`.

## 21. Внешние протокольные и рыночные источники

- Telegram Bot Payments for Digital Goods:
  <https://core.telegram.org/bots/payments-stars>
- Telegram Star subscriptions:
  <https://core.telegram.org/api/subscriptions>
- Telegram Bot API:
  <https://core.telegram.org/bots/api>
- Telegram Bot Platform Developer Terms, включая reward-value Stars:
  <https://telegram.org/tos/bot-developers>
- RevenueCat State of Subscription Apps 2026 — ориентиры, не SLA:
  <https://www.revenuecat.com/state-of-subscription-apps>

При расхождении документации Telegram с данным ТЗ протокольное поведение Telegram
имеет приоритет, а изменение реализации должно быть отдельно зафиксировано в
решении/ADR и тестах.
