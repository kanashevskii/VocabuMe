import { TIMEZONE_OPTIONS } from "../constants";

const REMINDER_INTERVAL_OPTIONS = [
  { value: 1, label: "Каждый день" },
  { value: 2, label: "Раз в 2 дня" },
  { value: 3, label: "Раз в 3 дня" },
  { value: 5, label: "Раз в 5 дней" },
  { value: 7, label: "Раз в неделю" },
  { value: 14, label: "Раз в 2 недели" },
  { value: 30, label: "Раз в 30 дней" },
];
const REMINDER_TIME_OPTIONS = Array.from({ length: 24 * 4 }, (_, index) => {
  const hours = String(Math.floor(index / 4)).padStart(2, "0");
  const minutes = String((index % 4) * 15).padStart(2, "0");
  const value = `${hours}:${minutes}`;

  return { value, label: value };
});

export default function SettingsScreen({
  onChange,
  onDeleteAvatar,
  onUploadAvatar,
  onSave,
  settings,
  uploadingAvatar,
}) {
  if (!settings) {
    return null;
  }

  const timezoneOptions = TIMEZONE_OPTIONS.some(
    (item) => item.value === settings.reminder_timezone,
  )
    ? TIMEZONE_OPTIONS
    : [
        ...TIMEZONE_OPTIONS,
        {
          value: settings.reminder_timezone,
          label: settings.reminder_timezone,
        },
      ];
  const studiedLanguageOptions =
    settings.available_studied_languages || [
      { code: "en", label: "Английский" },
      { code: "ka", label: "Грузинский" },
    ];
  const reminderIntervalOptions = REMINDER_INTERVAL_OPTIONS.some(
    (item) => item.value === settings.reminder_interval_days,
  )
    ? REMINDER_INTERVAL_OPTIONS
    : [
        ...REMINDER_INTERVAL_OPTIONS,
        {
          value: settings.reminder_interval_days,
          label: `Раз в ${settings.reminder_interval_days} дн.`,
        },
      ];
  const georgianDisplayModeOptions =
    settings.georgian_display_mode_options || [
      { code: "both", label: "Грузинский + латиница", recommended: true },
      { code: "native", label: "Только грузинский", recommended: false },
    ];
  const reminderTimeOptions = REMINDER_TIME_OPTIONS.some(
    (item) => item.value === settings.reminder_time,
  )
    ? REMINDER_TIME_OPTIONS
    : [
        ...REMINDER_TIME_OPTIONS,
        {
          value: settings.reminder_time,
          label: settings.reminder_time,
        },
      ];

  return (
    <section className="glass-card compact-section">
      <p className="overline">Настройки</p>
      <h3>Настройки ⚙️</h3>
      <form className="settings-grid" onSubmit={onSave}>
        <label>
          <span>Аватар профиля</span>
          <small>JPG, PNG или WEBP. До 5 MB. После загрузки сожмём в WEBP.</small>
          <div className="avatar-settings-row">
            {settings.avatar_url ? (
              <img className="settings-avatar-preview" src={settings.avatar_url} alt="Аватар профиля" />
            ) : (
              <div className="settings-avatar-preview settings-avatar-placeholder">
                <span>🙂</span>
              </div>
            )}
            <div className="stack-form">
              <input
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) {
                    onUploadAvatar(file);
                  }
                  event.target.value = "";
                }}
                disabled={uploadingAvatar}
              />
              {settings.avatar_url ? (
                <button
                  className="secondary-button"
                  type="button"
                  onClick={onDeleteAvatar}
                  disabled={uploadingAvatar}
                >
                  Удалить аватар
                </button>
              ) : null}
            </div>
          </div>
        </label>
        <label>
          <span>Изучаемый язык</span>
          <small>Выбери язык, который изучаешь сейчас. Прогресс хранится отдельно.</small>
          <select
            value={settings.active_studied_language}
            onChange={(event) =>
              onChange("active_studied_language", event.target.value)
            }
          >
            {studiedLanguageOptions.map((item) => (
              <option key={item.code} value={item.code}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        {settings.active_studied_language === "ka" ? (
          <label>
            <span>Показ грузинского</span>
            <small>
              Для старта рекомендуем показывать и грузинское письмо, и латиницу.
            </small>
            <select
              value={settings.georgian_display_mode}
              onChange={(event) =>
                onChange("georgian_display_mode", event.target.value)
              }
            >
              {georgianDisplayModeOptions.map((item) => (
                <option key={item.code} value={item.code}>
                  {item.label}{item.recommended ? " (рекомендуется)" : ""}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        <label>
          <span>Упражнений до статуса "выучено"</span>
          <small>
            Сколько разных упражнений нужно выполнить, чтобы слово стало
            выученным.
          </small>
          <select
            value={settings.exercise_goal}
            onChange={(event) =>
              onChange("exercise_goal", Number(event.target.value))
            }
          >
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4">4</option>
            <option value="5">5</option>
          </select>
        </label>
        <label>
          <span>Вопросов за прогон</span>
          <small>
            Максимум заданий за один прогон практики без повторов слов.
          </small>
          <input
            type="number"
            min="1"
            max="50"
            value={settings.session_question_limit}
            onChange={(event) =>
              onChange("session_question_limit", Number(event.target.value))
            }
          />
        </label>
        <label>
          <span>Дней до повторения</span>
          <input
            type="number"
            min="1"
            max="365"
            value={settings.days_before_review}
            onChange={(event) =>
              onChange("days_before_review", Number(event.target.value))
            }
          />
        </label>
        <label>
          <span>Как часто напоминать</span>
          <small>Выбери понятный ритм, например каждый день или раз в неделю.</small>
          <select
            value={settings.reminder_interval_days}
            onChange={(event) =>
              onChange("reminder_interval_days", Number(event.target.value))
            }
          >
            {reminderIntervalOptions.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Время напоминания</span>
          <small>24-часовой формат.</small>
          <select
            value={settings.reminder_time}
            onChange={(event) => onChange("reminder_time", event.target.value)}
          >
            {reminderTimeOptions.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Часовой пояс</span>
          <select
            value={settings.reminder_timezone}
            onChange={(event) =>
              onChange("reminder_timezone", event.target.value)
            }
          >
            {timezoneOptions.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        <label className="toggle-row">
          <input
            type="checkbox"
            checked={settings.reminder_enabled}
            onChange={(event) =>
              onChange("reminder_enabled", event.target.checked)
            }
          />
          <span>Включить напоминания</span>
        </label>
        <button className="primary-button" type="submit">
          Сохранить настройки
        </button>
      </form>
    </section>
  );
}
