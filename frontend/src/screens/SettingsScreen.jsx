import { TIMEZONE_OPTIONS } from "../constants";

export default function SettingsScreen({
  onChange,
  onSave,
  settings,
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
      { code: "en", label: "English" },
      { code: "ka", label: "Georgian" },
    ];

  return (
    <section className="glass-card compact-section">
      <p className="overline">Settings</p>
      <h3>Настройки ⚙️</h3>
      <form className="settings-grid" onSubmit={onSave}>
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
          <span>Интервал напоминаний</span>
          <input
            type="number"
            min="1"
            max="30"
            value={settings.reminder_interval_days}
            onChange={(event) =>
              onChange("reminder_interval_days", Number(event.target.value))
            }
          />
        </label>
        <label>
          <span>Время напоминания</span>
          <input
            type="time"
            value={settings.reminder_time}
            onChange={(event) => onChange("reminder_time", event.target.value)}
          />
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
