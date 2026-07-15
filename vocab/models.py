from django.db import models
from django.db.models import Q
from datetime import time
import secrets
import uuid

STUDIED_LANGUAGE_CHOICES = (
    ("en", "English"),
    ("ka", "Georgian"),
)
DEFAULT_STUDIED_LANGUAGE = "en"
GEORGIAN_DISPLAY_MODE_CHOICES = (
    ("both", "Georgian + Latin"),
    ("native", "Georgian only"),
)
DEFAULT_GEORGIAN_DISPLAY_MODE = "both"
WORD_PRIORITY_CHOICES = (
    ("new_first", "New first"),
    ("old_first", "Old first"),
)
DEFAULT_WORD_PRIORITY = "new_first"
SUBSCRIPTION_STATUS_CHOICES = (
    ("pending", "Pending"),
    ("active", "Active"),
    ("expired", "Expired"),
    ("cancelled", "Cancelled"),
)
PAYMENT_ATTEMPT_STATUS_CHOICES = (
    ("pending", "Pending"),
    ("paid", "Paid"),
    ("failed", "Failed"),
    ("cancelled", "Cancelled"),
)


def generate_web_login_token() -> str:
    return secrets.token_urlsafe(32)


class TelegramUser(models.Model):
    chat_id = models.BigIntegerField(unique=True)
    username = models.CharField(max_length=255, null=True, blank=True)
    custom_avatar_url = models.URLField(max_length=500, blank=True, default="")
    avatar_path = models.CharField(max_length=500, blank=True, default="")
    avatar_updated_at = models.DateTimeField(null=True, blank=True)
    email = models.EmailField(unique=True, null=True, blank=True)
    password_hash = models.CharField(max_length=255, blank=True, default="")
    auth_provider = models.CharField(max_length=20, default="telegram")
    has_selected_studied_language = models.BooleanField(default=False)
    active_studied_language = models.CharField(
        max_length=10,
        choices=STUDIED_LANGUAGE_CHOICES,
        default=DEFAULT_STUDIED_LANGUAGE,
    )
    has_selected_georgian_display_mode = models.BooleanField(default=False)
    georgian_display_mode = models.CharField(
        max_length=20,
        choices=GEORGIAN_DISPLAY_MODE_CHOICES,
        default=DEFAULT_GEORGIAN_DISPLAY_MODE,
    )
    word_priority = models.CharField(
        max_length=20,
        choices=WORD_PRIORITY_CHOICES,
        default=DEFAULT_WORD_PRIORITY,
    )
    has_completed_onboarding = models.BooleanField(default=False)
    repeat_threshold = models.PositiveIntegerField(default=4)
    session_question_limit = models.PositiveIntegerField(default=12)
    enable_review_old_words = models.BooleanField(default=True)
    days_before_review = models.PositiveIntegerField(default=30)
    listening_paused_until = models.DateTimeField(null=True, blank=True)
    speaking_paused_until = models.DateTimeField(null=True, blank=True)

    reminder_enabled = models.BooleanField(default=False)
    reminder_time = models.TimeField(default=time(8, 0))
    reminder_interval_days = models.PositiveIntegerField(default=1)
    last_reminder_sent_at = models.DateField(null=True, blank=True)
    reminder_timezone = models.CharField(max_length=50, default="UTC")

    joined_at = models.DateField(auto_now_add=True)
    total_study_days = models.PositiveIntegerField(default=0)
    consecutive_days = models.PositiveIntegerField(default=0)
    last_study_date = models.DateField(null=True, blank=True)
    irregular_correct = models.PositiveIntegerField(default=0)
    practice_correct = models.PositiveIntegerField(default=0)
    listening_correct = models.PositiveIntegerField(default=0)
    speaking_correct = models.PositiveIntegerField(default=0)
    review_correct = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f"{self.username or self.email or 'User'} ({self.chat_id})"


class UserCourseProgress(models.Model):
    user = models.ForeignKey(TelegramUser, on_delete=models.CASCADE)
    course_code = models.CharField(
        max_length=10,
        choices=STUDIED_LANGUAGE_CHOICES,
        default=DEFAULT_STUDIED_LANGUAGE,
    )
    total_study_days = models.PositiveIntegerField(default=0)
    consecutive_days = models.PositiveIntegerField(default=0)
    last_study_date = models.DateField(null=True, blank=True)
    irregular_correct = models.PositiveIntegerField(default=0)
    practice_correct = models.PositiveIntegerField(default=0)
    listening_correct = models.PositiveIntegerField(default=0)
    speaking_correct = models.PositiveIntegerField(default=0)
    review_correct = models.PositiveIntegerField(default=0)
    total_points = models.PositiveIntegerField(default=0)

    class Meta:
        unique_together = ("user", "course_code")

    def __str__(self):
        return f"{self.user} [{self.course_code}]"


class UserStudyDay(models.Model):
    """Auditable, qualified learning activity for one course and local date."""

    user = models.ForeignKey(TelegramUser, on_delete=models.CASCADE)
    course_code = models.CharField(
        max_length=10,
        choices=STUDIED_LANGUAGE_CHOICES,
        default=DEFAULT_STUDIED_LANGUAGE,
    )
    study_date = models.DateField()
    correct_answers = models.PositiveIntegerField(default=0)
    streak_qualified_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["user", "course_code", "study_date"],
                name="unique_user_course_study_day",
            )
        ]
        indexes = [
            models.Index(
                fields=["user", "course_code", "-study_date"],
                name="vocab_study_day_idx",
            )
        ]

    def __str__(self):
        return f"{self.user} [{self.course_code}] {self.study_date}"


class UserDailyEntitlementUsage(models.Model):
    user = models.ForeignKey(TelegramUser, on_delete=models.CASCADE)
    usage_date = models.DateField()
    new_items_added = models.PositiveIntegerField(default=0)
    extra_image_regenerations = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "usage_date")
        ordering = ["-usage_date", "-id"]

    def __str__(self):
        return f"{self.user} [{self.usage_date}]"


class SubscriptionPlan(models.Model):
    code = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=100)
    billing_period = models.CharField(max_length=20)
    currency = models.CharField(max_length=10, default="USD")
    price_amount = models.DecimalField(max_digits=10, decimal_places=2)
    duration_days = models.PositiveIntegerField(default=30)
    is_active = models.BooleanField(default=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["price_amount", "id"]

    def __str__(self):
        return f"{self.code} ({self.price_amount} {self.currency})"


class UserSubscription(models.Model):
    user = models.ForeignKey(TelegramUser, on_delete=models.CASCADE)
    plan = models.ForeignKey(SubscriptionPlan, on_delete=models.PROTECT)
    status = models.CharField(
        max_length=20, choices=SUBSCRIPTION_STATUS_CHOICES, default="pending"
    )
    started_at = models.DateTimeField(null=True, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    activated_at = models.DateTimeField(null=True, blank=True)
    cancelled_at = models.DateTimeField(null=True, blank=True)
    source = models.CharField(max_length=20, default="telegram")
    invoice_payload = models.CharField(max_length=255, blank=True, default="")
    telegram_payment_charge_id = models.CharField(
        max_length=255, blank=True, default=""
    )
    provider_payment_charge_id = models.CharField(
        max_length=255, blank=True, default=""
    )
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self):
        return f"{self.user} -> {self.plan.code} [{self.status}]"


class PaymentAttempt(models.Model):
    user = models.ForeignKey(TelegramUser, on_delete=models.CASCADE)
    plan = models.ForeignKey(SubscriptionPlan, on_delete=models.PROTECT)
    provider = models.CharField(max_length=20, default="telegram")
    status = models.CharField(
        max_length=20, choices=PAYMENT_ATTEMPT_STATUS_CHOICES, default="pending"
    )
    invoice_payload = models.CharField(max_length=255, unique=True)
    invoice_link = models.URLField(max_length=1000, blank=True, default="")
    amount_minor = models.PositiveIntegerField(default=0)
    currency = models.CharField(max_length=10, default="USD")
    paid_at = models.DateTimeField(null=True, blank=True)
    failed_at = models.DateTimeField(null=True, blank=True)
    cancelled_at = models.DateTimeField(null=True, blank=True)
    telegram_payment_charge_id = models.CharField(
        max_length=255, blank=True, default=""
    )
    provider_payment_charge_id = models.CharField(
        max_length=255, blank=True, default=""
    )
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at", "-id"]
        constraints = [
            models.UniqueConstraint(
                fields=["telegram_payment_charge_id"],
                condition=~Q(telegram_payment_charge_id=""),
                name="unique_nonempty_telegram_payment_charge",
            ),
            models.UniqueConstraint(
                fields=["provider_payment_charge_id"],
                condition=~Q(provider_payment_charge_id=""),
                name="unique_nonempty_provider_payment_charge",
            ),
        ]

    def __str__(self):
        return f"{self.user} -> {self.plan.code} [{self.status}]"


class VocabularyItem(models.Model):
    user = models.ForeignKey(TelegramUser, on_delete=models.CASCADE)
    course_code = models.CharField(
        max_length=10,
        choices=STUDIED_LANGUAGE_CHOICES,
        default=DEFAULT_STUDIED_LANGUAGE,
    )
    word = models.CharField(max_length=255)  # исходное
    normalized_word = models.CharField(max_length=255)
    translation = models.CharField(max_length=255)
    transcription = models.CharField(max_length=255)
    example = models.TextField()
    example_translation = models.TextField(blank=True, default="")
    correct_count = models.IntegerField(default=0)
    completed_exercise_types = models.JSONField(default=list, blank=True)
    is_learned = models.BooleanField(default=False)
    learned_at = models.DateTimeField(null=True, blank=True)
    image_regeneration_count = models.PositiveIntegerField(default=0)
    image_generation_version = models.PositiveIntegerField(default=0)
    image_generation_in_progress = models.BooleanField(default=False)
    part_of_speech = models.CharField(max_length=50, default="unknown")
    image_path = models.CharField(max_length=500, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["user", "course_code", "normalized_word"],
                name="unique_user_course_normalized_word",
            )
        ]
        indexes = [
            models.Index(
                fields=["user", "course_code", "-updated_at", "-id"],
                name="vocab_word_list_idx",
            ),
            models.Index(
                fields=["user", "course_code", "is_learned", "-created_at", "-id"],
                name="vocab_word_new_idx",
            ),
            models.Index(
                fields=["user", "course_code", "is_learned", "updated_at", "id"],
                name="vocab_word_review_idx",
            ),
        ]

    def __str__(self):
        return f"{self.word} ({self.user})"


class LearningSession(models.Model):
    user_id = models.BigIntegerField(unique=True)
    current_index = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"Session for {self.user_id}"


class IssuedLearningQuestion(models.Model):
    """Server-side source of truth for a Mini App learning attempt."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(TelegramUser, on_delete=models.CASCADE)
    item = models.ForeignKey(VocabularyItem, on_delete=models.CASCADE)
    exercise_type = models.CharField(max_length=40)
    expires_at = models.DateTimeField()
    answered_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(
                fields=["user", "expires_at"], name="vocab_issue_user_id_d712f8_idx"
            )
        ]


class IssuedIrregularQuestion(models.Model):
    """One server-issued irregular-verb attempt, consumed exactly once."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(TelegramUser, on_delete=models.CASCADE)
    course_code = models.CharField(
        max_length=10,
        choices=STUDIED_LANGUAGE_CHOICES,
        default=DEFAULT_STUDIED_LANGUAGE,
    )
    verb_base = models.CharField(max_length=50)
    expires_at = models.DateTimeField()
    answered_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(
                fields=["user", "expires_at"], name="vocab_irregular_issue_idx"
            )
        ]


class OpenAIUsageEvent(models.Model):
    """Auditable cost estimate for one OpenAI request."""

    user = models.ForeignKey(
        TelegramUser, on_delete=models.SET_NULL, null=True, blank=True
    )
    operation = models.CharField(max_length=64)
    model = models.CharField(max_length=128)
    input_tokens = models.PositiveIntegerField(default=0)
    output_tokens = models.PositiveIntegerField(default=0)
    cached_input_tokens = models.PositiveIntegerField(default=0)
    image_count = models.PositiveSmallIntegerField(default=0)
    cost_microusd = models.PositiveBigIntegerField()
    usage_available = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(
                fields=["created_at"], name="vocab_openai_usage_created_idx"
            ),
            models.Index(
                fields=["user", "created_at"], name="vocab_openai_usage_user_idx"
            ),
        ]


BACKGROUND_JOB_STATUS_CHOICES = (
    ("queued", "Queued"),
    ("running", "Running"),
    ("succeeded", "Succeeded"),
    ("failed", "Failed"),
)


class BackgroundJob(models.Model):
    """Durable work item processed outside the Django request process."""

    kind = models.CharField(max_length=64)
    priority = models.PositiveSmallIntegerField(default=100)
    deduplication_key = models.CharField(max_length=255, unique=True)
    payload = models.JSONField(default=dict)
    status = models.CharField(
        max_length=16, choices=BACKGROUND_JOB_STATUS_CHOICES, default="queued"
    )
    attempts = models.PositiveSmallIntegerField(default=0)
    max_attempts = models.PositiveSmallIntegerField(default=3)
    run_after = models.DateTimeField()
    locked_at = models.DateTimeField(null=True, blank=True)
    last_error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(
                fields=["status", "priority", "run_after"],
                name="vocab_bgjob_queue_idx",
            )
        ]
        ordering = ["priority", "run_after", "id"]


class Achievement(models.Model):
    user = models.ForeignKey("TelegramUser", on_delete=models.CASCADE)
    course_code = models.CharField(
        max_length=10,
        choices=STUDIED_LANGUAGE_CHOICES,
        default=DEFAULT_STUDIED_LANGUAGE,
    )
    code = models.CharField(max_length=100)
    date_awarded = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "course_code", "code")

    def __str__(self):
        return f"{self.code} for {self.user.username or self.user.chat_id}"


class IrregularVerbProgress(models.Model):
    """Track user's progress for each irregular verb."""

    user = models.ForeignKey(TelegramUser, on_delete=models.CASCADE)
    course_code = models.CharField(
        max_length=10,
        choices=STUDIED_LANGUAGE_CHOICES,
        default=DEFAULT_STUDIED_LANGUAGE,
    )
    verb_base = models.CharField(max_length=50)
    correct_count = models.PositiveIntegerField(default=0)
    is_learned = models.BooleanField(default=False)

    class Meta:
        unique_together = ("user", "course_code", "verb_base")

    def __str__(self):
        return f"{self.verb_base} ({self.user})"


class WebLoginToken(models.Model):
    token = models.CharField(
        max_length=64, unique=True, default=generate_web_login_token
    )
    user = models.ForeignKey(
        TelegramUser, on_delete=models.CASCADE, null=True, blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    consumed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"WebLoginToken({self.token[:8]})"


class AddWordDraft(models.Model):
    user = models.ForeignKey(TelegramUser, on_delete=models.CASCADE)
    course_code = models.CharField(
        max_length=10,
        choices=STUDIED_LANGUAGE_CHOICES,
        default=DEFAULT_STUDIED_LANGUAGE,
    )
    source_text = models.CharField(max_length=255)
    word = models.CharField(max_length=255)
    normalized_word = models.CharField(max_length=255)
    translation = models.CharField(max_length=255, blank=True, default="")
    translation_confirmed = models.BooleanField(default=False)
    transcription = models.CharField(max_length=255, blank=True, default="")
    example = models.TextField(blank=True, default="")
    example_translation = models.TextField(blank=True, default="")
    part_of_speech = models.CharField(max_length=50, default="unknown")
    image_prompt = models.TextField(blank=True, default="")
    image_path = models.CharField(max_length=500, blank=True, default="")
    image_regeneration_count = models.PositiveIntegerField(default=0)
    image_generation_version = models.PositiveIntegerField(default=0)
    image_generation_in_progress = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at", "-id"]

    def __str__(self):
        return f"AddWordDraft({self.word} -> {self.translation})"


class AppErrorLog(models.Model):
    user = models.ForeignKey(
        TelegramUser, on_delete=models.SET_NULL, null=True, blank=True
    )
    category = models.CharField(max_length=50, default="server")
    level = models.CharField(max_length=20, default="error")
    message = models.TextField()
    path = models.CharField(max_length=255, blank=True, default="")
    method = models.CharField(max_length=10, blank=True, default="")
    status_code = models.PositiveIntegerField(null=True, blank=True)
    context = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-id"]
        indexes = [
            models.Index(fields=["created_at"], name="vocab_error_log_created_idx")
        ]

    def __str__(self):
        return f"{self.category}:{self.level} {self.path or '-'}"


class PackPreparedWord(models.Model):
    course_code = models.CharField(
        max_length=10,
        choices=STUDIED_LANGUAGE_CHOICES,
        default=DEFAULT_STUDIED_LANGUAGE,
    )
    pack_id = models.CharField(max_length=64)
    level_id = models.CharField(max_length=64)
    word = models.CharField(max_length=255)
    normalized_word = models.CharField(max_length=255)
    translation = models.CharField(max_length=255)
    transcription = models.CharField(max_length=255, blank=True, default="")
    example = models.TextField(blank=True, default="")
    example_translation = models.TextField(blank=True, default="")
    part_of_speech = models.CharField(max_length=50, default="unknown")
    image_path = models.CharField(max_length=500, blank=True, default="")
    image_generation_in_progress = models.BooleanField(default=False)
    last_failure_at = models.DateTimeField(null=True, blank=True)
    failure_count = models.PositiveIntegerField(default=0)
    prepared_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("course_code", "pack_id", "level_id", "normalized_word")
        ordering = ["course_code", "pack_id", "level_id", "word"]

    def __str__(self):
        return f"{self.pack_id}:{self.level_id}:{self.word}"
