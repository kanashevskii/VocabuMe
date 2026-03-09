from django.db import models
from datetime import time
import secrets

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


def generate_web_login_token() -> str:
    return secrets.token_urlsafe(32)


class TelegramUser(models.Model):
    chat_id = models.BigIntegerField(unique=True)
    username = models.CharField(max_length=255, null=True, blank=True)
    custom_avatar_url = models.URLField(max_length=500, blank=True, default="")
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
    has_completed_onboarding = models.BooleanField(default=False)
    repeat_threshold = models.PositiveIntegerField(default=4)
    session_question_limit = models.PositiveIntegerField(default=12)
    enable_review_old_words = models.BooleanField(default=True)
    days_before_review = models.PositiveIntegerField(default=30)

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

    def __str__(self):
        return f"{self.word} ({self.user})"


class LearningSession(models.Model):
    user_id = models.BigIntegerField(unique=True)
    current_index = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"Session for {self.user_id}"


class Achievement(models.Model):
    user = models.ForeignKey('TelegramUser', on_delete=models.CASCADE)
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
    token = models.CharField(max_length=64, unique=True, default=generate_web_login_token)
    user = models.ForeignKey(TelegramUser, on_delete=models.CASCADE, null=True, blank=True)
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
    user = models.ForeignKey(TelegramUser, on_delete=models.SET_NULL, null=True, blank=True)
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
    prepared_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("course_code", "pack_id", "level_id", "normalized_word")
        ordering = ["course_code", "pack_id", "level_id", "word"]

    def __str__(self):
        return f"{self.pack_id}:{self.level_id}:{self.word}"
