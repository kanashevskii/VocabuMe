from django.db import models
from datetime import time


class TelegramUser(models.Model):
    chat_id = models.BigIntegerField(unique=True)
    username = models.CharField(max_length=255, null=True, blank=True)
    repeat_threshold = models.PositiveIntegerField(default=3)
    enable_review_old_words = models.BooleanField(default=True)
    days_before_review = models.PositiveIntegerField(default=30)

    reminder_enabled = models.BooleanField(default=False)
    reminder_time = models.TimeField(default=time(8, 0))
    reminder_interval_days = models.PositiveIntegerField(default=1)
    last_reminder_sent_at = models.DateField(null=True, blank=True)

    joined_at = models.DateField(auto_now_add=True)
    total_study_days = models.PositiveIntegerField(default=0)
    consecutive_days = models.PositiveIntegerField(default=0)
    last_study_date = models.DateField(null=True, blank=True)

    def __str__(self):
        return f"{self.username or 'User'} ({self.chat_id})"


class VocabularyItem(models.Model):
    user = models.ForeignKey(TelegramUser, on_delete=models.CASCADE)
    word = models.CharField(max_length=255)  # исходное
    normalized_word = models.CharField(max_length=255)
    translation = models.CharField(max_length=255)
    transcription = models.CharField(max_length=255)
    example = models.TextField()
    correct_count = models.IntegerField(default=0)
    is_learned = models.BooleanField(default=False)
    part_of_speech = models.CharField(max_length=50, default="unknown")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['user', 'normalized_word'], name='unique_user_normalized_word')
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
    code = models.CharField(max_length=100)
    date_awarded = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'code')

    def __str__(self):
        return f"{self.code} for {self.user.username or self.user.chat_id}"
