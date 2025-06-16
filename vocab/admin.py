from django.contrib import admin
from .models import TelegramUser, VocabularyItem, Achievement, LearningSession

@admin.register(TelegramUser)
class TelegramUserAdmin(admin.ModelAdmin):
    list_display = (
        "chat_id",
        "username",
        "repeat_threshold",
        "reminder_enabled",
        "last_study_date",
        "irregular_correct",
    )
    search_fields = ("chat_id", "username")
    list_filter = ("reminder_enabled", "enable_review_old_words")

@admin.register(VocabularyItem)
class VocabularyItemAdmin(admin.ModelAdmin):
    list_display = ("word", "translation", "part_of_speech", "user", "is_learned", "correct_count", "created_at")
    list_filter = ("is_learned", "part_of_speech")
    search_fields = ("word", "translation", "example", "example_translation", "user__username")
    ordering = ("-created_at",)

@admin.register(Achievement)
class AchievementAdmin(admin.ModelAdmin):
    list_display = ("code", "user", "date_awarded")
    search_fields = ("code", "user__username")

@admin.register(LearningSession)
class LearningSessionAdmin(admin.ModelAdmin):
    list_display = ("user_id", "current_index", "is_active")
