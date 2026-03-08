from django.contrib import admin
from .models import (
    TelegramUser,
    UserCourseProgress,
    VocabularyItem,
    Achievement,
    LearningSession,
    IrregularVerbProgress,
    WebLoginToken,
)

@admin.register(TelegramUser)
class TelegramUserAdmin(admin.ModelAdmin):
    list_display = (
        "chat_id",
        "username",
        "active_studied_language",
        "repeat_threshold",
        "reminder_enabled",
        "last_study_date",
        "irregular_correct",
    )
    search_fields = ("chat_id", "username", "email")
    list_filter = (
        "active_studied_language",
        "reminder_enabled",
        "enable_review_old_words",
    )


@admin.register(UserCourseProgress)
class UserCourseProgressAdmin(admin.ModelAdmin):
    list_display = (
        "user",
        "course_code",
        "consecutive_days",
        "practice_correct",
        "listening_correct",
        "speaking_correct",
        "review_correct",
    )
    list_filter = ("course_code",)
    search_fields = ("user__username", "user__email", "user__chat_id")

@admin.register(VocabularyItem)
class VocabularyItemAdmin(admin.ModelAdmin):
    list_display = (
        "word",
        "translation",
        "course_code",
        "part_of_speech",
        "user",
        "is_learned",
        "correct_count",
        "created_at",
    )
    list_filter = ("course_code", "is_learned", "part_of_speech")
    search_fields = ("word", "translation", "example", "example_translation", "user__username")
    ordering = ("-created_at",)

@admin.register(Achievement)
class AchievementAdmin(admin.ModelAdmin):
    list_display = ("code", "course_code", "user", "date_awarded")
    list_filter = ("course_code",)
    search_fields = ("code", "user__username")

@admin.register(LearningSession)
class LearningSessionAdmin(admin.ModelAdmin):
    list_display = ("user_id", "current_index", "is_active")


@admin.register(IrregularVerbProgress)
class IrregularVerbProgressAdmin(admin.ModelAdmin):
    list_display = ("user", "course_code", "verb_base", "correct_count", "is_learned")
    list_filter = ("course_code", "is_learned")


@admin.register(WebLoginToken)
class WebLoginTokenAdmin(admin.ModelAdmin):
    list_display = ("token", "user", "created_at", "expires_at", "consumed_at")
    search_fields = ("token", "user__username", "user__chat_id")
