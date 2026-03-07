from django.urls import path

from . import views


urlpatterns = [
    path("", views.spa_index, name="spa-index"),
    path("api/app-config", views.app_config, name="app-config"),
    path("api/auth/me", views.auth_me, name="auth-me"),
    path("api/auth/logout", views.auth_logout, name="auth-logout"),
    path("api/auth/telegram/request-link", views.auth_request_link, name="auth-request-link"),
    path("api/auth/telegram/poll/<str:token>", views.auth_poll_link, name="auth-poll-link"),
    path("api/auth/telegram/widget", views.auth_telegram_widget, name="auth-telegram-widget"),
    path("api/auth/telegram/webapp", views.auth_telegram_webapp, name="auth-telegram-webapp"),
    path("api/dashboard", views.dashboard, name="dashboard"),
    path("api/words", views.words, name="words"),
    path("api/words/<int:word_id>", views.word_detail, name="word-detail"),
    path("api/settings", views.settings_view, name="settings"),
    path("api/study/cards", views.study_cards, name="study-cards"),
    path("api/study/answer", views.study_answer, name="study-answer"),
    path("api/practice/question", views.practice_question, name="practice-question"),
    path("api/practice/answer", views.practice_answer, name="practice-answer"),
    path("api/listening/question", views.listening_question, name="listening-question"),
    path("api/listening/answer", views.listening_answer, name="listening-answer"),
    path("api/audio/<int:word_id>", views.word_audio, name="word-audio"),
    path("api/irregular/list", views.irregular_list, name="irregular-list"),
    path("api/irregular/question", views.irregular_question, name="irregular-question"),
    path("api/irregular/answer", views.irregular_answer, name="irregular-answer"),
]
