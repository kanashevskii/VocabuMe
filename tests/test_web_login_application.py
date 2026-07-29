import pytest

from vocab.application import web_login
from vocab.models import TelegramUser
from vocab.services import consume_web_login_token as consume_web_login_token_facade


@pytest.mark.django_db
def test_unbound_login_token_remains_available_for_telegram_binding():
    token = web_login.create_web_login_token()
    user = TelegramUser.objects.create(chat_id=1002, username="tester")

    assert web_login.consume_web_login_token(token.token) is None
    assert web_login.bind_web_login_token(token.token, user) is not None
    assert web_login.consume_web_login_token(token.token) == user
    assert web_login.consume_web_login_token(token.token) is None


@pytest.mark.django_db
def test_services_keeps_web_login_consume_compatibility_facade():
    user = TelegramUser.objects.create(chat_id=1003, username="tester")
    token = web_login.create_web_login_token()
    web_login.bind_web_login_token(token.token, user)

    assert consume_web_login_token_facade(token.token) == user
