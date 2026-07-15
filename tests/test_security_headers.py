import pytest

from core.security import CONTENT_SECURITY_POLICY, PERMISSIONS_POLICY


@pytest.mark.django_db
def test_public_response_has_telegram_compatible_security_headers(client):
    response = client.get("/api/app-config")

    assert response["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
    assert "https://telegram.org" in response["Content-Security-Policy"]
    assert "https://cdn.jsdelivr.net" in response["Content-Security-Policy"]
    assert response["Permissions-Policy"] == PERMISSIONS_POLICY
