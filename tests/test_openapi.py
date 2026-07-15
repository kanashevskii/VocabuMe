import pytest


@pytest.mark.django_db
def test_openapi_schema_is_public_and_describes_audio_preparation(client):
    response = client.get("/api/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert schema["openapi"] == "3.1.0"
    assert "/api/audio/{word_id}/prepare" in schema["paths"]
    assert "TelegramInitData" in schema["components"]["securitySchemes"]


@pytest.mark.django_db
def test_swagger_ui_page_is_public_and_pinned(client):
    response = client.get("/api/docs")

    assert response.status_code == 200
    assert b"swagger-ui-dist@5.17.14" in response.content
    assert b"/api/openapi.json" in response.content
