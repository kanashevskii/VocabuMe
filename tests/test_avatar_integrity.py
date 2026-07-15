from io import BytesIO

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image

from vocab.models import TelegramUser
from vocab.services import save_user_avatar


def _png_upload() -> SimpleUploadedFile:
    image = Image.new("RGB", (64, 64), color=(20, 40, 120))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return SimpleUploadedFile("avatar.png", buffer.getvalue(), content_type="image/png")


@pytest.mark.django_db
def test_invalid_avatar_replacement_preserves_existing_file(monkeypatch, tmp_path):
    avatar_dir = tmp_path / "profile_avatars"
    monkeypatch.setattr("vocab.services.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("vocab.services.PROFILE_AVATAR_DIR", avatar_dir)
    user = TelegramUser.objects.create(chat_id=9042, username="avatar-test")
    save_user_avatar(user, _png_upload())
    user.refresh_from_db()
    previous_path = tmp_path / user.avatar_path
    previous_bytes = previous_path.read_bytes()

    invalid_upload = SimpleUploadedFile(
        "avatar.png", b"not-an-image", content_type="image/png"
    )
    with pytest.raises(ValueError, match="Не удалось обработать изображение"):
        save_user_avatar(user, invalid_upload)

    user.refresh_from_db()
    assert user.avatar_path == f"profile_avatars/user_{user.id}.webp"
    assert previous_path.read_bytes() == previous_bytes
