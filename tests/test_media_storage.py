import pytest

from vocab.media_storage import LocalMediaStorage


def test_local_media_storage_preserves_project_relative_paths(tmp_path, monkeypatch):
    storage = LocalMediaStorage()
    monkeypatch.setattr("vocab.media_storage.MEDIA_ROOT", tmp_path / "media")
    monkeypatch.setattr("vocab.media_storage.PROJECT_ROOT", tmp_path)
    image_path = storage.directory("card_images", create=True) / "word.jpg"
    image_path.write_bytes(b"image")

    assert storage.to_project_relative(image_path) == "media/card_images/word.jpg"
    assert (
        storage.resolve_existing(
            "media/card_images/word.jpg", allowed_kinds={"card_images"}
        )
        == image_path.resolve()
    )


def test_local_media_storage_rejects_paths_outside_allowed_media_roots(
    tmp_path, monkeypatch
):
    storage = LocalMediaStorage()
    monkeypatch.setattr("vocab.media_storage.MEDIA_ROOT", tmp_path / "media")
    monkeypatch.setattr("vocab.media_storage.PROJECT_ROOT", tmp_path)
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")

    assert storage.resolve_existing(outside, allowed_kinds={"card_images"}) is None

    with pytest.raises(ValueError, match="Unsupported media kind"):
        storage.directory("unknown")
