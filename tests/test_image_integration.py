from types import SimpleNamespace

from vocab.integrations.images import compute_image_cache_path, get_image_urls


def test_image_integration_builds_deterministic_cache_path_and_provider_urls():
    word = SimpleNamespace(
        id=7,
        word="arrive",
        translation="прибывать",
        example="We arrive tomorrow.",
        part_of_speech="verb",
    )

    assert compute_image_cache_path(word).name == "7_arrive.jpg"
    urls = get_image_urls(word, seed=42)

    assert urls[-1] == "https://picsum.photos/seed/42/1280/720"
    assert any("arrive" in url for url in urls)
