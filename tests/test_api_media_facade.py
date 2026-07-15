from vocab import views
from vocab.api import media


def test_views_keep_legacy_media_exports():
    assert views.word_audio is media.word_audio
    assert views.word_audio_prepare is media.word_audio_prepare
    assert views.alphabet_audio is media.alphabet_audio
    assert views.alphabet_audio_prepare is media.alphabet_audio_prepare
