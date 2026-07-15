from vocab import views
from vocab.api import images
from vocab.api import irregular
from vocab.api import learning
from vocab.api import media
from vocab.api import speaking


def test_views_keep_legacy_media_exports():
    assert views.word_audio is media.word_audio
    assert views.word_audio_prepare is media.word_audio_prepare
    assert views.alphabet_audio is media.alphabet_audio
    assert views.alphabet_audio_prepare is media.alphabet_audio_prepare


def test_views_keep_legacy_image_exports(client):
    assert views.word_image is images.word_image
    assert views.draft_image is images.draft_image
    assert client.get("/api/image/1").status_code == 401
    assert client.get("/api/draft-image/1").status_code == 401


def test_views_keep_legacy_irregular_exports(client):
    assert views.irregular_list is irregular.irregular_list
    assert views.irregular_question is irregular.irregular_question
    assert views.irregular_answer is irregular.irregular_answer
    assert client.get("/api/irregular/question").status_code == 401


def test_views_keep_legacy_learning_exports(client):
    assert views.learn_question is learning.learn_question
    assert views.learn_answer is learning.learn_answer
    assert views.study_cards is learning.study_cards
    assert views.study_answer is learning.study_answer
    assert (
        client.post(
            "/api/learn/question", data="{}", content_type="application/json"
        ).status_code
        == 401
    )


def test_views_keep_legacy_speaking_exports(client):
    assert views.speaking_question is speaking.speaking_question
    assert views.speaking_answer is speaking.speaking_answer
    assert client.get("/api/speaking/question").status_code == 401
