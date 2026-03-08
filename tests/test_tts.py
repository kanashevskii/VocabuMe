from vocab.tts import get_audio_path, normalize_tts_language


def test_normalize_tts_language_detects_georgian_script():
    assert normalize_tts_language("გამარჯობა") == "ka"
    assert normalize_tts_language("hello") == "en"


def test_audio_path_is_language_aware_for_same_text():
    english_path = get_audio_path("bank", language_code="en")
    georgian_path = get_audio_path("bank", language_code="ka")

    assert english_path != georgian_path
    assert "en_bank_" in english_path
    assert "ka_bank_" in georgian_path


def test_audio_path_for_georgian_words_is_not_collapsed():
    first_path = get_audio_path("გამარჯობა", language_code="ka")
    second_path = get_audio_path("მადლობა", language_code="ka")

    assert first_path != second_path
