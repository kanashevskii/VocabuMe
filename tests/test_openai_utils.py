from vocab.openai_utils import CARD_IMAGE_STYLE_PROMPT


def test_card_image_style_prompt_has_brand_guardrails():
    prompt = CARD_IMAGE_STYLE_PROMPT.lower()

    assert "vocabume" in prompt
    assert "georgia relocation context" in prompt
    assert "no visible text" in prompt
    assert "logos" in prompt
    assert "watermarks" in prompt
    assert "full-bleed square scene" in prompt
    assert "no rounded card frame" in prompt
