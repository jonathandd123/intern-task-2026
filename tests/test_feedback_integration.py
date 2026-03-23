"""Integration tests -- require OPENAI_API_KEY to be set.

Run with: pytest tests/test_feedback_integration.py -v

These tests make real API calls. Skip them in CI or when no key is available.
"""

import os

import pytest
from app.feedback import get_feedback
from app.models import FeedbackRequest

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set -- skipping integration tests",
)

VALID_ERROR_TYPES = {
    "grammar",
    "spelling",
    "word_choice",
    "punctuation",
    "word_order",
    "missing_word",
    "extra_word",
    "conjugation",
    "gender_agreement",
    "number_agreement",
    "tone_register",
    "other",
}
VALID_DIFFICULTIES = {"A1", "A2", "B1", "B2", "C1", "C2"}


@pytest.mark.asyncio
async def test_spanish_error():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert result.difficulty in VALID_DIFFICULTIES
    for error in result.errors:
        assert error.error_type in VALID_ERROR_TYPES
        assert len(error.explanation) > 0


@pytest.mark.asyncio
async def test_correct_german():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Ich habe gestern einen interessanten Film gesehen.",
            target_language="German",
            native_language="English",
        )
    )
    assert result.is_correct is True
    assert result.errors == []
    assert result.difficulty in VALID_DIFFICULTIES


@pytest.mark.asyncio
async def test_french_gender_errors():
    result = await get_feedback(
        FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1


@pytest.mark.asyncio
async def test_japanese_particle():
    result = await get_feedback(
        FeedbackRequest(
            sentence="私は東京を住んでいます。",
            target_language="Japanese",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert any("に" in e.correction for e in result.errors)


@pytest.mark.asyncio
async def test_korean_particle_error():
    # 을 (object marker) should be 에 (location marker) with 살다 (to live)
    result = await get_feedback(
        FeedbackRequest(
            sentence="나는 서울을 살아요.",
            target_language="Korean",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert result.difficulty in VALID_DIFFICULTIES
    for error in result.errors:
        assert error.error_type in VALID_ERROR_TYPES
        assert len(error.explanation) > 0


@pytest.mark.asyncio
async def test_russian_gender_agreement():
    # красивый (masc.) should agree with девушка (fem.) → красивую девушку in accusative
    result = await get_feedback(
        FeedbackRequest(
            sentence="Я вижу красивый девушка.",
            target_language="Russian",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert result.difficulty in VALID_DIFFICULTIES


@pytest.mark.asyncio
async def test_chinese_missing_measure_word():
    # 一苹果 is missing the measure word 个 → 一个苹果
    result = await get_feedback(
        FeedbackRequest(
            sentence="我有一苹果。",
            target_language="Chinese",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert any("个" in e.correction for e in result.errors)


@pytest.mark.asyncio
async def test_correct_japanese():
    # Grammatically correct sentence -- should return no errors
    result = await get_feedback(
        FeedbackRequest(
            sentence="私は毎朝コーヒーを飲みます。",
            target_language="Japanese",
            native_language="English",
        )
    )
    assert result.is_correct is True
    assert result.errors == []
    assert result.difficulty in VALID_DIFFICULTIES


@pytest.mark.asyncio
async def test_portuguese_spelling_error():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Eu quero comprar um prezente para minha irmã.",
            target_language="Portuguese",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert any("presente" in e.correction for e in result.errors)
