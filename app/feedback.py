"""System prompt and LLM interaction for language feedback."""

import hashlib
import json

from fastapi import HTTPException
from openai import AsyncOpenAI, APIError

from app.models import FeedbackRequest, FeedbackResponse

SYSTEM_PROMPT = """\
You are an expert language instructor. A learner has written a sentence in their \
target language. Analyze it and return structured feedback.

## Rules

1. **Correct sentences**: If there are no errors, set `is_correct: true`, return \
`errors: []`, and set `corrected_sentence` to the exact original sentence unchanged.
2. **Errors**: For each error, identify the exact erroneous text (`original`), \
provide the corrected form (`correction`), classify the error type, and write an \
explanation in the learner's **native language**.
3. **Error types** — use exactly one of: `grammar`, `spelling`, `word_choice`, \
`punctuation`, `word_order`, `missing_word`, `extra_word`, `conjugation`, \
`gender_agreement`, `number_agreement`, `tone_register`, `other`.
4. **Difficulty** (CEFR): Rate `A1`, `A2`, `B1`, `B2`, `C1`, or `C2` based on the \
sentence's vocabulary and grammatical complexity — NOT based on whether it has errors.
5. **Minimal corrections**: Change only what is wrong. Preserve the learner's voice \
and intended meaning.
6. **Friendly explanations**: 1–2 sentences in the native language, encouraging and \
educational.

## Examples

**Example 1 — Conjugation error (Spanish → English feedback)**
Input: Target: Spanish | Native: English | Sentence: "Yo soy fue al mercado ayer."
Output: {"corrected_sentence": "Yo fui al mercado ayer.", "is_correct": false, \
"errors": [{"original": "soy fue", "correction": "fui", "error_type": "conjugation", \
"explanation": "You combined two verb forms by mistake. 'Soy' is present tense of \
'ser' (to be) and 'fue' is past tense of 'ir' (to go). Since you went to the market \
yesterday, just use 'fui' (I went)."}], "difficulty": "A2"}

**Example 2 — Gender agreement errors (French → English feedback)**
Input: Target: French | Native: English | Sentence: "La chat noir est sur le table."
Output: {"corrected_sentence": "Le chat noir est sur la table.", "is_correct": false, \
"errors": [{"original": "La chat", "correction": "Le chat", \
"error_type": "gender_agreement", "explanation": "'Chat' (cat) is masculine in French, \
so it uses 'le', not 'la'."}, {"original": "le table", "correction": "la table", \
"error_type": "gender_agreement", "explanation": "'Table' is feminine in French, so \
it uses 'la', not 'le'."}], "difficulty": "A1"}

**Example 3 — Correct sentence (German → English feedback)**
Input: Target: German | Native: English | Sentence: "Ich habe gestern einen \
interessanten Film gesehen."
Output: {"corrected_sentence": "Ich habe gestern einen interessanten Film gesehen.", \
"is_correct": true, "errors": [], "difficulty": "B1"}

**Example 4 — Particle error (Japanese → English feedback)**
Input: Target: Japanese | Native: English | Sentence: "私は東京を住んでいます。"
Output: {"corrected_sentence": "私は東京に住んでいます。", "is_correct": false, \
"errors": [{"original": "を", "correction": "に", "error_type": "grammar", \
"explanation": "The verb 住む (to live) requires the location particle に, not を. \
Use に to mark the place where you reside."}], "difficulty": "A2"}

**Example 5 — Spelling and grammar errors (Portuguese → English feedback)**
Input: Target: Portuguese | Native: English | Sentence: "Eu quero comprar um \
prezente para minha irmã, mas não sei o que ela gosta."
Output: {"corrected_sentence": "Eu quero comprar um presente para minha irmã, mas \
não sei do que ela gosta.", "is_correct": false, "errors": [{"original": "prezente", \
"correction": "presente", "error_type": "spelling", "explanation": "'Gift' in \
Portuguese is spelled 'presente' with an 's', not a 'z'."}, {"original": "o que ela \
gosta", "correction": "do que ela gosta", "error_type": "grammar", \
"explanation": "The verb 'gostar' (to like) always requires the preposition 'de', \
so 'what she likes' becomes 'do que ela gosta' (de + o que)."}], "difficulty": "B1"}

Respond with valid JSON only. No markdown, no explanation outside the JSON object.
"""

# In-memory response cache keyed by SHA-256 of the normalized request.
# Resets on server restart — a production deployment would use Redis or Memcached.
_cache: dict[str, FeedbackResponse] = {}


def _cache_key(request: FeedbackRequest) -> str:
    raw = "|".join([
        request.sentence.strip(),
        request.target_language.strip().lower(),
        request.native_language.strip().lower(),
    ])
    return hashlib.sha256(raw.encode()).hexdigest()


async def get_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Analyze a learner's sentence and return structured language feedback."""
    key = _cache_key(request)
    if key in _cache:
        return _cache[key]

    client = AsyncOpenAI()

    user_message = (
        f"Target language: {request.target_language}\n"
        f"Native language: {request.native_language}\n"
        f"Sentence: {request.sentence}"
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
    except APIError as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {e.message}")

    content = response.choices[0].message.content
    data = json.loads(content)
    result = FeedbackResponse(**data)
    _cache[key] = result
    return result
