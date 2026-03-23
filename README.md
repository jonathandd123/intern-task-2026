# Language Feedback API

An LLM-powered REST API that analyzes learner-written sentences and returns structured
grammar feedback. Built with FastAPI and OpenAI.

## Design Decisions

### Model: `gpt-4o`

The example code used `gpt-4o-mini`. Ultimately, I decided to go with `gpt-4o` because accuracy is scored, and why wouldn't I use the better model
At the end of the day, the model is what impacts corrections the most.
right across 8+ languages, including non-Latin scripts (Japanese, Korean, Russian,
Chinese). The cost increase is real but manageable in the context of the other
optimizations below.

### Prompt Strategy

The system prompt has two components:

**Rules block** — Six numbered rules covering every edge case in the spec: correct
sentences, error classification, CEFR rating methodology, minimal-edit philosophy,
and native-language explanations. Explicit rules reduce hallucinated error types
and prevent the model from "correcting" things that aren't wrong.

**Five few-shot examples** — There's one example per language (Spanish, French, German,
Japanese, and Portuguese), each demonstrating a different error category. Few-shot
examples one of the most reliable ways to enforce output format and show the model what
"good" feedback looks like across scripts and writing systems. The examples are drawn
directly from the spec so the model sees the exact quality bar expected.

Temperature is set to `0.1` because this is a factual task, because of that of course we'd want consistent, accurate corrections, not creative variation.

### Structured Output

`response_format={"type": "json_object"}` combined with explicit JSON schema in the
system prompt gives reliable schema compliance. Pydantic validates the response on
the way out, so a malformed LLM response raises a `ValidationError` rather than
silently returning bad data. `Literal` types on `error_type` and `difficulty` mean
Pydantic rejects any value not in the allowed set.

### Response Caching

Identical requests return the cached result without hitting the API. The cache key is
a SHA-256 hash of the normalized request (sentence + target language + native language,
case-insensitive). This is an in-memory dict (resets on restart); a production
deployment would use Redis with a TTL. Even in-memory caching provides a meaningful
cost reduction in real use, where learners often resubmit the same sentence after
making edits.

### Error Handling

OpenAI `APIError` is caught and converted to an HTTP 502 so the caller gets a clean
JSON error rather than an unhandled exception.

## Running Locally

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# 4. Start the server
uvicorn app.main:app --reload

# 5. Try it
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"sentence": "Yo soy fue al mercado ayer.", "target_language": "Spanish", "native_language": "English"}'
```

## Running with Docker

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
docker compose up --build
```

The service starts on port 8000. The health check endpoint is `GET /health`.

## Running Tests

```bash
# Unit tests (no API key needed -- LLM is mocked)
pytest tests/test_feedback_unit.py tests/test_schema.py -v

# Integration tests (require OPENAI_API_KEY in .env)
pytest tests/test_feedback_integration.py -v

# All tests
pytest -v
```

## Assumptions

- **Verification for unknown languages**: Since I don't speak all 8+ tested languages,
  I verified feedback accuracy by cross-referencing with reference grammars and native
  speaker resources for each test case. The model handles the linguistic heavy lifting;
  the tests check that the output structure and key corrections are correct.
- **Cache scope**: In-memory caching is per-process. This is sufficient for a single
  Docker container; horizontal scaling would require a shared cache layer.
- **Correct sentences**: `is_correct: true` is returned with the original sentence
  unchanged and an empty errors array.
