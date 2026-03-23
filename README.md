# Language Feedback API

An LLM-powered REST API that analyzes learner-written sentences and returns structured grammar feedback. Built with FastAPI and OpenAI.

## Design Decisions

### Model: `gpt-4o`

The example code used `gpt-4o-mini`, but I chose `gpt-4o` since accuracy is actually being scored. If there's a better model available, it just makes sense to use it.

At the end of the day, the model is the biggest factor in how good the corrections are. `gpt-4o` performs well across 8+ languages, including non-Latin scripts like Japanese, Korean, Russian, and Chinese. The cost is higher, but it's still manageable, especially with the optimizations below.

### Prompt Strategy

The system prompt is split into two parts:

**Rules block** — This has six clear rules that cover edge cases from the spec: handling correct sentences, classifying errors, CEFR levels, keeping edits minimal, and giving explanations in the user's native language. Being explicit here helps prevent weird corrections or made-up error types.

**Five few-shot examples** — There's one example per language (Spanish, French, German, Japanese, and Portuguese), each showing a different type of mistake. These examples help a lot with consistency. They show the model exactly what "good" feedback should look like across different languages and scripts.

I also noticed that without examples, the outputs were less consistent, so adding them made a big difference.

Temperature is set to `0.1` because this is a factual task. I want consistent, accurate corrections, not creative variation.

### Structured Output

I used `response_format={"type": "json_object"}` along with a defined JSON schema in the prompt to keep responses structured.

On top of that, Pydantic validates everything before returning it. So if the model outputs something malformed, it throws a `ValidationError` instead of silently passing bad data. Using `Literal` types for fields like `error_type` and `difficulty` also makes sure only valid values get through.

### Response Caching

If the same request comes in more than once, it returns a cached result instead of calling the API again.

The cache key is a SHA-256 hash of the normalized input (sentence, target language, and native language, case-insensitive). Right now it's just an in-memory dictionary, so it resets on restart. In a real deployment, I'd switch this to Redis with a TTL.

Even with this simple setup, it cuts down on cost since users often retry the same sentence after making small changes.

### Error Handling

If there's an OpenAI `APIError`, it gets caught and returned as an HTTP 502. That way the client still gets a clean JSON response instead of the server crashing.

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

- **Verification for unknown languages**: I don't speak all 8+ languages, so I verified correctness by cross-checking with grammar references and native speaker resources. The model handles the actual language understanding, and the tests make sure the structure and key corrections are right.
- **Cache scope**: The cache is per-process since it's in memory. That's fine for a single container, but scaling would require something like Redis.
- **Correct sentences**: If a sentence is correct, the API returns `is_correct: true`, keeps the sentence unchanged, and returns an empty errors array.
