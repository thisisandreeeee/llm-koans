import pytest
from fastapi.testclient import TestClient

from llm_koans import koans as K


def test_estimate_tokens_uses_a_simple_capacity_planning_heuristic():
    assert K.estimate_tokens("") == 0
    assert K.estimate_tokens("abcd") == 1
    assert K.estimate_tokens("abcde") == 2
    assert K.estimate_tokens("LLM deployment koans") == 5


def test_validate_generation_budget_reports_context_window_overflow():
    ok = K.validate_generation_budget(
        prompt="abcd efgh", max_new_tokens=4, context_window=8, reserved_tokens=2
    )
    assert ok == {
        "prompt_tokens": 2,
        "requested_tokens": 4,
        "reserved_tokens": 2,
        "total_tokens": 8,
        "fits": True,
        "overflow_tokens": 0,
    }

    too_large = K.validate_generation_budget(
        prompt="a" * 17, max_new_tokens=6, context_window=8, reserved_tokens=1
    )
    assert too_large["fits"] is False
    assert too_large["overflow_tokens"] == 4


def test_create_generation_app_serves_health_and_generation_endpoints():
    def fake_generate(prompt: str, max_new_tokens: int, temperature: float) -> str:
        return f"{prompt} :: {max_new_tokens} :: {temperature}"

    app = K.create_generation_app(fake_generate, model_name="tiny-koan")
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model": "tiny-koan"}

    response = client.post(
        "/generate",
        json={"prompt": "hello", "max_new_tokens": 3, "temperature": 0.2},
    )
    assert response.status_code == 200
    assert response.json() == {"model": "tiny-koan", "text": "hello :: 3 :: 0.2"}


def test_create_generation_app_rejects_empty_prompts_before_the_model_call():
    calls = []

    def fake_generate(prompt: str, max_new_tokens: int, temperature: float) -> str:
        calls.append(prompt)
        return "should not be called"

    client = TestClient(K.create_generation_app(fake_generate, model_name="tiny-koan"))
    response = client.post("/generate", json={"prompt": "   ", "max_new_tokens": 1})

    assert response.status_code == 400
    assert calls == []


def test_select_backend_chooses_the_smallest_healthy_backend_that_fits():
    backends = [
        K.InferenceBackend(name="dev-cpu", max_context_tokens=16, max_batch_tokens=64, healthy=True),
        K.InferenceBackend(name="broken-gpu", max_context_tokens=128, max_batch_tokens=256, healthy=False),
        K.InferenceBackend(name="prod-gpu", max_context_tokens=64, max_batch_tokens=256, healthy=True),
    ]

    selected = K.select_backend(
        prompt="x" * 40, max_new_tokens=20, backends=backends, reserved_tokens=4
    )
    assert selected.name == "prod-gpu"

    with pytest.raises(ValueError, match="No healthy backend"):
        K.select_backend(prompt="x" * 500, max_new_tokens=200, backends=backends)


def test_pack_micro_batch_greedily_keeps_requests_under_token_budget():
    requests = [
        K.GenerationRequest(prompt="a" * 8, max_new_tokens=2),   # cost 4
        K.GenerationRequest(prompt="b" * 12, max_new_tokens=3),  # cost 6
        K.GenerationRequest(prompt="c" * 16, max_new_tokens=1),  # would exceed budget
    ]

    batch = K.pack_micro_batch(requests, max_batch_tokens=10)
    assert batch == requests[:2]


def test_should_retry_error_separates_transient_and_permanent_failures():
    assert K.should_retry_error(status_code=429, error_message="rate limited") is True
    assert K.should_retry_error(status_code=503, error_message="model loading") is True
    assert K.should_retry_error(status_code=None, error_message="upstream timeout") is True

    assert K.should_retry_error(status_code=400, error_message="bad prompt") is False
    assert K.should_retry_error(status_code=401, error_message="bad api key") is False
    assert K.should_retry_error(status_code=None, error_message="context length exceeded") is False
