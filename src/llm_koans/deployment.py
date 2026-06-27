"""Koan 08: LLM deployment from toy serving to production pressure."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from fastapi import FastAPI

from .common import TODO


@dataclass(frozen=True)
class GenerationRequest:
    """A minimal generation request used by deployment koans.

    Real systems usually include more fields, but these three show up almost
    everywhere: the prompt, a generation budget, and a sampling temperature.
    """

    prompt: str
    max_new_tokens: int
    temperature: float = 0.7


@dataclass(frozen=True)
class InferenceBackend:
    """A deployable model endpoint with capacity metadata.

    `max_context_tokens` protects a single request from exceeding the model's
    context window. `max_batch_tokens` protects throughput-oriented batching.
    """

    name: str
    max_context_tokens: int
    max_batch_tokens: int
    healthy: bool = True


def estimate_tokens(text: str) -> int:
    """Estimate tokens for capacity planning.

    Production systems should use the model tokenizer. This koan intentionally
    starts with the common rough heuristic: about four characters per token,
    rounded up, with blank strings costing zero.
    """
    TODO("Return 0 for blank text; otherwise return ceil(len(text) / 4).")


def validate_generation_budget(
    prompt: str,
    max_new_tokens: int,
    context_window: int,
    reserved_tokens: int = 0,
) -> dict[str, int | bool]:
    """Report whether a generation request fits inside a context window.

    The total budget is prompt tokens + requested output tokens + reserved
    tokens. Reserved tokens cover system prompts, tool schemas, routing headers,
    or safety wrappers that are easy to forget in prototypes.
    """
    TODO("Return prompt_tokens, requested_tokens, reserved_tokens, total_tokens, fits, and overflow_tokens.")


def create_generation_app(
    generate_text: Callable[[str, int, float], str],
    model_name: str,
) -> FastAPI:
    """Create a tiny FastAPI app for LLM text generation.

    Implement two endpoints:
        GET  /health   -> {"status": "ok", "model": model_name}
        POST /generate -> {"model": model_name, "text": generated_text}

    Reject empty prompts with HTTP 400 before calling the model. This is the
    bridge from "I can run inference in a notebook" to "I can serve it safely".
    """
    TODO("Build a FastAPI app with /health and /generate endpoints.")


def select_backend(
    prompt: str,
    max_new_tokens: int,
    backends: Sequence[InferenceBackend],
    reserved_tokens: int = 0,
) -> InferenceBackend:
    """Choose the smallest healthy backend that can fit the request.

    This models a common production move: route tiny requests to cheaper
    capacity, skip unhealthy backends, and escalate large context-window jobs to
    bigger serving pools.
    """
    TODO("Filter to healthy backends that fit the token budget, then return the one with the smallest max_context_tokens.")


def pack_micro_batch(
    requests: Sequence[GenerationRequest],
    max_batch_tokens: int,
) -> list[GenerationRequest]:
    """Greedily pack generation requests under a token budget.

    Dynamic batching improves throughput, but production systems must bound the
    combined prompt + output budget so one batch does not blow up latency or
    memory. Preserve input order and stop before the first request that would
    exceed the budget.
    """
    TODO("Accumulate request token costs until the next request would exceed max_batch_tokens.")


def should_retry_error(status_code: int | None, error_message: str) -> bool:
    """Return whether a failed generation call should be retried.

    Retry transient failures such as rate limits, overloaded/model-loading
    servers, and timeouts. Do not retry permanent caller problems like bad
    prompts, auth failures, or context-window overflow.
    """
    TODO("Retry 429/5xx/timeouts; do not retry 4xx auth/bad-request or context-length errors.")
