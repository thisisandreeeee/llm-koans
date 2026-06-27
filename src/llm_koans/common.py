"""Shared helpers for the LLM koans."""

from __future__ import annotations

from typing import NoReturn


class KoanIncomplete(NotImplementedError):
    """Raised by unfinished koans."""


def TODO(message: str = "Replace this TODO with your implementation.") -> NoReturn:
    raise KoanIncomplete(message)
