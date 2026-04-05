"""
API Key Rotation Manager — round-robin key selection with automatic
failover, per-key usage tracking, and thread-safe retry logic.

Used as the single entry-point for all Groq LLM calls across the system.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any

from groq import Groq

from scholarsync.config.settings import get_settings
from scholarsync.utils.logger import get_logger

logger = get_logger(__name__)


# ── Exceptions ───────────────────────────────────────────────────────

class AllKeysExhaustedError(Exception):
    """Raised when every available API key has failed."""
    pass


# ── Per-key state ────────────────────────────────────────────────────

@dataclass
class _KeyState:
    key: str
    request_count: int = 0
    failure_count: int = 0
    last_used: float = 0.0
    disabled_until: float = 0.0  # Unix timestamp; 0 = enabled

    @property
    def is_active(self) -> bool:
        return time.time() >= self.disabled_until


# ── Singleton Manager ───────────────────────────────────────────────

class KeyManager:
    """Thread-safe round-robin API key manager with failover."""

    _instance: KeyManager | None = None
    _lock = threading.Lock()

    def __new__(cls) -> KeyManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        settings = get_settings()

        # Build key list: extra keys from GROQ_API_KEYS + the primary key
        raw_keys: list[str] = []
        if settings.groq_api_keys:
            raw_keys.extend(settings.groq_api_keys)
        if settings.groq_api_key and settings.groq_api_key not in raw_keys:
            raw_keys.append(settings.groq_api_key)

        if not raw_keys:
            raise ValueError("No Groq API keys configured. Set GROQ_API_KEY or GROQ_API_KEYS.")

        self._keys: list[_KeyState] = [_KeyState(key=k) for k in raw_keys]
        self._index = 0
        self._mu = threading.Lock()

        logger.info("KeyManager initialized with %d API key(s)", len(self._keys))

    # ── Key selection (round-robin) ──────────────────────────────────

    def _next_active_key(self) -> _KeyState | None:
        """Pick the next active key in round-robin order."""
        n = len(self._keys)
        for _ in range(n):
            ks = self._keys[self._index % n]
            self._index += 1
            if ks.is_active:
                return ks
        return None

    # ── Public: call LLM with automatic retry ────────────────────────

    def call_llm(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        max_retries: int = 3,
    ) -> str:
        """
        Call Groq LLM chat completions with automatic key rotation.

        Returns:
            The assistant's response text.

        Raises:
            AllKeysExhaustedError: If every key has been tried and failed.
        """
        settings = get_settings()
        _model = model or settings.groq_model
        _temperature = temperature if temperature is not None else settings.groq_temperature
        _max_tokens = max_tokens or settings.groq_max_tokens

        last_error: Exception | None = None

        for attempt in range(max_retries):
            with self._mu:
                ks = self._next_active_key()

            if ks is None:
                logger.error("All API keys exhausted (attempt %d/%d)", attempt + 1, max_retries)
                raise AllKeysExhaustedError(
                    "All Groq API keys are rate-limited or failed. "
                    "Please wait and try again later."
                )

            try:
                client = Groq(api_key=ks.key)
                kwargs: dict[str, Any] = {
                    "model": _model,
                    "messages": messages,
                    "temperature": _temperature,
                    "max_tokens": _max_tokens,
                }
                if response_format:
                    kwargs["response_format"] = response_format

                response = client.chat.completions.create(**kwargs)

                # Success
                with self._mu:
                    ks.request_count += 1
                    ks.last_used = time.time()

                text = response.choices[0].message.content.strip()
                logger.info(
                    "LLM call succeeded (key ...%s, attempt %d)",
                    ks.key[-6:], attempt + 1,
                )
                return text

            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                with self._mu:
                    ks.failure_count += 1
                    # On rate-limit, disable key for 60 seconds
                    if "rate" in err_str or "429" in err_str or "limit" in err_str:
                        ks.disabled_until = time.time() + 60
                        logger.warning(
                            "Key ...%s rate-limited, disabled for 60s. Rotating.",
                            ks.key[-6:],
                        )
                    else:
                        logger.warning(
                            "Key ...%s failed: %s. Rotating.",
                            ks.key[-6:], e,
                        )

        # All retries exhausted
        raise AllKeysExhaustedError(
            f"All API keys failed after {max_retries} attempts. Last error: {last_error}"
        )

    def call_llm_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 3,
    ):
        """
        Stream Groq LLM chat completions with automatic key rotation.

        Yields:
            Text chunks as they arrive.
        """
        settings = get_settings()
        _model = model or settings.groq_model
        _temperature = temperature if temperature is not None else settings.groq_temperature
        _max_tokens = max_tokens or settings.groq_max_tokens

        last_error: Exception | None = None

        for attempt in range(max_retries):
            with self._mu:
                ks = self._next_active_key()

            if ks is None:
                raise AllKeysExhaustedError("All Groq API keys are rate-limited or failed.")

            try:
                client = Groq(api_key=ks.key)
                stream = client.chat.completions.create(
                    model=_model,
                    messages=messages,
                    temperature=_temperature,
                    max_tokens=_max_tokens,
                    stream=True,
                )

                with self._mu:
                    ks.request_count += 1
                    ks.last_used = time.time()

                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content

                return  # Success — exit retry loop

            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                with self._mu:
                    ks.failure_count += 1
                    if "rate" in err_str or "429" in err_str or "limit" in err_str:
                        ks.disabled_until = time.time() + 60
                        logger.warning("Key ...%s rate-limited (stream). Rotating.", ks.key[-6:])
                    else:
                        logger.warning("Key ...%s stream failed: %s. Rotating.", ks.key[-6:], e)

        raise AllKeysExhaustedError(
            f"All API keys failed after {max_retries} attempts. Last error: {last_error}"
        )

    # ── Diagnostics ──────────────────────────────────────────────────

    def get_stats(self) -> list[dict]:
        """Return usage stats for each key (for debugging / health check)."""
        with self._mu:
            return [
                {
                    "key_suffix": ks.key[-6:],
                    "requests": ks.request_count,
                    "failures": ks.failure_count,
                    "active": ks.is_active,
                }
                for ks in self._keys
            ]


# ── Module-level convenience ─────────────────────────────────────────

def get_key_manager() -> KeyManager:
    """Return the singleton KeyManager instance."""
    return KeyManager()
