"""LLM client abstraction.

Three providers are supported:

* "openai": any OpenAI-compatible endpoint, including a local vLLM server,
  which is the default so the whole system runs without API keys.
* "anthropic": the Anthropic Messages API.
* "echo": a deterministic offline stub used by unit tests and CI.

All three query-path call sites -- planning, agent inspection and answer
generation -- go through one client, so a run has exactly one model to
attribute its behaviour to.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import LLMConfig


@dataclass
class LLMResponse:
    """One completion and what it cost.

    Attributes:
      text: Generated text, with any reasoning preamble stripped.
      prompt_tokens: Input tokens, as reported by the provider.
      completion_tokens: Output tokens, as reported by the provider.
      model: Model that produced the response.
      latency_ms: Wall-clock time for the call.
      cost_usd: Estimated cost from the configured prices; 0.0 for a local
        server, which is the default.
      raw: Provider payload, when a caller needs more than the fields above.
    """

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    """Interface every LLM backend implements.

    Attributes:
      name: Provider-qualified model name, e.g. "openai:Qwen/Qwen3-4B".
    """

    name: str

    def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int | None = None
    ) -> LLMResponse:
        """Complete a single-turn prompt.

        Args:
          prompt: User message.
          system: System message, if any.
          max_tokens: Output cap; falls back to the configured value.

        Returns:
          The completion and its usage.
        """
        ...


def _cost(config: LLMConfig, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate the dollar cost of one call.

    Args:
      config: LLM configuration holding the per-million-token prices.
      prompt_tokens: Input tokens.
      completion_tokens: Output tokens.

    Returns:
      The estimated cost in USD, which is 0.0 unless prices are configured.
    """
    return round(
        prompt_tokens / 1e6 * config.price_in_per_mtok
        + completion_tokens / 1e6 * config.price_out_per_mtok,
        6,
    )


class OpenAICompatibleClient:
    """Client for any OpenAI-compatible chat endpoint.

    Works against a local vLLM server, OpenAI itself, or any clone. The
    default configuration points at a local server, so nothing in the system
    requires an API key.

    Attributes:
      config: LLM configuration.
      name: Provider-qualified model name.
      client: The underlying OpenAI SDK client.
    """

    def __init__(self, config: LLMConfig) -> None:
        """Construct the SDK client.

        Retries are handled here rather than by the SDK, so that the backoff
        is visible and identical across providers.

        Args:
          config: LLM configuration naming the endpoint and model.
        """
        from openai import OpenAI

        self.config = config
        self.name = f"openai:{config.model}"
        self.client = OpenAI(
            base_url=config.base_url or None,
            api_key=os.environ.get(config.api_key_env) or os.environ.get("OPENAI_API_KEY") or "EMPTY",
            timeout=config.timeout,
            max_retries=0,
        )

    def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int | None = None
    ) -> LLMResponse:
        """Complete a prompt, retrying on transient failures.

        Args:
          prompt: User message.
          system: System message, if any.
          max_tokens: Output cap; falls back to the configured value.

        Returns:
          The completion and its usage.

        Raises:
          RuntimeError: Every attempt failed. The last underlying error is
            included in the message.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            start = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=max_tokens or self.config.max_tokens,
                    extra_body=self.config.extra_body or None,
                )
            except Exception as exc:  # network / server hiccup
                last_error = exc
                time.sleep(min(2**attempt, 8))
                continue
            usage = getattr(response, "usage", None)
            text = response.choices[0].message.content or ""
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            return LLMResponse(
                text=_strip_reasoning(text),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=self.config.model,
                latency_ms=(time.perf_counter() - start) * 1000,
                cost_usd=_cost(self.config, prompt_tokens, completion_tokens),
            )
        raise RuntimeError(f"LLM call failed after retries: {last_error}")


class AnthropicClient:
    """Client for the Anthropic Messages API.

    Attributes:
      config: LLM configuration.
      name: Provider-qualified model name.
      client: The underlying Anthropic SDK client.
    """

    def __init__(self, config: LLMConfig) -> None:
        """Construct the SDK client.

        Args:
          config: LLM configuration naming the model and key environment
            variable.
        """
        import anthropic

        self.config = config
        self.name = f"anthropic:{config.model}"
        self.client = anthropic.Anthropic(
            api_key=os.environ.get(config.api_key_env) or os.environ.get("ANTHROPIC_API_KEY"),
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int | None = None
    ) -> LLMResponse:
        """Complete a prompt via the Messages API.

        Args:
          prompt: User message.
          system: System message, if any.
          max_tokens: Output cap; falls back to the configured value.

        Returns:
          The completion and its usage, with non-text content blocks
          discarded. Retries are left to the SDK here.
        """
        start = time.perf_counter()
        message = self.client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        return LLMResponse(
            text=text,
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
            model=self.config.model,
            latency_ms=(time.perf_counter() - start) * 1000,
            cost_usd=_cost(self.config, message.usage.input_tokens, message.usage.output_tokens),
        )


class EchoClient:
    """Deterministic offline stub used by unit tests and CI.

    It cites the first evidence blocks it is shown and returns an empty plan
    for any prompt asking for JSON, so the full pipeline -- including
    citation parsing and validation -- runs end to end with no model, no
    network and no GPU.

    Attributes:
      config: LLM configuration, defaulted when not supplied.
      name: Always "echo".
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        """Configure the stub.

        Args:
          config: LLM configuration; an "echo" default is used when omitted.
        """
        self.config = config or LLMConfig(provider="echo", model="echo")
        self.name = "echo"

    def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int | None = None
    ) -> LLMResponse:
        """Return a canned response derived from the prompt.

        Args:
          prompt: User message; scanned for citation-shaped locations.
          system: System message, ignored.
          max_tokens: Output cap, ignored.

        Returns:
          An empty JSON plan for prompts requesting JSON, otherwise a fixed
          sentence citing up to three locations found in the prompt. Token
          counts are character estimates.
        """
        if "Respond with JSON" in prompt or "JSON object" in prompt:
            text = json.dumps({"queries": [], "category": "code_lookup", "done": True})
        else:
            citations = re.findall(r"\[([^\]\n]+:\d+-\d+)\]", prompt)[:3]
            body = "Offline echo backend: no model was called."
            text = body + ("\n" + " ".join(f"[{c}]" for c in citations) if citations else "")
        return LLMResponse(
            text=text,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(text) // 4,
            model="echo",
        )


_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_reasoning(text: str) -> str:
    """Strip a reasoning preamble from a model response.

    Qwen-style reasoning models emit `<think>...</think>` before the answer,
    which would otherwise be parsed for citations and shown to the user.

    Args:
      text: Raw model output.

    Returns:
      The response with complete think blocks removed, truncated at an
      unclosed opening tag -- which is what a hit output cap looks like.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>")[0]
    return cleaned.strip()


def build_llm(config: LLMConfig) -> LLMClient:
    """Construct the LLM client a config asks for.

    Args:
      config: LLM configuration.

    Returns:
      The client for the configured provider.

    Raises:
      ValueError: The provider name is not recognised.
    """
    provider = config.provider.lower()
    if provider in {"openai", "vllm", "openai_compatible"}:
        return OpenAICompatibleClient(config)
    if provider == "anthropic":
        return AnthropicClient(config)
    if provider in {"echo", "none", "offline"}:
        return EchoClient(config)
    raise ValueError(f"unknown LLM provider {config.provider!r}")


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from a chatty model response.

    Planning and agent inspection both need structured output from a model
    that may wrap it in prose or a code fence. Every caller treats `None` as
    "fall back to the deterministic path", so a malformed response degrades
    the run rather than failing it.

    Args:
      text: Raw model output.

    Returns:
      The parsed object, or `None` if nothing parseable was found. A fenced
      block is preferred, then the outermost braces; trailing commas are
      tolerated on a second attempt.
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # tolerate trailing commas
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))
        except json.JSONDecodeError:
            return None
