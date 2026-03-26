"""LLM client supporting Anthropic Claude and local models (Ollama/vLLM)."""

import json
import time

import anthropic

from config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_anthropic_client: anthropic.Anthropic | None = None
_local_client = None  # openai.OpenAI instance, lazily imported


def _get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _anthropic_client


def _get_local_client() -> object:
    """Return an OpenAI-compatible client pointed at the local LLM server."""
    global _local_client
    if _local_client is None:
        import openai
        _local_client = openai.OpenAI(
            base_url=settings.local_llm_base_url,
            api_key="not-needed",
        )
    return _local_client


def _is_local() -> bool:
    return settings.llm_provider == "local"


def get_llm_response(
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.0,
) -> str:
    """Send a prompt to the configured LLM and return the text response.

    Retries up to 3 times with exponential backoff on transient errors.
    """
    if _is_local():
        return _local_chat(prompt, system_prompt, temperature)
    return _anthropic_chat(prompt, system_prompt, temperature)


def get_structured_response(
    prompt: str,
    response_model: type,
    system_prompt: str | None = None,
) -> dict:
    """Get a response matching a Pydantic model's JSON schema.

    Uses tool use on Anthropic, or JSON-mode prompting on local models.
    """
    if _is_local():
        return _local_structured(prompt, response_model, system_prompt)
    return _anthropic_structured(prompt, response_model, system_prompt)


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

def _anthropic_chat(prompt: str, system_prompt: str | None, temperature: float) -> str:
    client = _get_anthropic_client()
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict = {
        "model": settings.anthropic_model,
        "max_tokens": settings.anthropic_max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.messages.create(**kwargs)
            _log_anthropic_usage(response)
            return response.content[0].text
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as exc:
            last_error = exc
            wait = 2 ** attempt
            logger.warning("Anthropic API error (attempt %d), retrying in %ds: %s", attempt + 1, wait, exc)
            time.sleep(wait)
        except anthropic.APIError as exc:
            logger.error("Anthropic API error (non-retryable): %s", exc)
            raise

    raise RuntimeError(f"Failed after 3 retries: {last_error}")


def _anthropic_structured(prompt: str, response_model: type, system_prompt: str | None) -> dict:
    client = _get_anthropic_client()
    schema = response_model.model_json_schema()
    tool_def = {
        "name": "structured_output",
        "description": "Return the structured response matching the requested schema.",
        "input_schema": schema,
    }
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict = {
        "model": settings.anthropic_model,
        "max_tokens": settings.anthropic_max_tokens,
        "temperature": 0.0,
        "messages": messages,
        "tools": [tool_def],
        "tool_choice": {"type": "tool", "name": "structured_output"},
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.messages.create(**kwargs)
            _log_anthropic_usage(response)
            for block in response.content:
                if block.type == "tool_use":
                    return block.input
            raise ValueError("No tool_use block in response")
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as exc:
            last_error = exc
            wait = 2 ** attempt
            logger.warning("Anthropic API error (attempt %d), retrying in %ds: %s", attempt + 1, wait, exc)
            time.sleep(wait)
        except anthropic.APIError as exc:
            logger.error("Anthropic API error (non-retryable): %s", exc)
            raise

    raise RuntimeError(f"Failed after 3 retries: {last_error}")


def _log_anthropic_usage(response: anthropic.types.Message) -> None:
    usage = response.usage
    logger.info(
        "Token usage - input: %d, output: %d",
        usage.input_tokens,
        usage.output_tokens,
    )


# ---------------------------------------------------------------------------
# Local model backend (OpenAI-compatible API)
# ---------------------------------------------------------------------------

def _local_chat(prompt: str, system_prompt: str | None, temperature: float) -> str:
    client = _get_local_client()
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=settings.local_llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=settings.local_llm_max_tokens,
            )
            usage = response.usage
            if usage:
                logger.info(
                    "Local LLM usage - input: %d, output: %d",
                    usage.prompt_tokens,
                    usage.completion_tokens,
                )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_error = exc
            wait = 2 ** attempt
            logger.warning("Local LLM error (attempt %d), retrying in %ds: %s", attempt + 1, wait, exc)
            time.sleep(wait)

    raise RuntimeError(f"Failed after 3 retries: {last_error}")


def _local_structured(prompt: str, response_model: type, system_prompt: str | None) -> dict:
    """Get structured output from a local model via JSON-mode prompting."""
    schema = response_model.model_json_schema()
    schema_str = json.dumps(schema, indent=2)

    augmented_prompt = (
        f"{prompt}\n\n"
        f"Respond with ONLY valid JSON matching this schema:\n{schema_str}"
    )
    raw = _local_chat(augmented_prompt, system_prompt, temperature=0.0)

    # Try to parse JSON from the response
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Local model returned invalid JSON: %s", raw[:200])
        raise ValueError(f"Local model did not return valid JSON: {raw[:200]}")
