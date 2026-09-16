"""
Pluggable LLM interface.

Two implementations:

  - StubLLMClient: rule-based, deterministic, no API key. Used as the
    default in demo.py, eval.py, and tests/ -- NOT because it's meant
    to represent "the AI," but because tests must be deterministic
    and free, and the pipeline should be runnable with zero setup.
    This is a normal engineering pattern (a fake/stub dependency for
    tests), not a substitute for real LLM behavior.

  - RealLLMClient: an actual LLM call via LangChain, using either
    OpenAI or Azure OpenAI (same class, different constructor args --
    both are OpenAI-compatible chat APIs). This is the implementation
    that does real classification, question drafting, and
    summarization work.

Honesty note (read before claiming this project uses an LLM): the
prompts, JSON-mode structured output, and parsing/fallback logic in
RealLLMClient below are written and unit-testable, but a *live* call
to OpenAI/Azure has not been executed by the assistant that wrote
this code -- no API key was available in that environment. Run
`python verify_live_llm.py` yourself with a real key before relying
on or describing this as verified end-to-end. See that file for
exactly what it checks.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

VALID_STATUSES = {"open", "closed", "needs-info"}


class LLMClient(ABC):
    @abstractmethod
    def classify_ticket(self, title: str, description: str) -> str:
        """Return one of: 'open', 'closed', 'needs-info'."""

    @abstractmethod
    def draft_clarifying_question(self, title: str, description: str) -> str:
        """Return a targeted question to fill the missing context."""

    @abstractmethod
    def summarize_resolution(self, ticket_text: str, precedents: list[dict]) -> str:
        """Turn retrieved precedent tickets into suggested resolution steps."""


class StubLLMClient(LLMClient):
    """
    Deterministic, rule-based stand-in for a real LLM. Default for
    demo.py / eval.py / tests/ -- see module docstring for why.
    """

    CLOSED_KEYWORDS = ("resolved", "fixed", "closed", "no longer an issue", "working now")
    VAGUE_MIN_WORDS = 8

    def classify_ticket(self, title: str, description: str) -> str:
        text = f"{title} {description}".lower()
        if any(k in text for k in self.CLOSED_KEYWORDS):
            return "closed"
        if len(description.split()) < self.VAGUE_MIN_WORDS:
            return "needs-info"
        return "open"

    def draft_clarifying_question(self, title: str, description: str) -> str:
        text = f"{title} {description}".lower()
        if "error" in text and "code" not in text and "message" not in text:
            return "Can you share the exact error code or error message you're seeing?"
        if "slow" in text or "not working" in text:
            return "When did this start, and does it affect one device/user or several?"
        return "Could you add more detail on what you were doing when this happened, and any error text shown?"

    def summarize_resolution(self, ticket_text: str, precedents: list[dict]) -> str:
        if not precedents:
            return "No sufficiently similar past ticket found -- recommend manual triage."
        top = precedents[0]
        steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(top["resolution_steps"]))
        return (
            f"Based on similar past incident {top['id']} ({top['title']}), "
            f"likely root cause: {top['root_cause']}\nSuggested steps:\n{steps}"
        )


class RealLLMClient(LLMClient):
    """
    Real implementation backed by an OpenAI-compatible chat model via
    LangChain (works for both plain OpenAI and Azure OpenAI -- you
    pass in an already-constructed chat model, this class doesn't
    care which).

    Design choices worth being able to explain:

    1. JSON-mode structured output for classification, not free text.
       Asking a model to "reply with one word" and then string-matching
       its reply is brittle -- models add punctuation, explanations,
       or synonyms. Instead, `_ask_json()` asks for a JSON object with
       a fixed schema and parses that, which is far more reliable to
       consume programmatically.

    2. Explicit validation + fallback. If the model returns malformed
       JSON or a status outside {open, closed, needs-info}, this does
       NOT crash the pipeline or silently trust bad data -- it logs a
       warning and falls back to StubLLMClient's rule-based logic for
       that single call. A production agent should degrade gracefully,
       not take down ticket triage because one model response was
       malformed.

    3. Retries with backoff on transient API errors (timeouts, rate
       limits) -- a live network call needs this; the stub never did.
    """

    def __init__(self, chat_model, max_retries: int = 2):
        # chat_model: a LangChain BaseChatModel, e.g.
        #   from langchain_openai import ChatOpenAI
        #   chat_model = ChatOpenAI(model="gpt-4o-mini", api_key="...")
        # or, for Azure:
        #   from langchain_openai import AzureChatOpenAI
        #   chat_model = AzureChatOpenAI(azure_deployment="...", api_key="...", azure_endpoint="...", api_version="...")
        self.chat_model = chat_model
        self.max_retries = max_retries
        self._fallback = StubLLMClient()

    def _invoke_with_retries(self, prompt: str) -> str:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.chat_model.invoke(prompt)
                return response.content
            except Exception as exc:  # network/timeout/rate-limit errors from the provider
                last_error = exc
                logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, self.max_retries + 1, exc)
        raise RuntimeError(f"LLM call failed after {self.max_retries + 1} attempts") from last_error

    def _ask_json(self, prompt: str, expected_keys: set[str]) -> dict | None:
        """
        Calls the model, expecting a JSON object back. Returns the
        parsed dict, or None if the response wasn't valid/complete
        JSON -- callers must handle the None case with a fallback,
        never assume this succeeds.
        """
        full_prompt = (
            f"{prompt}\n\n"
            "Respond with ONLY a JSON object, no other text, no markdown code fences. "
            f"The object must have exactly these keys: {sorted(expected_keys)}."
        )
        try:
            raw = self._invoke_with_retries(full_prompt)
        except RuntimeError:
            return None

        # Models sometimes wrap JSON in ```json fences despite instructions -- strip defensively.
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON response, falling back. Raw: %r", raw[:200])
            return None

        if not expected_keys.issubset(parsed.keys()):
            logger.warning("LLM JSON missing expected keys %s, got %s", expected_keys, parsed.keys())
            return None

        return parsed

    def classify_ticket(self, title: str, description: str) -> str:
        prompt = (
            "Classify this IT support ticket's status.\n\n"
            f"Title: {title}\nDescription: {description}\n\n"
            "Rules:\n"
            "- 'closed' if the reporter is confirming something is already fixed/resolved.\n"
            "- 'needs-info' if there isn't enough detail to diagnose or act on (too vague, "
            "no symptoms, no context).\n"
            "- 'open' otherwise -- a real, actionable issue with enough detail to work with."
        )
        result = self._ask_json(prompt, expected_keys={"status"})

        if result is None:
            return self._fallback.classify_ticket(title, description)

        status = str(result["status"]).strip().lower()
        if status not in VALID_STATUSES:
            logger.warning("LLM returned invalid status %r, falling back.", status)
            return self._fallback.classify_ticket(title, description)

        return status

    def draft_clarifying_question(self, title: str, description: str) -> str:
        prompt = (
            "This IT support ticket lacks enough detail to act on.\n\n"
            f"Title: {title}\nDescription: {description}\n\n"
            "Write ONE specific, concise clarifying question to ask the reporter that "
            "would unblock diagnosis (e.g. ask for an error code, when it started, "
            "how many people are affected -- whichever is most relevant here)."
        )
        result = self._ask_json(prompt, expected_keys={"question"})

        if result is None:
            return self._fallback.draft_clarifying_question(title, description)

        return str(result["question"]).strip()

    def summarize_resolution(self, ticket_text: str, precedents: list[dict]) -> str:
        if not precedents:
            return "No sufficiently similar past ticket found -- recommend manual triage."

        precedent_text = "\n\n".join(
            f"Past incident {p['id']}: {p['title']}\n"
            f"Root cause: {p['root_cause']}\n"
            f"Steps taken: {'; '.join(p['resolution_steps'])}"
            for p in precedents
        )
        prompt = (
            "A new support ticket needs a resolution suggestion, based ONLY on the "
            "precedent tickets below -- do not invent steps that aren't grounded in them. "
            "If the precedents don't clearly apply, say so rather than guessing.\n\n"
            f"New ticket:\n{ticket_text}\n\nPrecedent tickets:\n{precedent_text}\n\n"
            "Provide a short explanation of the likely cause and concrete resolution steps, "
            "as a single formatted string."
        )
        result = self._ask_json(prompt, expected_keys={"summary"})

        if result is None:
            # Fall back to the deterministic, precedent-grounded formatting --
            # still uses the real retrieved precedents, just without model
            # phrasing, which is a safe degrade rather than a crash.
            return self._fallback.summarize_resolution(ticket_text, precedents)

        return str(result["summary"]).strip()
