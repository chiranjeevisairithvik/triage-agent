"""
Tests for RealLLMClient using a fake chat model -- no API key needed.

These tests do NOT prove a real OpenAI/Azure call works end-to-end
(that requires a live key -- see verify_live_llm.py at the project
root, which you must run yourself). What they DO prove: the JSON
parsing, key validation, and fallback-to-stub logic all behave
correctly given the kinds of responses a real model could plausibly
return, including malformed ones.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage_agent.llm import RealLLMClient


class FakeChatModel:
    """Stands in for a LangChain chat model. Returns canned responses
    in sequence, or raises, to simulate specific API behaviors."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    def invoke(self, prompt):
        self.call_count += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item

        class FakeResponse:
            def __init__(self, content):
                self.content = content

        return FakeResponse(item)


def test_classify_ticket_parses_clean_json():
    fake = FakeChatModel(['{"status": "open"}'])
    client = RealLLMClient(chat_model=fake)
    assert client.classify_ticket("t", "d") == "open"


def test_classify_ticket_strips_markdown_fences():
    fake = FakeChatModel(['```json\n{"status": "closed"}\n```'])
    client = RealLLMClient(chat_model=fake)
    assert client.classify_ticket("t", "d") == "closed"


def test_classify_ticket_falls_back_on_malformed_json():
    fake = FakeChatModel(["this is not json at all"])
    client = RealLLMClient(chat_model=fake)
    # Falls back to StubLLMClient's rule-based logic rather than crashing.
    result = client.classify_ticket("Ticket resolved", "Fixed now, thanks, working now.")
    assert result == "closed"  # the fallback stub correctly detects this from keywords


def test_classify_ticket_falls_back_on_invalid_status_value():
    fake = FakeChatModel(['{"status": "urgent"}'])  # not a valid status
    client = RealLLMClient(chat_model=fake)
    result = client.classify_ticket("It doesn't work.", "It doesn't work.")
    assert result in {"open", "closed", "needs-info"}  # fell back, still a valid status


def test_classify_ticket_retries_then_succeeds():
    fake = FakeChatModel([ConnectionError("timeout"), '{"status": "open"}'])
    client = RealLLMClient(chat_model=fake, max_retries=2)
    result = client.classify_ticket("t", "d")
    assert result == "open"
    assert fake.call_count == 2  # first attempt failed, second succeeded


def test_classify_ticket_falls_back_after_exhausting_retries():
    fake = FakeChatModel([ConnectionError("timeout"), ConnectionError("timeout")])
    client = RealLLMClient(chat_model=fake, max_retries=1)
    # Both attempts fail -> _ask_json returns None -> falls back to stub.
    result = client.classify_ticket("It doesn't work.", "It doesn't work.")
    assert result == "needs-info"  # stub's rule-based answer for this input


def test_draft_clarifying_question_parses_json():
    fake = FakeChatModel(['{"question": "What error code do you see?"}'])
    client = RealLLMClient(chat_model=fake)
    assert client.draft_clarifying_question("t", "d") == "What error code do you see?"


def test_summarize_resolution_with_no_precedents_short_circuits():
    fake = FakeChatModel([])  # should never be called
    client = RealLLMClient(chat_model=fake)
    result = client.summarize_resolution("some ticket", precedents=[])
    assert "No sufficiently similar" in result
    assert fake.call_count == 0
