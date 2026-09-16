import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from triage_agent import TriageAgent, TicketRetriever

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "resolved_tickets.json"


@pytest.fixture
def agent():
    retriever = TicketRetriever(DATA_PATH)
    return TriageAgent(retriever=retriever)


def test_closed_ticket_detected(agent):
    result = agent.triage("T1", "Update", "This is resolved now, working now, thanks.")
    assert result.status == "closed"


def test_vague_short_ticket_needs_info(agent):
    result = agent.triage("T2", "App broken", "It doesn't work.")
    assert result.status == "needs-info"
    assert result.clarifying_question is not None


def test_short_but_specific_ticket_is_not_flagged_vague(agent):
    # 4 words, but contains a digit (error code) -> should NOT be
    # treated as needing more context, per _needs_more_context's
    # "short AND generic" rule.
    result = agent.triage("T3", "VPN failing", "VPN error 809 today.")
    assert result.status != "needs-info"


def test_well_formed_ticket_gets_grounded_suggestion(agent):
    result = agent.triage(
        "T4",
        "Cannot access shared drive",
        "After resetting my password I lost access to the shared drive, "
        "VPN and other apps still work fine for me.",
    )
    assert result.status == "resolved-suggestion"
    assert result.suggestion is not None
    assert len(result.precedents_used) >= 1
    assert result.precedents_used[0] == "INC-1001"  # the password-reset precedent


def test_retriever_returns_nothing_below_similarity_threshold():
    retriever = TicketRetriever(DATA_PATH)
    results = retriever.find_similar("completely unrelated query about lunch menus", top_k=2)
    assert results == []


def test_retriever_finds_relevant_precedent():
    retriever = TicketRetriever(DATA_PATH)
    results = retriever.find_similar("VPN keeps disconnecting every few minutes", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "INC-1003"
