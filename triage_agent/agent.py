"""
Ticket Triage & Resolution Agent.

Pipeline: classify -> (enrich if needed) -> retrieve precedent ->
suggest resolution.

The core design decision documented here (and in the README): how
does the agent decide a ticket is "vague" and needs a clarifying
question, versus "well-formed" and ready for retrieval? Handled by
`_needs_more_context()` -- a small, explicit heuristic rather than an
opaque LLM judgment call, because a *consistent, auditable* rule for
something this operationally important beats a slightly-smarter but
unpredictable one. See the README for the reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .llm import LLMClient, StubLLMClient
from .retriever import TicketRetriever

MIN_DESCRIPTION_WORDS = 8
VAGUE_PHRASES = ("doesn't work", "not working", "broken", "issue", "problem")


@dataclass
class TriageResult:
    ticket_id: str
    status: str  # "closed" | "needs-info" | "resolved-suggestion"
    clarifying_question: str | None = None
    suggestion: str | None = None
    precedents_used: list[str] = field(default_factory=list)


class TriageAgent:
    def __init__(self, retriever: TicketRetriever, llm: LLMClient | None = None):
        self.retriever = retriever
        self.llm = llm or StubLLMClient()

    def _needs_more_context(self, title: str, description: str) -> bool:
        """
        A ticket needs enrichment if it's short AND generic -- either
        condition alone isn't enough (a short ticket can still be
        precise: "VPN error 809" is 3 words and perfectly actionable;
        a long ticket can still be vague if it's mostly restating the
        title). Requiring both cuts false positives on short-but-clear
        tickets, which is the failure mode that would otherwise make
        the agent annoying to actual users.
        """
        word_count = len(description.split())
        text = f"{title} {description}".lower()
        is_generic = any(phrase in text for phrase in VAGUE_PHRASES) and not any(
            char.isdigit() for char in text
        )
        return word_count < MIN_DESCRIPTION_WORDS and is_generic

    def triage(self, ticket_id: str, title: str, description: str) -> TriageResult:
        status = self.llm.classify_ticket(title, description)

        if status == "closed":
            return TriageResult(ticket_id=ticket_id, status="closed")

        if self._needs_more_context(title, description):
            question = self.llm.draft_clarifying_question(title, description)
            return TriageResult(ticket_id=ticket_id, status="needs-info", clarifying_question=question)

        query_text = f"{title} {description}"
        precedents = self.retriever.find_similar(query_text, top_k=2)
        suggestion = self.llm.summarize_resolution(query_text, precedents)

        return TriageResult(
            ticket_id=ticket_id,
            status="resolved-suggestion",
            suggestion=suggestion,
            precedents_used=[p["id"] for p in precedents],
        )
