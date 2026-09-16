"""
Run: python demo.py

Demonstrates all three pipeline outcomes: closed, needs-info, and a
grounded resolution suggestion.
"""
from triage_agent import TriageAgent, TicketRetriever

DEMO_TICKETS = [
    {
        "id": "NEW-2001",
        "title": "Can't access shared drive",
        "description": (
            "After resetting my password this morning I can no longer open "
            "the marketing shared drive, even though VPN and email work fine."
        ),
    },
    {
        "id": "NEW-2002",
        "title": "App broken",
        "description": "It doesn't work.",
    },
    {
        "id": "NEW-2003",
        "title": "Ticket update",
        "description": "This issue is resolved now, thanks for the help earlier, working now.",
    },
]


def main():
    retriever = TicketRetriever("data/resolved_tickets.json")
    agent = TriageAgent(retriever=retriever)  # uses StubLLMClient by default

    for ticket in DEMO_TICKETS:
        result = agent.triage(ticket["id"], ticket["title"], ticket["description"])
        print(f"\n=== {result.ticket_id} -> {result.status} ===")
        if result.clarifying_question:
            print(f"Clarifying question: {result.clarifying_question}")
        if result.suggestion:
            print(f"Suggestion:\n{result.suggestion}")
            print(f"Precedents used: {result.precedents_used}")


if __name__ == "__main__":
    main()
