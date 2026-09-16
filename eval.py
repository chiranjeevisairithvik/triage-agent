"""
Run: python eval.py

A tiny eval harness -- the kind of thing worth having in any agent
project, even a small one. Two things get measured:

  1. Classification accuracy: does the agent land on the right
     status (closed / needs-info / resolved-suggestion) for a set of
     hand-labeled test tickets?
  2. Retrieval precision: when the agent does suggest a resolution,
     is the precedent it retrieved actually the right one?

This is deliberately small (10 cases) -- the point isn't scale, it's
having *any* structured way to catch a regression before it ships,
rather than eyeballing demo.py output.
"""
from triage_agent import TriageAgent, TicketRetriever

LABELED_CASES = [
    {"id": "E1", "title": "Ticket closed", "description": "This is now resolved, fixed on my end.", "expected_status": "closed"},
    {"id": "E2", "title": "It's broken", "description": "Doesn't work.", "expected_status": "needs-info"},
    {"id": "E3", "title": "Problem again", "description": "Same issue as before, broken.", "expected_status": "needs-info"},
    {"id": "E4", "title": "Cannot send large attachment", "description": "I can't send an email with a 15MB attachment, smaller files work fine for me though.", "expected_status": "resolved-suggestion", "expected_precedent": "INC-1004"},
    {"id": "E5", "title": "Laptop very slow", "description": "My laptop has been extremely slow since the update installed last night, high disk usage shown.", "expected_status": "resolved-suggestion", "expected_precedent": "INC-1005"},
    {"id": "E6", "title": "VPN drops", "description": "My VPN disconnects constantly, about every ten minutes since I got a new router at home.", "expected_status": "resolved-suggestion", "expected_precedent": "INC-1003"},
    {"id": "E7", "title": "Login error", "description": "The HR portal shows a 500 error when I try to log in, other people can log in fine though.", "expected_status": "resolved-suggestion", "expected_precedent": "INC-1002"},
    {"id": "E8", "title": "Shared drive gone", "description": "Lost access to the shared drive right after resetting my password this morning, other apps work.", "expected_status": "resolved-suggestion", "expected_precedent": "INC-1001"},
    {"id": "E9", "title": "Random unrelated request", "description": "Can someone update the lunch menu poster in the break room please, it's outdated.", "expected_status": "resolved-suggestion", "expected_precedent": None},
    {"id": "E10", "title": "All set", "description": "No longer an issue, working now, thank you for the quick fix.", "expected_status": "closed"},
]


def run_eval():
    retriever = TicketRetriever("data/resolved_tickets.json")
    agent = TriageAgent(retriever=retriever)

    correct_status = 0
    correct_precedent = 0
    precedent_cases = 0

    for case in LABELED_CASES:
        result = agent.triage(case["id"], case["title"], case["description"])
        status_ok = result.status == case["expected_status"]
        correct_status += status_ok

        line = f"[{'OK' if status_ok else 'FAIL'}] {case['id']}: expected={case['expected_status']} got={result.status}"

        if "expected_precedent" in case:
            precedent_cases += 1
            got_precedent = result.precedents_used[0] if result.precedents_used else None
            precedent_ok = got_precedent == case["expected_precedent"]
            correct_precedent += precedent_ok
            line += f" | precedent expected={case['expected_precedent']} got={got_precedent} {'OK' if precedent_ok else 'FAIL'}"

        print(line)

    n = len(LABELED_CASES)
    print(f"\nClassification accuracy: {correct_status}/{n} ({100*correct_status/n:.0f}%)")
    if precedent_cases:
        print(f"Retrieval precision (on resolvable cases): {correct_precedent}/{precedent_cases} ({100*correct_precedent/precedent_cases:.0f}%)")


if __name__ == "__main__":
    run_eval()
