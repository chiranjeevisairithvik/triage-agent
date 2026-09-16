"""
Run this YOURSELF with a real API key to verify RealLLMClient actually
works against a live model. This was not run by the assistant that
wrote this code -- no API key was available in that environment.
This script is what closes that gap: run it, read the output, and if
anything looks wrong, that's real signal to go fix (and worth
understanding why, not just re-running until it passes).

Usage (OpenAI):
    export OPENAI_API_KEY=sk-...
    python verify_live_llm.py --provider openai --model gpt-4o-mini

Usage (Azure OpenAI):
    export AZURE_OPENAI_API_KEY=...
    python verify_live_llm.py --provider azure \\
        --azure-endpoint https://YOUR-RESOURCE.openai.azure.com \\
        --azure-deployment YOUR-DEPLOYMENT-NAME \\
        --azure-api-version 2024-08-01-preview

What this checks, and why each matters:
  1. classify_ticket() on a clearly "closed" ticket -- does the model
     + your prompt actually produce "closed", not something else?
  2. classify_ticket() on a clearly vague ticket -- does it correctly
     say "needs-info" rather than guessing "open"?
  3. draft_clarifying_question() -- is the returned question sane
     and non-empty?
  4. summarize_resolution() against a real precedent from the sample
     data -- does the model ground its answer in the precedent
     instead of inventing unrelated steps?

This is intentionally a small, readable script, not a test framework
-- the point is for you to read the printed output and judge it
yourself, not just check for a green checkmark.
"""
import argparse
import os
import sys

from triage_agent import RealLLMClient, TicketRetriever


def build_chat_model(args):
    if args.provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            sys.exit("Set OPENAI_API_KEY in your environment first.")
        return ChatOpenAI(model=args.model, api_key=api_key, temperature=0)

    elif args.provider == "azure":
        from langchain_openai import AzureChatOpenAI

        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not api_key:
            sys.exit("Set AZURE_OPENAI_API_KEY in your environment first.")
        if not (args.azure_endpoint and args.azure_deployment and args.azure_api_version):
            sys.exit("--azure-endpoint, --azure-deployment, and --azure-api-version are all required for --provider azure.")
        return AzureChatOpenAI(
            azure_endpoint=args.azure_endpoint,
            azure_deployment=args.azure_deployment,
            api_version=args.azure_api_version,
            api_key=api_key,
            temperature=0,
        )

    else:
        sys.exit(f"Unknown provider: {args.provider}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=["openai", "azure"], required=True)
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name (OpenAI only)")
    parser.add_argument("--azure-endpoint")
    parser.add_argument("--azure-deployment")
    parser.add_argument("--azure-api-version")
    args = parser.parse_args()

    chat_model = build_chat_model(args)
    client = RealLLMClient(chat_model=chat_model)
    retriever = TicketRetriever("data/resolved_tickets.json")

    print("=" * 70)
    print("1. Classification -- clearly closed ticket")
    print("=" * 70)
    status = client.classify_ticket(
        "Update on my earlier ticket",
        "This is resolved now, working fine, thanks for the fix.",
    )
    print(f"Result: {status}  (expected: closed)")
    print("PASS" if status == "closed" else "CHECK THIS -- unexpected result")

    print("\n" + "=" * 70)
    print("2. Classification -- clearly vague ticket")
    print("=" * 70)
    status = client.classify_ticket("Problem", "It doesn't work.")
    print(f"Result: {status}  (expected: needs-info)")
    print("PASS" if status == "needs-info" else "CHECK THIS -- unexpected result")

    print("\n" + "=" * 70)
    print("3. Clarifying question for a vague ticket")
    print("=" * 70)
    question = client.draft_clarifying_question("App broken", "It doesn't work.")
    print(f"Result: {question}")
    print("PASS (non-empty)" if question.strip() else "CHECK THIS -- empty response")

    print("\n" + "=" * 70)
    print("4. Resolution suggestion, grounded in real precedent")
    print("=" * 70)
    query = "Cannot access shared drive after resetting my password this morning"
    precedents = retriever.find_similar(query, top_k=1)
    print(f"Retrieved precedent: {precedents[0]['id'] if precedents else 'NONE'}")
    suggestion = client.summarize_resolution(query, precedents)
    print(f"Result:\n{suggestion}")
    print(
        "\nManually check: does the suggestion actually reference the retrieved "
        "precedent's real steps, or does it look like the model ignored the "
        "precedent and invented generic advice? This is the one step that "
        "needs your judgment, not just a pass/fail."
    )


if __name__ == "__main__":
    main()
