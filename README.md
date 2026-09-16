# Ticket Triage & Resolution Agent

A small agent that classifies incoming support tickets, asks for more
context when a ticket is too vague to act on, and — for well-formed
tickets — retrieves similar past resolved tickets and suggests
resolution steps grounded in that precedent.

This is a standalone project built to explore the same problem shape
as production incident-triage systems I've worked on professionally
(classify → enrich → retrieve → suggest), reimplemented independently
here as a portfolio piece.

```
Incoming ticket
      │
      ▼
 1. Classifier ──────► closed? → done
      │
      ▼ open / unclear
 2. Context sufficiency check
      │
      ├── insufficient ──► ask a targeted clarifying question
      │
      ▼ sufficient
 3. Retriever (TF-IDF similarity over past resolved tickets)
      │
      ▼
 4. Resolution suggestion, grounded in retrieved precedent
```

## Quickstart

```bash
pip install -r requirements.txt
python demo.py         # runs the pipeline on 3 sample tickets (CLI, stub LLM)
python eval.py          # scores classification + retrieval on labeled test cases
python -m pytest tests/ -v
streamlit run app.py    # interactive UI -- stub by default, or your own OpenAI/Azure key
```

No API key needed for any of the above. To verify the real LLM path
end-to-end with your own key:

```bash
export OPENAI_API_KEY=sk-...
python verify_live_llm.py --provider openai --model gpt-4o-mini
```

## The core design decision: when does a ticket need more context?

The trickiest part of this pipeline isn't the LLM call, it's deciding
*when to ask a clarifying question at all*. Two bad failure modes to
avoid:

- **Too aggressive** — asking for clarification on tickets that are
  already actionable is annoying and slows down real incidents.
- **Too passive** — letting genuinely vague tickets ("it's broken")
  through to the resolution step wastes a retrieval + LLM call on
  something no precedent search can meaningfully answer.

I initially considered using word count alone (short = vague), but
that fails on short-but-specific tickets like *"VPN error 809"* (3
words, completely actionable) and passes long-but-vague tickets that
just repeat the title in different words.

The heuristic in `agent.py::_needs_more_context()` instead requires
**both** conditions: short description **and** generic phrasing (no
specific identifiers like error codes/numbers, and language drawn
from a small set of vague phrases like "broken", "not working",
"issue"). Requiring both cut false positives on short-but-clear
tickets to zero in testing, at the cost of occasionally letting a
genuinely vague-but-wordy ticket through — a tradeoff I'd rather make
than annoy users with unnecessary clarification requests.

This is implemented as an explicit, auditable rule rather than an
LLM judgment call, deliberately — for something that gates whether a
person gets an extra round-trip email, a predictable rule you can
explain and tune beats a slightly-smarter but unpredictable model call.

## Retrieval: TF-IDF, not embeddings — and a known limitation

`retriever.py` uses TF-IDF cosine similarity, not embeddings. For a
small, single-domain ticket corpus, TF-IDF is fast, needs no API
calls or vector DB, and — importantly — is fully explainable: you can
point to the exact overlapping terms that drove a match.

**The tradeoff shows up in `eval.py`'s output.** One eval case, an
unrelated "update the lunch menu poster" ticket, incorrectly matched
past incident `INC-1005` ("laptop slow after Windows **update**") at
0.396 similarity — well above the 0.12 threshold — purely because
both texts contain the word "update." TF-IDF matches vocabulary, not
meaning, so this kind of lexical false-positive is expected at the
margins.

Current retrieval precision on the 6 labeled resolvable cases in
`eval.py`: **5/6 (83%)** — this is the one failure.

What I'd do at larger scale: swap in sentence embeddings (e.g., a
small sentence-transformer model) for semantic rather than lexical
matching, and/or add a stopword list of common-but-non-diagnostic
terms ("update," "issue," "problem") that shouldn't drive a match on
their own. Didn't do this here because TF-IDF's explainability is
genuinely valuable at this corpus size, and the failure mode is easy
to reason about — but it's the first thing I'd revisit before
production use on a larger, noisier ticket corpus.

## LLM interface: pluggable, stub by default, real implementation included

Every LLM call goes through one interface (`llm.py::LLMClient`), with
two implementations:

- **`StubLLMClient`** (default for `demo.py`, `eval.py`, and most
  tests) — fast, free, deterministic, rule-based. This exists so the
  pipeline is fully runnable and testable with zero API keys or
  cost. It is **not** a stand-in for real model behavior — it's a
  test double, same as you'd use for any external dependency.

- **`RealLLMClient`** — the actual implementation: calls an
  OpenAI-compatible chat model (OpenAI or Azure OpenAI, via
  LangChain) with structured JSON-mode prompts, validates the
  response, retries on transient failures, and falls back to the
  stub's rule-based logic if the model returns something malformed
  rather than crashing.

**Important honesty note:** the parsing, validation, retry, and
fallback logic in `RealLLMClient` is unit-tested against a mock chat
model (`tests/test_real_llm_client.py`, 8 tests, no API key needed) —
but a *live* call to a real OpenAI/Azure endpoint has not been run by
whoever last edited this file without also running
`verify_live_llm.py` themselves. Do not describe this project as a
verified end-to-end LLM integration until you've run that script with
your own key and read its output.

```bash
export OPENAI_API_KEY=sk-...
python verify_live_llm.py --provider openai --model gpt-4o-mini
```

You can also test it live from the browser: `streamlit run app.py`,
then choose "OpenAI" or "Azure OpenAI" in the sidebar and enter your
key there (used only for that session, never stored or logged).

## Project structure

```
triage_agent/
  agent.py       # orchestration + the context-sufficiency heuristic
  retriever.py   # TF-IDF similarity search over past tickets
  llm.py         # pluggable LLM interface (stub + real LangChain implementation)
data/
  resolved_tickets.json   # sample past-incident knowledge base
tests/
  test_agent.py            # unit tests for classification, heuristic, retrieval
  test_real_llm_client.py  # unit tests for RealLLMClient's parsing/fallback (mocked, no key)
demo.py             # runnable end-to-end demo (CLI, uses the stub)
eval.py              # small labeled eval harness (accuracy + precision, uses the stub)
app.py               # Streamlit UI -- switch between stub and real LLM live, with your own key
verify_live_llm.py   # run yourself with a real API key to confirm live calls actually work
```

`app.py` is intentionally a thin wrapper: it imports `TriageAgent` and
`TicketRetriever` from `triage_agent/` and renders whatever
`agent.triage()` returns. It contains zero classification, retrieval,
or heuristic logic itself — that separation means the core pipeline
stays fully testable and usable from a plain script even though a UI
exists on top of it.

## What's intentionally out of scope

This is a portfolio-sized project, not a production system. Not
included: persistence/database layer, auth, a real ServiceNow/Jira
integration, retry/backoff around live LLM calls, and monitoring —
all things a production version would need, deliberately left out
here to keep the focus on the retrieval + heuristic design decisions
above.
