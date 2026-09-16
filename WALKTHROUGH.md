# How This Project Works — Full Walkthrough

This document explains every piece of the Ticket Triage Agent in
detail: what each file does, why it's built that way, and what
happens step-by-step when a ticket runs through the pipeline. Read
this alongside the actual code — the goal is that you can close this
file and still explain any part of the system from memory.

---

## 1. The problem this solves

A support/IT team receives tickets in three rough states:

1. **Already resolved** — someone's just confirming a fix worked. No
   action needed.
2. **Too vague to act on** — "it's broken" tells a human (or an
   agent) nothing about what to actually do.
3. **Actionable** — enough detail to diagnose, and possibly enough
   similarity to a past incident that the fix is already known.

The agent's job is to sort every incoming ticket into one of these
three buckets, and for bucket 3, go one step further and suggest
concrete steps grounded in a real past resolution — not a guess.

---

## 2. The four-stage pipeline, in order

```
title + description
        │
        ▼
┌───────────────────┐
│ 1. Classifier      │  LLMClient.classify_ticket()
└───────────────────┘
        │
   status = "closed" / "open" / "needs-info"
        │
    ┌───┴────────────────┐
    │                     │
 closed              not closed
    │                     │
  STOP              ┌─────▼──────────────┐
                     │ 2. Context check    │  TriageAgent._needs_more_context()
                     └─────┬──────────────┘
                           │
                  ┌────────┴─────────┐
                  │                  │
             insufficient        sufficient
                  │                  │
       ┌──────────▼─────┐   ┌────────▼──────────┐
       │ 3a. Ask a       │   │ 3b. Retrieve       │  TicketRetriever.find_similar()
       │ clarifying      │   │ similar past       │
       │ question        │   │ tickets            │
       └─────────────────┘   └────────┬───────────┘
                                       │
                              ┌────────▼───────────┐
                              │ 4. Suggest a        │  LLMClient.summarize_resolution()
                              │ resolution, grounded │
                              │ in retrieved         │
                              │ precedent            │
                              └──────────────────────┘
```

Every one of these stages is a separate, swappable piece — that
separation is itself a design decision worth being able to explain:
each stage does exactly one job, so you can test, replace, or reason
about it in isolation. This is the difference between "a script that
does the whole thing in one function" and "a system."

---

## 3. File-by-file breakdown

### `triage_agent/llm.py` — the LLM interface

**What it is:** an abstract contract (`LLMClient`) with two
implementations. Every place in the codebase that needs "ask the
model something" goes through this contract instead of calling an
LLM API directly.

**`LLMClient` (abstract base class)** defines three methods every
implementation must provide:
- `classify_ticket(title, description) -> str` — returns `"open"`,
  `"closed"`, or `"needs-info"`
- `draft_clarifying_question(title, description) -> str` — returns a
  question to ask the reporter
- `summarize_resolution(ticket_text, precedents) -> str` — turns a
  list of similar past tickets into suggested steps

**`StubLLMClient`** — the default, no-API-key implementation:
- `classify_ticket`: lowercases the text, checks for closing phrases
  like `"resolved"`, `"fixed"`, `"working now"`. If none found and the
  description has fewer than 8 words, returns `"needs-info"`.
  Otherwise `"open"`.
- `draft_clarifying_question`: simple keyword rules — if the ticket
  mentions "error" but no code/message, it asks for the exact error
  text; if it mentions "slow" or "not working", it asks about timing
  and scope; otherwise a generic prompt for more detail.
- `summarize_resolution`: takes the top-ranked precedent ticket
  (`precedents[0]`), and formats its `root_cause` and
  `resolution_steps` into a readable suggestion. If no precedents
  were passed in, it says so explicitly rather than inventing steps.

**`RealLLMClient`** — the actual implementation, using a real
OpenAI-compatible chat model (OpenAI or Azure OpenAI) via LangChain.
This is meaningfully more involved than "call the model and return
the text," for reasons worth understanding one at a time:

1. **JSON-mode structured output, not free text.** Early on this
   asked the model to "reply with one word" and string-matched the
   reply. That's brittle — a real model might reply `"Open."`,
   `"This is open"`, or `"OPEN"`, and naive string matching breaks on
   all three. `_ask_json()` instead asks for a JSON object with a
   fixed schema (e.g. `{"status": "open"}`) and parses that. Far more
   reliable to consume programmatically, at the cost of a slightly
   more complex prompt.

2. **Validation, not blind trust.** After parsing JSON,
   `classify_ticket()` checks the returned status is actually one of
   `{"open", "closed", "needs-info"}` — a model can return valid JSON
   with an invalid value (`{"status": "urgent"}`), and that has to be
   caught, not passed downstream where `TriageAgent` would silently
   mishandle an unrecognized status.

3. **Fallback, not crash.** If the JSON is malformed, missing keys,
   or the API call fails after retries, `_ask_json()` returns `None`
   and every public method falls back to `StubLLMClient`'s rule-based
   logic for that one call. The design intent: one bad model response
   should degrade the pipeline's *quality* for that ticket, not take
   the whole triage system down.

4. **Retries with backoff** (`_invoke_with_retries`) — a live network
   call can transiently fail (timeout, rate limit); the stub never
   needed this because it's pure local computation.

**Why a stub exists at all — and an honesty point worth being direct
about:** the stub is a legitimate engineering pattern (a deterministic
test double for an external dependency, same reasoning you'd use for
mocking a database in tests). But you should be able to say plainly,
if asked: *"the stub is what runs by default in the tests and CLI
demo; the real classification logic is in RealLLMClient, and here's
how I verified that separately"* — pointing to
`tests/test_real_llm_client.py` (parsing/fallback logic, tested
against a mock model) and `verify_live_llm.py` (a script you run
yourself with a real API key to confirm the live call actually works
end-to-end, not just that the code compiles). Don't claim the live
path is verified unless you've actually run that script and read its
output.

**The key idea to internalize:** `TriageAgent` never knows or cares
which implementation it's talking to. It just calls
`self.llm.classify_ticket(...)`. This is the Dependency Inversion
principle in practice — the orchestration logic depends on an
abstract interface, not a concrete implementation, so swapping stub
for real is a one-line change at the call site, not a rewrite.

---

### `triage_agent/retriever.py` — finding similar past tickets

**What it does:** given a new ticket's text, searches a JSON file of
past resolved tickets (`data/resolved_tickets.json`) and returns the
most similar ones.

**How TF-IDF works (the actual mechanism):**

1. **TF (Term Frequency):** how often a word appears in a document,
   relative to the document's length. A word used 3 times in a
   50-word ticket has higher TF than the same word used 3 times in a
   500-word one.
2. **IDF (Inverse Document Frequency):** how *rare* a word is across
   the whole collection of tickets. Common words like "the" or
   "issue" appear in almost every ticket, so they get a low IDF score
   (low importance). Rare, specific words like "VPN" or "attachment"
   appear in few tickets, so they get a high IDF score (high
   importance).
3. **TF-IDF score** for a word in a document = TF × IDF. A word that's
   frequent in *this* ticket but rare *across all tickets* scores
   highest — this is exactly what you want for distinguishing "what
   is this ticket actually about."
4. Every ticket becomes a vector — one number per unique word in the
   whole corpus, most of them zero (this is why it's called a
   "sparse" vector).
5. **Cosine similarity** measures the angle between two vectors,
   ignoring their length. Two tickets that use similar important
   words point in a similar direction in this high-dimensional space,
   giving a similarity score from 0 (nothing in common) to 1
   (identical word usage).

**In the code, concretely:**
```python
self._vectorizer = TfidfVectorizer(stop_words="english")
self._matrix = self._vectorizer.fit_transform(self._corpus)
```
This builds the vocabulary from all 5 sample tickets and converts
each into a TF-IDF vector, once, when `TicketRetriever` is
constructed (`self._corpus` is `"title description"` for every past
ticket).

```python
query_vec = self._vectorizer.transform([query_text])
scores = cosine_similarity(query_vec, self._matrix)[0]
```
The new ticket is converted into a vector *using the same vocabulary*
(note: `.transform()`, not `.fit_transform()` — the vocabulary is
fixed at this point), then compared against every past ticket's
vector.

```python
ranked = sorted(..., key=lambda pair: pair[0], reverse=True)
return [ticket for score, ticket in ranked[:top_k] if score >= min_similarity]
```
Results are sorted by score, the top `top_k` are kept, but only if
they clear `min_similarity` (0.12 by default). **This threshold is
the second design decision worth understanding**: without it, the
retriever always returns *something*, even for a completely unrelated
query, because cosine similarity always produces some ranking even
among bad matches. The threshold is what lets the system say "I don't
have a good match" instead of confidently returning garbage.

**Why this threshold isn't foolproof — the documented limitation:**
in `eval.py`, the query about a "lunch menu poster" scored 0.396
against `INC-1005` ("laptop slow after Windows **update**") — both
texts share the word "update," and TF-IDF has no way to know
"software update" and "update a poster" are unrelated meanings of the
same word. This is a real, measured failure, not a hypothetical — you
can reproduce it by running `eval.py` yourself and looking at case
`E9`.

---

### `triage_agent/agent.py` — the orchestrator

**`TriageResult`** — a small dataclass holding the outcome: which
ticket, what status, and (depending on status) either a clarifying
question or a suggestion + list of precedent ticket IDs used.

**`TriageAgent.__init__`** — takes a `TicketRetriever` (required) and
an `LLMClient` (optional — defaults to `StubLLMClient()` if you don't
pass one). This default is itself a small design choice: it means
`TriageAgent(retriever=my_retriever)` just works out of the box for
demos and tests, without forcing every caller to explicitly
instantiate a stub.

**`_needs_more_context()` — the heuristic, in full detail:**
```python
word_count = len(description.split())
text = f"{title} {description}".lower()
is_generic = any(phrase in text for phrase in VAGUE_PHRASES) and not any(
    char.isdigit() for char in text
)
return word_count < MIN_DESCRIPTION_WORDS and is_generic
```

Walk through this line by line:
1. `word_count` — a naive whitespace split of the description. Not
   perfect (doesn't handle punctuation specially), but good enough
   for a length signal.
2. `is_generic` has **two** conditions, both must hold:
   - The combined title+description contains at least one of
     `VAGUE_PHRASES = ("doesn't work", "not working", "broken",
     "issue", "problem")` — generic complaint language.
   - **AND** the text contains **no digits at all**. This is the
     clever part: a ticket mentioning "error 809" or "VPN error
     0x80070005" has a digit, so it's immediately excluded from being
     "generic," regardless of how short it is. Error codes are
     almost always genuinely diagnostic information.
3. The final `return` requires **both** `word_count < 8` **and**
   `is_generic` to be true. This "AND" is the actual design decision:

   | Ticket | word count < 8? | is_generic? | needs-info? |
   |---|---|---|---|
   | "It doesn't work." | yes (3) | yes | **yes** |
   | "VPN error 809 today." | yes (4) | no (has digit) | **no** — short but specific |
   | "This has been an ongoing issue with the system for weeks and nobody has looked at it" | no (16 words) | yes | **no** — long but vague; slips through |
   | "After resetting my password I lost access to the shared drive" | no (11) | no | **no** — long and specific |

   That third row is the honest tradeoff: a long-but-vague ticket
   *does* slip past this check and go to retrieval, where it'll
   likely just fail to clear the similarity threshold and return "no
   precedent found." That's an acceptable failure mode — worse than
   catching it early, but much better than the alternative of
   flagging every short ticket (including genuinely clear short ones)
   as needing clarification.

**`triage()` — the main method, tying it together:**
```python
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
```
Note the **early returns** — as soon as a ticket is classified
"closed," the function stops; it never runs the context check or
retrieval on a closed ticket, because there's no reason to. Same
pattern for "needs-info": once we decide to ask a clarifying
question, we don't also try to retrieve precedent for a ticket we
just said doesn't have enough information yet. Every stage only runs
if it's actually needed — no wasted LLM calls or retrieval calls.

---

### `app.py` — Streamlit UI (presentation layer only)

**What it is:** a browser-based front end over the exact same
`TriageAgent` used by `demo.py`, `eval.py`, and the tests. It adds
zero new logic — it's a translation layer between form inputs and
`TriageResult` objects.

**Walking through it:**
```python
@st.cache_resource
def load_agent() -> TriageAgent:
    retriever = TicketRetriever("data/resolved_tickets.json")
    return TriageAgent(retriever=retriever)
```
`@st.cache_resource` matters here: Streamlit re-runs the entire
script top-to-bottom on every user interaction (every button click,
every widget change). Without caching, `TicketRetriever.__init__`
would rebuild the TF-IDF index from scratch on every single click —
wasteful, and would get slow as the ticket corpus grows. Caching
means the index is built once per server session and reused.

```python
with st.form("ticket_form"):
    title = st.text_input(...)
    description = st.text_area(...)
    submitted = st.form_submit_button(...)
```
Using `st.form` instead of bare widgets means the app only re-runs
and calls `agent.triage()` when the button is explicitly clicked —
without a form, Streamlit would re-run (and re-triage) on every
keystroke in the text area, which is both wasteful and would make the
UI feel laggy.

```python
if submitted:
    result = agent.triage(ticket_id="UI-TICKET", title=title, description=description)
```
This is the entire integration point — one method call into the same
`TriageAgent` class covered above. Everything after this is just
`if/elif` branching on `result.status` to decide what to render
(green success box for closed/resolved, blue info box for
needs-info).

**Why this separation matters (and is worth saying explicitly in an
interview):** the moment UI code and business logic mix together in
one file, the logic becomes hard to test (you'd need to simulate a
browser to test the heuristic) and hard to reuse (you couldn't call
it from a script or a different UI without dragging Streamlit along).
Keeping `app.py` as a thin, logic-free layer means `agent.py` and
`retriever.py` don't know or care that a UI exists at all.

### `demo.py` — a runnable, human-readable example

Three hardcoded tickets, one per outcome type (resolved-suggestion,
needs-info, closed), run through the pipeline with printed output.
This exists so anyone (including you, six months from now) can run
one command and see the whole system work without reading test code.

### `eval.py` — measuring, not just running

Ten hand-labeled test cases with `expected_status` and (where
relevant) `expected_precedent`. The script runs each through the
agent and tallies:
- **Classification accuracy** — did `result.status` match what was
  expected, across all 10 cases?
- **Retrieval precision** — of the cases where a resolution was
  expected, did the *top* precedent match the expected one?

This is meaningfully different from `demo.py`: `demo.py` shows the
system working, `eval.py` measures *how well* and gives you a number
you could track over time if you changed the retrieval logic or the
heuristic — e.g., "precision went from 83% to 90% after I added a
stopword list."

### `tests/test_agent.py` — correctness, not performance

Six `pytest` tests, each checking one specific behavior in isolation
(closed detection, vague-short detection, short-but-specific
exemption, grounded suggestion generation, empty retrieval on
unrelated queries, correct retrieval on a relevant query). Unlike
`eval.py`'s aggregate scoring, these are pass/fail assertions meant to
catch a regression — if someone changes `_needs_more_context()` later
and accidentally breaks the "VPN error 809" exemption, `test_short_
but_specific_ticket_is_not_flagged_vague` fails immediately and tells
you exactly what broke.

---

## 4. A full trace: what actually happens for one ticket

Let's trace `NEW-2001` from `demo.py` end to end:

```
title = "Can't access shared drive"
description = "After resetting my password this morning I can no
               longer open the marketing shared drive, even though
               VPN and email work fine."
```

**Step 1 — Classification** (`StubLLMClient.classify_ticket`):
lowercased text is checked against closing phrases
(`"resolved"`, `"fixed"`, `"closed"`, `"no longer an issue"`,
`"working now"`) — none present. Description has 22 words, well over
the 8-word threshold, so status is `"open"`.

**Step 2 — Context check** (`_needs_more_context`): word count is
22, which already fails `< MIN_DESCRIPTION_WORDS (8)` — so this
returns `False` immediately without even checking for vague phrases.
Ticket proceeds to retrieval.

**Step 3 — Retrieval** (`TicketRetriever.find_similar`): the query
text `"Can't access shared drive After resetting my password..."` is
vectorized with the same TF-IDF vocabulary built from the 5 sample
tickets. Cosine similarity against each:
- `INC-1001` ("User cannot access shared drive after password
  reset...") shares many high-IDF words: "shared," "drive,"
  "password," "reset," "access" — high similarity score.
- The others share far fewer distinctive terms — lower scores.

Top 2 results above the 0.12 threshold are returned:
`[INC-1001, INC-1004]` (from the actual demo output above) —
`INC-1004` (email attachments) sneaks in as a weaker secondary match,
which is a reasonable illustration of `top_k=2` sometimes returning a
second, less-relevant result alongside a strong first one.

**Step 4 — Suggestion** (`StubLLMClient.summarize_resolution`): takes
`precedents[0]` (`INC-1001`), and formats its `root_cause` and
`resolution_steps` fields into the final printed suggestion —
exactly what you saw in the demo output: log off/on to refresh
cached credentials, clear the cached credential entry, re-map the
drive.

**Final `TriageResult`:**
```python
TriageResult(
    ticket_id="NEW-2001",
    status="resolved-suggestion",
    suggestion="Based on similar past incident INC-1001...",
    precedents_used=["INC-1001", "INC-1004"],
)
```

---

## 5. Things worth being able to answer in an interview

- **"Why TF-IDF and not embeddings?"** — explainability and zero
  infra cost at this corpus size; documented tradeoff, with a real
  measured failure case (the "update" lexical collision) as evidence
  you understand the limitation, not just the choice.
- **"Why does the heuristic require both conditions, not one?"** —
  walk through the truth table above; the "short but specific" and
  "long but vague" rows are the two failure modes being traded off.
- **"What happens if the LLM call fails or times out?"** — honestly:
  not handled in this version. `LangChainLLMClient` has no retry/error
  handling — this is intentionally listed under "What's out of scope"
  in the README, and is a fair place for an interviewer to push. A
  good answer: "I'd wrap `chat_model.invoke()` with a retry/backoff
  and a fallback to `StubLLMClient`'s rule-based path if the real
  model is unavailable, so triage degrades gracefully instead of
  failing hard."
- **"How would you know if this was working well in production?"** —
  point to `eval.py`'s structure: you'd grow the labeled test set from
  real historical tickets, track accuracy/precision over time, and
  add a human-feedback loop (was the suggestion actually useful?) as
  a third metric.
