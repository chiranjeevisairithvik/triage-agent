"""
Streamlit UI for the Ticket Triage Agent.

Run: streamlit run app.py

Presentation layer only -- see triage_agent/ for all triage logic.
Includes a sidebar toggle so you can test the REAL LLM path live,
using your own API key entered at runtime (never hardcoded or
persisted). This is how you verify RealLLMClient actually works
end-to-end, without editing any code.
"""
import streamlit as st

from triage_agent import TriageAgent, TicketRetriever, StubLLMClient, RealLLMClient

st.set_page_config(page_title="Ticket Triage Agent", page_icon="🎫", layout="centered")


@st.cache_resource
def load_retriever() -> TicketRetriever:
    return TicketRetriever("data/resolved_tickets.json")


def build_llm_client(backend: str, **kwargs):
    if backend == "Stub (offline, no key needed)":
        return StubLLMClient()

    if backend == "OpenAI":
        from langchain_openai import ChatOpenAI

        chat_model = ChatOpenAI(model=kwargs["model"], api_key=kwargs["api_key"], temperature=0)
        return RealLLMClient(chat_model=chat_model)

    if backend == "Azure OpenAI":
        from langchain_openai import AzureChatOpenAI

        chat_model = AzureChatOpenAI(
            azure_endpoint=kwargs["azure_endpoint"],
            azure_deployment=kwargs["azure_deployment"],
            api_version=kwargs["azure_api_version"],
            api_key=kwargs["api_key"],
            temperature=0,
        )
        return RealLLMClient(chat_model=chat_model)

    raise ValueError(f"Unknown backend: {backend}")


st.title("🎫 Ticket Triage Agent")
st.caption(
    "Classifies a ticket, asks for more context if it's too vague, "
    "or suggests a resolution grounded in similar past incidents."
)

with st.sidebar:
    st.header("LLM backend")
    backend = st.radio(
        "Choose which implementation handles classification and suggestions:",
        ["Stub (offline, no key needed)", "OpenAI", "Azure OpenAI"],
    )

    llm_kwargs = {}
    if backend == "OpenAI":
        llm_kwargs["api_key"] = st.text_input("OpenAI API key", type="password")
        llm_kwargs["model"] = st.text_input("Model", value="gpt-4o-mini")
        st.caption("Your key is used only for this session and never stored.")
    elif backend == "Azure OpenAI":
        llm_kwargs["api_key"] = st.text_input("Azure OpenAI API key", type="password")
        llm_kwargs["azure_endpoint"] = st.text_input("Azure endpoint", placeholder="https://YOUR-RESOURCE.openai.azure.com")
        llm_kwargs["azure_deployment"] = st.text_input("Deployment name")
        llm_kwargs["azure_api_version"] = st.text_input("API version", value="2024-08-01-preview")
        st.caption("Your key is used only for this session and never stored.")
    else:
        st.caption("Rule-based, deterministic, free. Not a real model -- see WALKTHROUGH.md.")

with st.expander("ℹ️ How this works", expanded=False):
    st.markdown(
        """
        1. **Classify** — is this ticket already closed, open, or too vague to act on?
        2. **Context check** — if it's short *and* generic (no error codes/specifics), ask a clarifying question instead of guessing.
        3. **Retrieve** — search past resolved tickets for similar precedent, using TF-IDF similarity.
        4. **Suggest** — summarize the matched precedent's resolution steps.

        Switch to a real LLM backend in the sidebar to see actual model output instead of the rule-based stub.
        """
    )

st.divider()

with st.form("ticket_form"):
    title = st.text_input("Ticket title", placeholder="e.g. Cannot access shared drive")
    description = st.text_area(
        "Ticket description",
        placeholder="e.g. After resetting my password this morning I can no longer open "
        "the shared drive, even though VPN and email work fine.",
        height=120,
    )
    submitted = st.form_submit_button("Triage ticket", use_container_width=True)

if submitted:
    if not title.strip() or not description.strip():
        st.warning("Please fill in both the title and description.")
    elif backend != "Stub (offline, no key needed)" and not llm_kwargs.get("api_key"):
        st.warning("Enter an API key in the sidebar, or switch to the Stub backend.")
    else:
        try:
            llm = build_llm_client(backend, **llm_kwargs)
        except Exception as exc:
            st.error(f"Could not initialize the {backend} client: {exc}")
            st.stop()

        retriever = load_retriever()
        agent = TriageAgent(retriever=retriever, llm=llm)

        with st.spinner("Triaging..."):
            result = agent.triage(ticket_id="UI-TICKET", title=title, description=description)

        st.divider()

        if result.status == "closed":
            st.success("✅ **Status: Closed** — no action needed, this reads as already resolved.")

        elif result.status == "needs-info":
            st.info("❓ **Status: Needs more info**")
            st.markdown(f"**Suggested clarifying question:**\n\n> {result.clarifying_question}")

        else:  # resolved-suggestion
            st.success("💡 **Status: Resolution suggested**")
            st.markdown("**Suggested steps:**")
            st.markdown(result.suggestion)
            if result.precedents_used:
                st.caption(f"Grounded in past incident(s): {', '.join(result.precedents_used)}")
            else:
                st.caption("No sufficiently similar past incident was found.")

st.divider()
st.caption(
    "Portfolio project — pipeline logic in `triage_agent/`, this file is presentation only. "
    "See WALKTHROUGH.md for a full technical breakdown."
)
