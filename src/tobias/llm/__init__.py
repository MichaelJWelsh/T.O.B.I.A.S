from collections.abc import Iterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tobias.llm.graph import SPEAKS, graph
from tobias.llm.sentences import into_sentences

__all__ = ["ask", "ask_stream", "interrupted"]

# One long-running conversation. History grows unbounded — a phase-2 problem.
CONVERSATION = {"configurable": {"thread_id": "tobias"}}

# What ask_stream has handed to the caller for this reply. Because the caller pulls lazily, this
# is exactly what was spoken: the generator never runs ahead of the speaker.
_delivered: list[str] = []

# A separate message, and deliberately not part of his own reply. A note appended inside the
# AIMessage reads to him as self-narration and gets reinterpreted — asked who interrupted him, he
# answered "not so much an interruption as a self-imposed halt". Stated from outside his voice it
# is a fact about what happened rather than something he apparently said.
CUT_OFF = "The user cut TOBIAS off at this point. Everything he had planned to say after this was never spoken aloud, so the user has not heard it."


def ask(text: str) -> str:
    """Answer one line of conversation, remembering everything said before it."""
    state = graph().invoke({"messages": [HumanMessage(text)]}, CONVERSATION)
    return state["messages"][-1].content


def ask_stream(text: str) -> Iterator[str]:
    """Answer as above, yielding each sentence the moment it is complete rather than at the end."""
    _delivered.clear()
    chunks = (
        chunk.text
        for chunk, meta in graph().stream(
            {"messages": [HumanMessage(text)]}, CONVERSATION, stream_mode="messages"
        )
        if meta.get("langgraph_node") == SPEAKS
    )
    for sentence in into_sentences(chunks):
        _delivered.append(sentence)
        yield sentence


def interrupted() -> None:
    """Trim the last reply in the conversation to the part he actually got to say.

    The API call completes server-side whether or not anyone is still listening, so the graph
    records the whole answer even when playback stopped after one sentence. Left alone he later
    refers to things the user never heard. Replacing the message by its own id is what makes
    this an edit rather than an append — `add_messages` merges on id.
    """
    said = " ".join(_delivered).strip()
    messages = (graph().get_state(CONVERSATION).values or {}).get("messages", [])
    if not said or not messages or not isinstance(messages[-1], AIMessage):
        return

    graph().update_state(
        CONVERSATION,
        {
            "messages": [
                AIMessage(content=said, id=messages[-1].id),  # same id, so this edits
                SystemMessage(content=CUT_OFF),  # new, so this appends
            ]
        },
    )
