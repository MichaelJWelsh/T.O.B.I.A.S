from collections.abc import Iterator

from langchain_core.messages import HumanMessage

from tobias.llm.graph import SPEAKS, graph
from tobias.llm.sentences import into_sentences

__all__ = ["ask", "ask_stream"]

# One long-running conversation. History grows unbounded — a phase-2 problem.
CONVERSATION = {"configurable": {"thread_id": "tobias"}}


def ask(text: str) -> str:
    """Answer one line of conversation, remembering everything said before it."""
    state = graph().invoke({"messages": [HumanMessage(text)]}, CONVERSATION)
    return state["messages"][-1].content


def ask_stream(text: str) -> Iterator[str]:
    """Answer as above, yielding each sentence the moment it is complete rather than at the end."""
    chunks = (
        chunk.text
        for chunk, meta in graph().stream(
            {"messages": [HumanMessage(text)]}, CONVERSATION, stream_mode="messages"
        )
        if meta.get("langgraph_node") == SPEAKS
    )
    return into_sentences(chunks)
