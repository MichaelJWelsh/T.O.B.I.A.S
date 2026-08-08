import sys
from functools import cache

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

import tobias.llm
from tobias.llm import CONVERSATION
from tobias.llm.graph import SPEAKS

# `tobias/llm/__init__.py` binds `graph` at import time, so the patch has to replace that name in
# the package namespace. Patching the submodule leaves ask_stream holding the original — and the
# test then talks to the real API.
PACKAGE = sys.modules["tobias.llm"]


def two_node_graph():
    """A graph shaped like phase 1.5: something thinks first, then the spoken node replies."""

    def thinking(state: MessagesState) -> dict:
        model = GenericFakeChatModel(messages=iter([AIMessage('{"tool": "get_weather"}')]))
        return {"messages": [model.invoke(state["messages"])]}

    def reply(state: MessagesState) -> dict:
        model = GenericFakeChatModel(messages=iter([AIMessage("Fifteen degrees, sir.")]))
        return {"messages": [model.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("thinking", thinking)
    builder.add_node(SPEAKS, reply)
    builder.add_edge(START, "thinking")
    builder.add_edge("thinking", SPEAKS)
    return builder.compile(checkpointer=InMemorySaver())


def test_only_the_spoken_node_reaches_tts(monkeypatch):
    monkeypatch.setattr(PACKAGE, "graph", cache(two_node_graph))

    spoken = " ".join(tobias.llm.ask_stream("what's the weather?"))
    assert "Fifteen degrees" in spoken
    assert "tool" not in spoken and "{" not in spoken


def test_without_the_filter_json_would_be_spoken():
    # Guards the guard: if this ever stops holding, the filter is no longer doing anything.
    graph = two_node_graph()
    everything = " ".join(
        chunk.text
        for chunk, _ in graph.stream(
            {"messages": [HumanMessage("hi")]}, CONVERSATION, stream_mode="messages"
        )
    )
    assert "tool" in everything
