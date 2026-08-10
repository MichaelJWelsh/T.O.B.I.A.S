import sys
from functools import cache

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

import tobias.llm
from tobias.llm import CONVERSATION, ask_stream, interrupted
from tobias.llm.graph import SPEAKS

PACKAGE = sys.modules["tobias.llm"]
REPLY = "One sentence here. Two sentences here. Three sentences here. Four sentences here."


@pytest.fixture
def conversation(monkeypatch):
    """A real graph with a real checkpointer, so update_state behaves as it does in anger."""

    def reply(state: MessagesState) -> dict:
        model = GenericFakeChatModel(messages=iter([AIMessage(REPLY)]))
        return {"messages": [model.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node(SPEAKS, reply)
    builder.add_edge(START, SPEAKS)
    monkeypatch.setattr(PACKAGE, "graph", cache(lambda: builder.compile(checkpointer=InMemorySaver())))

    def messages():
        return (tobias.llm.graph().get_state(CONVERSATION).values or {}).get("messages", [])

    return messages


def test_the_whole_reply_is_recorded_when_he_is_not_cut_off(conversation):
    list(ask_stream("go on"))
    assert conversation()[-1].content == REPLY


def reply_of(messages):
    return next(m for m in messages if isinstance(m, AIMessage))


def test_only_the_spoken_part_survives_an_interruption(conversation):
    stream = ask_stream("go on")
    spoke = next(stream)
    stream.close()

    assert conversation()[-1].content == REPLY, "the model answered in full regardless"
    interrupted()

    stored = reply_of(conversation()).content
    assert stored.startswith(spoke)
    assert "Three sentences" not in stored, "he must not remember what he never said"


def test_the_reply_is_edited_not_duplicated(conversation):
    stream = ask_stream("go on")
    next(stream)
    stream.close()
    before = conversation()[-1].id

    interrupted()
    after = conversation()

    assert len([m for m in after if isinstance(m, AIMessage)]) == 1, "not a second reply"
    assert reply_of(after).id == before, "same id, or add_messages appends instead of replacing"


def test_the_interruption_is_recorded_outside_his_own_voice(conversation):
    # A note appended inside the AIMessage reads to him as self-narration and gets explained
    # away — asked who interrupted him, he answered "not so much an interruption as a
    # self-imposed halt". Stated as a separate message it is a fact rather than something he said.
    stream = ask_stream("go on")
    next(stream)
    stream.close()
    interrupted()

    note = conversation()[-1]
    assert isinstance(note, SystemMessage), "it must not be in his voice"
    assert note.content == tobias.llm.CUT_OFF
    assert tobias.llm.CUT_OFF not in reply_of(conversation()).content


def test_interrupting_before_a_word_was_spoken_changes_nothing(conversation):
    tobias.llm.graph().update_state(CONVERSATION, {"messages": [HumanMessage("hello")]})
    tobias.llm._delivered.clear()
    before = conversation()
    interrupted()
    assert conversation() == before
