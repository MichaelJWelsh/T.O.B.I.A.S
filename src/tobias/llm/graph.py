from functools import cache

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from tobias.llm.primary import model
from tobias.llm.prompts import system_prompt


# The only node whose tokens are meant to be spoken. `stream_mode="messages"` emits tokens from
# every LLM call in the graph, so once routers, tool calls or summarisers exist, anything not from
# here would be read aloud — JSON arguments included. ask_stream() filters on this name.
SPEAKS = "reply"


def _reply(state: MessagesState) -> dict:
    return {"messages": [model().invoke([SystemMessage(system_prompt()), *state["messages"]])]}


@cache
def graph():
    builder = StateGraph(MessagesState)
    builder.add_node(SPEAKS, _reply)
    builder.add_edge(START, SPEAKS)
    # The checkpointer is what gives the conversation a memory; it lives in this process only.
    return builder.compile(checkpointer=InMemorySaver())
