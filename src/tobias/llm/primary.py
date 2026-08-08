from functools import cache

from langchain_openai import ChatOpenAI

from tobias.config import settings


@cache
def model() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        presence_penalty=settings.llm_presence_penalty,
        frequency_penalty=settings.llm_frequency_penalty,
        max_tokens=settings.llm_max_tokens,
    )
