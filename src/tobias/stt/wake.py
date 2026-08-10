import re

from tobias.config import settings

# Split after terminal punctuation, keeping it with the sentence it closes. This works on a
# finished transcript; llm/sentences.py solves the harder problem of a token stream.
BOUNDARY = re.compile(r"(?<=[.!?…])\s+")


def addressed(text: str) -> str | None:
    """Return the request, beginning at the sentence that names TOBIAS, or None if it does not.

    Anything said before that sentence was not meant for him and is dropped — "Right, where did
    I put my keys. Tobias, what's the weather?" asks about the weather, not the keys.
    """
    if not settings.stt_wake_word:
        return text

    name = re.compile(rf"\b{re.escape(settings.stt_wake_word)}\b", re.IGNORECASE)
    sentences = BOUNDARY.split(text)
    for i, sentence in enumerate(sentences):
        if name.search(sentence):
            return " ".join(sentences[i:])
    return None
