import re
from collections.abc import Iterable, Iterator

# Sentence end: terminal punctuation, any closing quote or bracket, then whitespace.
END = re.compile(r"[.!?…][\"')\]]*\s")

# Below this, a "sentence" is almost always an abbreviation (Mr., e.g.) or an initial, and
# speaking it alone sounds clipped. Hold it and let it join the next one.
MIN_CHARS = 16


def into_sentences(chunks: Iterable[str]) -> Iterator[str]:
    """Regroup a stream of token fragments into whole sentences, emitting each as it completes."""
    buffer = ""
    for chunk in chunks:
        buffer += chunk
        start = 0
        while match := END.search(buffer, start):
            if match.end() < MIN_CHARS:
                start = match.end()
                continue
            yield buffer[: match.end()].strip()
            buffer, start = buffer[match.end() :], 0
    if tail := buffer.strip():
        yield tail
