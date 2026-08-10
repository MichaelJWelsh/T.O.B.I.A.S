import pytest

from tobias.config import settings
from tobias.stt.interrupt import _request


@pytest.fixture(autouse=True)
def wake_word(monkeypatch):
    monkeypatch.setattr(settings, "stt_wake_word", "Tobias")


@pytest.mark.parametrize(
    "heard",
    [
        "Tobias",
        "Tobias stop",
        "Tobias, stop.",
        "Tobias stop talking",
        "Tobias, shut up!",
        "Tobias be quiet",
        "Tobias, that's enough",
        "Tobias never mind",
        "Tobias, wait.",
        "TOBIAS QUIET",
        # Whisper inflects and mishears; a phrase list missed every one of these.
        "Tobias stopped.",
        "Tobias, stopped",
        "Tobias stop talking now",
        "Tobias, ok stop",
        "Tobias please just stop",
        "Tobias, that's enough now",
    ],
)
def test_being_told_to_be_quiet_carries_no_question(heard):
    assert _request(heard) == "", f"{heard!r} should stop him, not get a reply about stopping"


@pytest.mark.parametrize(
    "heard",
    [
        "Tobias, what is the weather?",
        "Tobias stop and tell me the time",
        "Tobias, never mind that, what is the time?",
        "Tobias, quiet down the music",
        "Tobias, stop and tell me about the weather",
        "Tobias, wait — what did you say about London?",
    ],
)
def test_a_redirect_becomes_the_next_question(heard):
    assert _request(heard) == heard, f"{heard!r} asks for something and should be answered"


def test_nothing_heard_is_not_a_question():
    assert _request("") == ""
