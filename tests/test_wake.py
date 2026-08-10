import pytest

from tobias.config import settings
from tobias.stt.wake import addressed


@pytest.fixture(autouse=True)
def wake_word(monkeypatch):
    monkeypatch.setattr(settings, "stt_wake_word", "Tobias")


def test_named_at_the_start_keeps_everything():
    assert addressed("Tobias, what is the weather?") == "Tobias, what is the weather?"


def test_named_mid_sentence_still_counts():
    assert addressed("Right then Tobias, what is the weather?") == "Right then Tobias, what is the weather?"


def test_anything_before_the_naming_sentence_is_dropped():
    assert addressed("Now where did I put my keys. Tobias, what is the weather?") == (
        "Tobias, what is the weather?"
    )


def test_everything_after_the_naming_sentence_is_kept():
    assert addressed("Hmm. Tobias, what is the weather? And the time?") == (
        "Tobias, what is the weather? And the time?"
    )


def test_not_named_is_not_addressed():
    assert addressed("What is the weather like today?") is None


def test_matching_ignores_case_and_trailing_punctuation():
    assert addressed("TOBIAS. Weather?") == "TOBIAS. Weather?"
    assert addressed("tobias, weather?") == "tobias, weather?"


def test_the_name_must_be_a_whole_word():
    assert addressed("The Tobiases are visiting on Sunday.") is None


def test_an_empty_wake_word_disables_the_gate(monkeypatch):
    monkeypatch.setattr(settings, "stt_wake_word", "")
    assert addressed("no name here at all") == "no name here at all"


def test_only_the_first_naming_sentence_marks_the_start():
    assert addressed("One. Tobias two. Three Tobias four.") == "Tobias two. Three Tobias four."
