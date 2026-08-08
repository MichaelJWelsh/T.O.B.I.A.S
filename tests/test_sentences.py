from tobias.llm.sentences import into_sentences


def spoken(text: str) -> list[str]:
    # One character at a time, which is how token fragments actually arrive.
    return list(into_sentences(iter(text)))


def test_splits_on_terminal_punctuation():
    assert spoken("Good evening, sir. The weather is fine. Nothing else is on.") == [
        "Good evening, sir.",
        "The weather is fine.",
        "Nothing else is on.",
    ]


def test_abbreviation_does_not_split_the_sentence():
    assert spoken("Mr. Smith rang about the meeting. He said three.") == [
        "Mr. Smith rang about the meeting.",
        "He said three.",
    ]


def test_ellipsis_is_a_pause_not_a_boundary():
    assert spoken("Well... if you insist, sir.") == ["Well... if you insist, sir."]


def test_trailing_text_without_punctuation_is_still_emitted():
    assert spoken("no full stop here") == ["no full stop here"]


def test_question_and_exclamation_end_sentences():
    assert spoken("Shall I read them out? Of course I shall!") == [
        "Shall I read them out?",
        "Of course I shall!",
    ]


def test_nothing_is_lost_or_duplicated():
    for text in (
        "Good evening, sir. The weather is fine.",
        "Mr. Smith rang. He said half past three, apparently.",
        "Yes.",
        "Really? At this hour! Well... if you insist.",
        "no punctuation at all",
    ):
        assert "".join(spoken(text)).replace(" ", "") == text.replace(" ", "")


def test_empty_stream_yields_nothing():
    assert spoken("") == []
