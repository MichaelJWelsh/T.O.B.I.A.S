import json

import pytest

from tobias import config
from tobias.config import Settings


@pytest.fixture
def state(tmp_path, monkeypatch):
    """Point the state file somewhere disposable, and restore the live singleton afterwards."""
    path = tmp_path / "state.json"
    monkeypatch.setattr(config, "STATE_FILE", path)
    monkeypatch.setitem(Settings.model_config, "json_file", path)
    dials = {f: getattr(config.settings, f) for f in ("llm_sarcasm", "llm_warmth", "llm_anxiety")}
    yield path
    for field, value in dials.items():
        setattr(config.settings, field, value)


def test_update_changes_the_live_settings_immediately(state):
    config.update(llm_sarcasm=2)
    assert config.settings.llm_sarcasm == 2


def test_update_persists_only_what_changed(state):
    config.update(llm_sarcasm=2)
    assert json.loads(state.read_text()) == {"llm_sarcasm": 2}


def test_the_api_key_is_never_written(state):
    config.update(llm_sarcasm=2, llm_warmth=9)
    assert "openai_api_key" not in state.read_text()


def test_successive_updates_merge_rather_than_replace(state):
    config.update(llm_sarcasm=2)
    config.update(llm_anxiety=9)
    assert json.loads(state.read_text()) == {"llm_sarcasm": 2, "llm_anxiety": 9}


def test_a_new_process_reads_the_persisted_value(state):
    state.write_text(json.dumps({"llm_sarcasm": 3}))
    assert Settings().llm_sarcasm == 3


def test_state_outranks_dotenv_but_not_a_real_env_var(state, monkeypatch):
    state.write_text(json.dumps({"llm_sarcasm": 3}))
    assert Settings().llm_sarcasm == 3
    monkeypatch.setenv("LLM_SARCASM", "7")
    assert Settings().llm_sarcasm == 7


def test_missing_state_file_is_not_an_error(state):
    assert not state.exists()
    assert isinstance(Settings().llm_sarcasm, int)


def test_a_bad_value_raises_and_writes_nothing(state):
    with pytest.raises(Exception):
        config.update(llm_sarcasm="not a number")
    assert not state.exists()
