from focuslens.pipeline import _StateDebouncer
from focuslens.simulate import run_simulation
from focuslens.states import DistractionState as S


def test_debouncer_commits_after_hold():
    deb = _StateDebouncer(hold=2, initial=S.FOCUSED)
    assert deb.update(S.DRIFTING) == (S.FOCUSED, False)  # 1st occurrence
    assert deb.update(S.DRIFTING) == (S.DRIFTING, True)  # 2nd -> commit


def test_debouncer_resets_on_flapping():
    deb = _StateDebouncer(hold=2, initial=S.FOCUSED)
    assert deb.update(S.DRIFTING) == (S.FOCUSED, False)
    # different candidate resets the counter, so no commit yet
    assert deb.update(S.DISTRACTED) == (S.FOCUSED, False)
    assert deb.update(S.DISTRACTED) == (S.DISTRACTED, True)


def test_simulation_visits_all_states_and_logs():
    summary = run_simulation(seed=0)
    assert summary.frames == 660
    assert summary.windows == summary.db_window_count  # every window persisted
    assert set(summary.state_histogram) == {"FOCUSED", "DRIFTING", "DISTRACTED", "FATIGUED"}
    assert summary.notifications >= 1


def test_simulation_is_deterministic():
    a = run_simulation(seed=0)
    b = run_simulation(seed=0)
    assert a.transitions == b.transitions
    assert a.state_histogram == b.state_histogram
