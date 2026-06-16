from focuslens.context.active_app import ActiveAppReader


def test_disabled_reader_is_unavailable():
    reader = ActiveAppReader(enabled=False, runner=lambda: ("Code", "main.py"))
    ctx = reader.read(0.0)
    assert ctx.available is False
    assert ctx.app_name == ""


def test_reader_returns_app_and_title():
    reader = ActiveAppReader(runner=lambda: ("Safari", "reddit"))
    ctx = reader.read(0.0)
    assert ctx.available is True
    assert ctx.app_name == "Safari"
    assert ctx.window_title == "reddit"
    assert ctx.haystack == "safari reddit"


def test_reader_throttles_between_polls():
    calls = {"n": 0}

    def runner():
        calls["n"] += 1
        return (f"App{calls['n']}", "")

    reader = ActiveAppReader(poll_interval_s=1.0, runner=runner)
    first = reader.read(0.0)
    cached = reader.read(0.5)  # within the interval -> no re-poll
    assert first.app_name == cached.app_name == "App1"
    assert calls["n"] == 1
    later = reader.read(1.5)  # past the interval -> re-poll
    assert later.app_name == "App2"
    assert calls["n"] == 2


def test_runner_failure_keeps_last_value():
    state = {"fail": False}

    def runner():
        if state["fail"]:
            return None
        return ("Code", "main.py")

    reader = ActiveAppReader(poll_interval_s=0.0, runner=runner)
    assert reader.read(0.0).app_name == "Code"
    state["fail"] = True
    # A failed poll returns the last cached context rather than going blank.
    assert reader.read(1.0).app_name == "Code"
