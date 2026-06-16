from focuslens.context.active_app import AppContext
from focuslens.context.activity import (
    ActivityCategory,
    ActivityClassifier,
    is_distracting_activity,
)


def _ctx(app="", title="", available=True):
    return AppContext(app_name=app, window_title=title, available=available)


def test_unavailable_context_is_unknown():
    assert ActivityClassifier().classify(_ctx(available=False)) == ActivityCategory.UNKNOWN


def test_empty_app_is_idle():
    assert ActivityClassifier().classify(_ctx(app="")) == ActivityCategory.IDLE


def test_editor_is_work():
    assert ActivityClassifier().classify(_ctx("Code", "main.py — focus-lens")) == (
        ActivityCategory.WORK
    )


def test_terminal_is_work():
    assert ActivityClassifier().classify(_ctx("iTerm2", "zsh")) == ActivityCategory.WORK


def test_social_app_is_social():
    assert ActivityClassifier().classify(_ctx("Safari", "reddit — front page")) == (
        ActivityCategory.SOCIAL
    )


def test_youtube_in_browser_is_entertainment_not_browsing():
    # Content keywords win over the generic browser bucket.
    assert ActivityClassifier().classify(_ctx("Google Chrome", "lo-fi beats - YouTube")) == (
        ActivityCategory.ENTERTAINMENT
    )


def test_steam_is_gaming():
    assert ActivityClassifier().classify(_ctx("Steam", "Library")) == ActivityCategory.GAMING


def test_slack_is_communication():
    assert ActivityClassifier().classify(_ctx("Slack", "general")) == (
        ActivityCategory.COMMUNICATION
    )


def test_unrecognised_browser_is_browsing():
    assert ActivityClassifier().classify(_ctx("Firefox", "some blog")) == (
        ActivityCategory.BROWSING
    )


def test_distracting_activity_set():
    assert is_distracting_activity(ActivityCategory.SOCIAL)
    assert is_distracting_activity(ActivityCategory.GAMING)
    assert not is_distracting_activity(ActivityCategory.WORK)
    assert not is_distracting_activity(ActivityCategory.UNKNOWN)
