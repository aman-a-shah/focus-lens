"""Map an app/window context to an activity category.

A deliberately simple, transparent keyword matcher over the (lowercased) app name + window
title. The keyword lists live in ``ActivityConfig`` so they're user-editable without code
changes. Content-specific categories (gaming/social/entertainment/communication) are checked
before the generic work/browser buckets so that, say, a YouTube tab inside a browser reads as
ENTERTAINMENT rather than "just a browser".
"""

from __future__ import annotations

from enum import StrEnum

from ..config import ActivityConfig
from .active_app import AppContext


class ActivityCategory(StrEnum):
    """What the foreground app says you're doing."""

    WORK = "WORK"
    COMMUNICATION = "COMMUNICATION"  # Slack/Mail/Zoom — could be work, treated as neutral
    SOCIAL = "SOCIAL"  # Twitter/Reddit/Instagram — doom-scroll territory
    ENTERTAINMENT = "ENTERTAINMENT"  # YouTube/Netflix/Twitch
    GAMING = "GAMING"
    BROWSING = "BROWSING"  # a browser with unrecognised content
    IDLE = "IDLE"  # no foreground app / locked
    UNKNOWN = "UNKNOWN"  # app context unavailable (disabled / off-platform)


# Categories that, on their own, mean "not working" — the classifier escalates on these.
DISTRACTING_ACTIVITIES: frozenset[ActivityCategory] = frozenset(
    {ActivityCategory.SOCIAL, ActivityCategory.ENTERTAINMENT, ActivityCategory.GAMING}
)


def is_distracting_activity(category: ActivityCategory) -> bool:
    return category in DISTRACTING_ACTIVITIES


class ActivityClassifier:
    """Keyword classifier mapping ``AppContext`` → ``ActivityCategory``."""

    def __init__(self, config: ActivityConfig | None = None) -> None:
        self.config = config or ActivityConfig()

    def classify(self, ctx: AppContext) -> ActivityCategory:
        if not ctx.available:
            return ActivityCategory.UNKNOWN
        if not ctx.app_name:
            return ActivityCategory.IDLE

        text = ctx.haystack
        c = self.config

        def hit(keywords: list[str]) -> bool:
            return any(kw in text for kw in keywords)

        # Content-specific first so browser tabs resolve to their real activity.
        if hit(c.gaming_keywords):
            return ActivityCategory.GAMING
        if hit(c.social_keywords):
            return ActivityCategory.SOCIAL
        if hit(c.entertainment_keywords):
            return ActivityCategory.ENTERTAINMENT
        if hit(c.communication_keywords):
            return ActivityCategory.COMMUNICATION
        if hit(c.work_keywords):
            return ActivityCategory.WORK
        if hit(c.browser_keywords):
            return ActivityCategory.BROWSING
        return ActivityCategory.UNKNOWN
