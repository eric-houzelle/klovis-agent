"""Unit tests for daemon event filtering."""

from __future__ import annotations

import pytest

structlog = pytest.importorskip("structlog")

from klovis_agent.daemon import AgentDaemon
from klovis_agent.perception.base import Event, EventKind


class TestDecisionEventFiltering:
    def test_keeps_suggested_and_reactions(self):
        events = [
            Event(source="moltbook", kind=EventKind.OTHER, title="Suggested: write a post"),
            Event(source="moltbook", kind=EventKind.REACTION, title="upvoted your post"),
            Event(source="moltbook", kind=EventKind.MENTION, title="@you can you help?"),
        ]
        kept, dropped = AgentDaemon._filter_decision_events(events)
        assert len(kept) == 3
        assert dropped == 0

    def test_keeps_unread_counter(self):
        events = [
            Event(source="moltbook", kind=EventKind.NOTIFICATION, title="12 unread notification(s)"),
            Event(source="moltbook", kind=EventKind.NOTIFICATION, title="alice replied to your comment"),
        ]
        kept, dropped = AgentDaemon._filter_decision_events(events)
        assert len(kept) == 2
        assert dropped == 0
