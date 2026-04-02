"""Unit tests for klovis_agent.perception.

Depends on structlog (perception.base imports it at top level).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

structlog = pytest.importorskip("structlog")

from klovis_agent.perception.base import (
    Event,
    EventKind,
    PerceptionResult,
    PerceptionSource,
    perceive,
)
from klovis_agent.perception.inbox import InboxPerceptionSource


class TestEvent:
    def test_basic(self):
        e = Event(source="test", kind=EventKind.MESSAGE, title="Hello")
        assert e.source == "test"
        assert e.detail == ""
        assert e.timestamp > 0

    def test_summary_line(self):
        e = Event(source="inbox", kind=EventKind.REQUEST, title="Do X")
        assert "[inbox:request]" in e.summary_line()


class TestPerceptionResult:
    def test_empty(self):
        r = PerceptionResult()
        assert not r.has_events
        assert r.summary() == "nothing new"
        assert r.as_text() == "No events detected."

    def test_with_events(self):
        events = [
            Event(source="a", kind=EventKind.MESSAGE, title="msg1"),
            Event(source="a", kind=EventKind.NOTIFICATION, title="notif"),
            Event(source="b", kind=EventKind.MENTION, title="mention"),
        ]
        r = PerceptionResult(events=events)
        assert r.has_events
        assert "a: 2 event(s)" in r.summary()
        assert "b: 1 event(s)" in r.summary()

    def test_as_text(self):
        events = [
            Event(source="x", kind=EventKind.MESSAGE, title="hi", detail="details here"),
        ]
        r = PerceptionResult(events=events)
        text = r.as_text()
        assert "[x]" in text
        assert "details here" in text


class FakeSource(PerceptionSource):
    def __init__(self, events: list[Event] | None = None, should_fail: bool = False):
        self._events = events or []
        self._should_fail = should_fail

    @property
    def name(self) -> str:
        return "fake"

    async def poll(self) -> list[Event]:
        if self._should_fail:
            raise RuntimeError("poll failed")
        return self._events


class TestPerceive:
    @pytest.mark.asyncio
    async def test_empty_sources(self):
        result = await perceive([])
        assert not result.has_events

    @pytest.mark.asyncio
    async def test_aggregates_events(self):
        s1 = FakeSource([Event(source="s1", kind=EventKind.MESSAGE, title="a")])
        s2 = FakeSource([Event(source="s2", kind=EventKind.NOTIFICATION, title="b")])
        result = await perceive([s1, s2])
        assert len(result.events) == 2

    @pytest.mark.asyncio
    async def test_handles_errors(self):
        ok = FakeSource([Event(source="ok", kind=EventKind.MESSAGE, title="fine")])
        bad = FakeSource(should_fail=True)
        result = await perceive([ok, bad])
        assert len(result.events) == 1
        assert len(result.errors) == 1
        assert "fake" in result.errors[0]


class TestInboxPerceptionSource:
    @pytest.mark.asyncio
    async def test_empty_inbox(self, tmp_path: Path):
        inbox = InboxPerceptionSource(inbox_dir=tmp_path / "inbox")
        events = await inbox.poll()
        assert events == []

    @pytest.mark.asyncio
    async def test_picks_up_txt_files(self, tmp_path: Path):
        inbox_dir = tmp_path / "inbox"
        inbox_dir.mkdir()
        (inbox_dir / "task.txt").write_text("Do something important")
        inbox = InboxPerceptionSource(inbox_dir=inbox_dir)
        events = await inbox.poll()
        assert len(events) == 1
        assert events[0].kind == EventKind.REQUEST
        assert "Do something important" in events[0].detail

    @pytest.mark.asyncio
    async def test_ignores_non_txt(self, tmp_path: Path):
        inbox_dir = tmp_path / "inbox"
        inbox_dir.mkdir()
        (inbox_dir / "data.json").write_text('{"key": "val"}')
        inbox = InboxPerceptionSource(inbox_dir=inbox_dir)
        events = await inbox.poll()
        assert events == []

    @pytest.mark.asyncio
    async def test_skips_empty_files(self, tmp_path: Path):
        inbox_dir = tmp_path / "inbox"
        inbox_dir.mkdir()
        (inbox_dir / "empty.txt").write_text("")
        inbox = InboxPerceptionSource(inbox_dir=inbox_dir)
        events = await inbox.poll()
        assert events == []

    def test_archive(self, tmp_path: Path):
        inbox_dir = tmp_path / "inbox"
        inbox_dir.mkdir()
        (inbox_dir / "done").mkdir()
        (inbox_dir / "task.txt").write_text("done")
        inbox = InboxPerceptionSource(inbox_dir=inbox_dir)
        inbox.archive("task.txt")
        assert not (inbox_dir / "task.txt").exists()
        done_files = list((inbox_dir / "done").iterdir())
        assert len(done_files) == 1
