from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_shell_mcp import browser_sessions
from local_shell_mcp.browser_sessions import (
    BrowserSessionManager,
    BrowserSessionState,
    close_browser_session_managers,
    get_browser_session_manager,
)
from local_shell_mcp.settings import get_settings


def _stub_browser(monkeypatch):
    calls = []
    closed = asyncio.Event()

    class Page:
        url = "about:blank"

        def is_closed(self):
            return False

        def on(self, *_args):
            pass

        async def title(self):
            return "Synthetic idle test"

    class Context:
        def __init__(self):
            self.pages = []

        async def new_page(self):
            page = Page()
            self.pages.append(page)
            return page

        async def close(self):
            calls.append("context")

    class Browser:
        async def new_context(self, **_kwargs):
            return Context()

        async def close(self):
            calls.append("browser")

    class BrowserType:
        async def launch(self, **_kwargs):
            return Browser()

    class Playwright:
        chromium = BrowserType()

        async def stop(self):
            calls.append("playwright")
            closed.set()

    class Starter:
        async def start(self):
            return Playwright()

    monkeypatch.setattr("playwright.async_api.async_playwright", lambda: Starter())
    monkeypatch.setattr(browser_sessions, "_CLEANUP_INTERVAL_S", 0.01)
    return SimpleNamespace(calls=calls, closed=closed)


async def _eventually(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=1)


def _chromium_installed() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            return Path(playwright.chromium.executable_path).is_file()
    except Exception:
        return False


requires_chromium = pytest.mark.skipif(
    not _chromium_installed(), reason="Playwright Chromium is not installed in this test environment"
)


@pytest.fixture(autouse=True)
def _settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_SHELL_MCP_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("LOCAL_SHELL_MCP_STATE_DIR", str(tmp_path / ".state"))
    monkeypatch.setenv("LOCAL_SHELL_MCP_AUTH_MODE", "none")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _pages(tmp_path: Path) -> tuple[str, str]:
    second = tmp_path / "second.html"
    second.write_text(
        "<html><head><title>Second</title></head><body><p>second page</p></body></html>",
        encoding="utf-8",
    )
    first = tmp_path / "index.html"
    first.write_text(
        f"""
<html>
<head><title>Browser Test</title></head>
<body>
  <button id="button" onclick="document.querySelector('#status').textContent='clicked'">Click me</button>
  <input id="input" placeholder="type here">
  <select id="select"><option value="a">A</option><option value="b">B</option></select>
  <input id="check" type="checkbox">
  <a id="link" href="{second.as_uri()}">Second page</a>
  <div id="status">ready</div>
  <img src="missing.png">
  <script>
    console.error('console-boom');
    setTimeout(() => {{ throw new Error('page-boom'); }}, 0);
  </script>
</body>
</html>
""".strip(),
        encoding="utf-8",
    )
    return first.as_uri(), second.as_uri()


@requires_chromium
@pytest.mark.asyncio
async def test_browser_session_snapshot_refs_actions_and_storage_state(tmp_path, monkeypatch):
    first_url, second_url = _pages(tmp_path)
    manager = BrowserSessionManager(tmp_path / ".state")

    started = await manager.manage(action="start", url=first_url, width=900, height=700)
    session_id = started["session_id"]
    page_id = started["current_page_id"]
    assert started["browser"] == "chromium"
    assert started["pages"][0]["title"] == "Browser Test"

    await manager.act(session_id, [{"action": "wait", "ms": 30}], page_id=page_id)
    snapshot = await manager.snapshot(
        session_id, page_id=page_id, screenshot=True, max_text_chars=12, max_elements=50
    )
    assert snapshot["text_truncated"] is True
    assert Path(snapshot["screenshot_path"]).is_file()
    assert [row["ref"] for row in snapshot["interactive_elements"][:5]] == [
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
    ]
    assert any(item["kind"] == "console" for item in snapshot["errors"])
    assert any(item["kind"] in {"pageerror", "requestfailed"} for item in snapshot["errors"])

    monkeypatch.setattr(browser_sessions, "_MAX_BROWSER_ARTIFACT_FILES", 2)
    for _ in range(3):
        await manager.snapshot(session_id, page_id=page_id, screenshot=True, include_text=False)
    assert len(list((tmp_path / ".state" / "browser-artifacts").glob("*.png"))) == 2

    acted = await manager.act(
        session_id,
        [
            {"action": "fill", "target": "e2", "value": "hello"},
            {"action": "press", "target": "#input", "key": "End"},
            {"action": "type", "target": "#input", "value": "!"},
            {"action": "select", "target": "#select", "value": "b"},
            {"action": "check", "target": "#check"},
            {"action": "uncheck", "target": "#check"},
            {"action": "hover", "target": "#button"},
            {"action": "click", "target": "e1"},
            {"action": "wait_for_text", "text": "clicked"},
            {"action": "wait", "ms": 1},
        ],
        page_id=page_id,
    )
    assert acted["page_id"] == page_id
    assert acted["results"][0]["target"] == "e2"

    state = await manager._get_session(session_id)
    original_page = state.pages[page_id].page
    assert await original_page.locator("#input").input_value() == "hello!"
    assert await original_page.locator("#select").input_value() == "b"
    assert await original_page.locator("#check").is_checked() is False
    assert await original_page.locator("#status").inner_text() == "clicked"

    await original_page.locator("#link").evaluate(
        """
element => {
  element.setAttribute('name', 'n'.repeat(10000));
  element.setAttribute('aria-label', 'a'.repeat(10000));
  element.href = 'https://example.test/' + 'x'.repeat(10000);
}
"""
    )
    bounded_metadata = await manager.snapshot(
        session_id, page_id=page_id, screenshot=False, include_text=False, max_elements=50
    )
    link = next(item for item in bounded_metadata["interactive_elements"] if item["tag"] == "a")
    assert len(link["name"]) == browser_sessions._MAX_ELEMENT_METADATA_CHARS
    assert len(link["href"]) == browser_sessions._MAX_ELEMENT_METADATA_CHARS
    assert len(link["text"]) <= 500

    new_page = await manager.act(
        session_id,
        [
            {"action": "new_page", "url": second_url},
            {"action": "wait_for_url", "url": second_url},
        ],
    )
    second_page_id = new_page["page_id"]
    assert second_page_id != page_id
    assert new_page["title"] == "Second"
    closed_page = await manager.act(
        session_id, [{"action": "close_page"}], page_id=second_page_id
    )
    assert closed_page["page_id"] == page_id

    navigated = await manager.act(
        session_id,
        [
            {"action": "navigate", "url": second_url, "wait_until": "load"},
            {"action": "wait_for_url", "url": second_url},
        ],
        page_id=page_id,
    )
    assert navigated["title"] == "Second"
    no_text = await manager.snapshot(
        session_id,
        page_id=page_id,
        include_text=False,
        screenshot=False,
        full_page=True,
        max_elements=1,
    )
    assert no_text["text"] is None
    assert no_text["screenshot_path"] is None

    storage_path = "browser-state.json"
    closed = await manager.manage(
        action="close", session_id=session_id, save_storage_state_path=storage_path
    )
    assert closed["closed"] is True
    assert (tmp_path / storage_path).is_file()

    restored = await manager.manage(
        action="start", url=first_url, storage_state_path=storage_path
    )
    assert restored["pages"][0]["title"] == "Browser Test"
    await manager.close(restored["session_id"])


@requires_chromium
@pytest.mark.asyncio
async def test_profiles_lists_cleanup_limits_and_singleton(tmp_path, monkeypatch):
    first_url, _ = _pages(tmp_path)
    manager = BrowserSessionManager(tmp_path / ".state")

    profile = await manager.start(url=first_url, profile_id="profile-1")
    assert profile["profile_id"] == "profile-1"
    assert (tmp_path / ".state" / "browser-profiles" / "profile-1").is_dir()
    listed = await manager.manage(action="list")
    assert [item["session_id"] for item in listed["sessions"]] == [profile["session_id"]]

    same_a = get_browser_session_manager(tmp_path / ".singleton")
    same_b = get_browser_session_manager(tmp_path / ".singleton")
    assert same_a is same_b

    state = await manager._get_session(profile["session_id"])
    state.last_used_at = time.time() - browser_sessions._IDLE_TIMEOUT_S - 1
    assert (await manager.manage(action="list"))["sessions"] == []

    monkeypatch.setattr(browser_sessions, "_MAX_SESSIONS", 1)
    one = await manager.start(url=first_url)
    with pytest.raises(ValueError, match="at most 1 browser sessions"):
        await manager.start(url=first_url)
    cleanup = await manager.manage(action="cleanup")
    assert cleanup["closed"] == 1
    with pytest.raises(ValueError, match="unknown browser session"):
        await manager.close(one["session_id"])


@requires_chromium
@pytest.mark.asyncio
async def test_browser_validation_and_action_errors(tmp_path, monkeypatch):
    first_url, _ = _pages(tmp_path)
    manager = BrowserSessionManager(tmp_path / ".state")

    with pytest.raises(ValueError, match="browser must be"):
        await manager.start(browser="netscape")
    with pytest.raises(ValueError, match="invalid wait_until"):
        await manager.start(wait_until="later")
    with pytest.raises(ValueError, match="profile_id"):
        await manager.start(profile_id="bad profile")
    with pytest.raises(ValueError, match="cannot be combined"):
        await manager.start(profile_id="p", storage_state_path="state.json")
    with pytest.raises(ValueError, match="action must be"):
        await manager.manage(action="wat")
    with pytest.raises(ValueError, match="session_id is required"):
        await manager.manage(action="close")
    with pytest.raises(ValueError, match="unknown browser session"):
        await manager.snapshot("missing")

    started = await manager.start(url=first_url)
    session_id = started["session_id"]
    with pytest.raises(ValueError, match="unknown browser page"):
        await manager.snapshot(session_id, page_id="missing")
    with pytest.raises(ValueError, match="at least one"):
        await manager.act(session_id, [])
    with pytest.raises(ValueError, match="at most"):
        await manager.act(session_id, [{"action": "wait"}] * (browser_sessions._MAX_ACTIONS + 1))
    with pytest.raises(ValueError, match=r"actions\[0\] must be an object"):
        await manager.act(session_id, ["bad"])
    with pytest.raises(ValueError, match=r"actions\[0\]\.action is required"):
        await manager.act(session_id, [{}])
    with pytest.raises(ValueError, match="requires url"):
        await manager.act(session_id, [{"action": "navigate"}])
    with pytest.raises(ValueError, match="invalid wait_until"):
        await manager.act(
            session_id,
            [{"action": "navigate", "url": first_url, "wait_until": "later"}],
        )
    with pytest.raises(ValueError, match="requires text"):
        await manager.act(session_id, [{"action": "wait_for_text"}])
    with pytest.raises(ValueError, match="requires url"):
        await manager.act(session_id, [{"action": "wait_for_url"}])
    with pytest.raises(ValueError, match="requires target"):
        await manager.act(session_id, [{"action": "click"}])
    with pytest.raises(ValueError, match="press requires key"):
        await manager.act(session_id, [{"action": "press", "target": "#input"}])
    with pytest.raises(ValueError, match="unsupported browser action"):
        await manager.act(session_id, [{"action": "dance", "target": "#input"}])

    await manager.act(
        session_id,
        [{"action": "select", "target": "#select", "value": ["a"]}],
        timeout_ms=999999,
    )
    await manager.close(session_id)


@pytest.mark.asyncio
async def test_profile_dot_segments_are_rejected_without_starting_browser(tmp_path):
    manager = BrowserSessionManager(tmp_path / ".state")
    for profile_id in (".", ".."):
        with pytest.raises(ValueError, match="profile_id"):
            await manager.start(profile_id=profile_id)


@pytest.mark.asyncio
async def test_start_reserves_slot_during_launch_and_releases_it_on_cancel(tmp_path, monkeypatch):
    manager = BrowserSessionManager(tmp_path / ".state")
    entered = asyncio.Event()
    release = asyncio.Event()
    stopped = {"value": False}

    class BlockingBrowserType:
        async def launch(self, **_kwargs):
            entered.set()
            await release.wait()
            raise AssertionError("cancelled launch should not resume")

    class FakePlaywright:
        chromium = BlockingBrowserType()
        firefox = BlockingBrowserType()
        webkit = BlockingBrowserType()

        async def stop(self):
            stopped["value"] = True

    class Starter:
        async def start(self):
            return FakePlaywright()

    monkeypatch.setattr(browser_sessions, "_MAX_SESSIONS", 1)
    monkeypatch.setattr("playwright.async_api.async_playwright", lambda: Starter())
    task = asyncio.create_task(manager.start())
    await entered.wait()
    with pytest.raises(ValueError, match="at most 1 browser sessions"):
        await manager.start()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager._starting_sessions == 0
    assert stopped["value"] is True


@pytest.mark.asyncio
async def test_start_cancellation_after_registration_removes_and_closes_session(
    tmp_path, monkeypatch
):
    manager = BrowserSessionManager(tmp_path / ".state")
    summary_entered = asyncio.Event()
    context_closed = {"value": False}
    browser_closed = {"value": False}
    playwright_stopped = {"value": False}

    class FakePage:
        url = "about:blank"

        def is_closed(self):
            return False

        def on(self, *_args):
            return None

        async def title(self):
            return ""

    class FakeContext:
        def __init__(self):
            self.pages = []

        async def new_page(self):
            page = FakePage()
            self.pages.append(page)
            return page

        async def close(self):
            context_closed["value"] = True

    context = FakeContext()

    class FakeBrowser:
        async def new_context(self, **_kwargs):
            return context

        async def close(self):
            browser_closed["value"] = True

    class FakeBrowserType:
        async def launch(self, **_kwargs):
            return FakeBrowser()

    class FakePlaywright:
        chromium = FakeBrowserType()
        firefox = FakeBrowserType()
        webkit = FakeBrowserType()

        async def stop(self):
            playwright_stopped["value"] = True

    class Starter:
        async def start(self):
            return FakePlaywright()

    async def blocked_summary(_session, **_kwargs):
        summary_entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("playwright.async_api.async_playwright", lambda: Starter())
    monkeypatch.setattr(manager, "_session_summary", blocked_summary)
    task = asyncio.create_task(manager.start())
    await summary_entered.wait()
    assert len(manager._sessions) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager._sessions == {}
    assert context_closed["value"] is True
    assert browser_closed["value"] is True
    assert playwright_stopped["value"] is True


@pytest.mark.asyncio
async def test_close_releases_browser_when_storage_state_save_fails(tmp_path):
    manager = BrowserSessionManager(tmp_path / ".state")
    closed = {"context": False, "browser": False, "playwright": False}

    class Context:
        async def storage_state(self, **_kwargs):
            raise RuntimeError("save failed")

        async def close(self):
            closed["context"] = True

    class Browser:
        async def close(self):
            closed["browser"] = True

    class Playwright:
        async def stop(self):
            closed["playwright"] = True

    session = BrowserSessionState(
        session_id="session",
        playwright=Playwright(),
        context=Context(),
        browser=Browser(),
        browser_name="chromium",
        profile_id=None,
        created_at=time.time(),
        last_used_at=time.time(),
    )
    manager._sessions[session.session_id] = session
    with pytest.raises(RuntimeError, match="save failed"):
        await manager.close("session", save_storage_state_path="state.json")
    assert manager._sessions == {}
    assert manager._closing_sessions == {}
    assert closed == {"context": True, "browser": True, "playwright": True}


@pytest.mark.asyncio
async def test_explicit_close_stays_within_capacity_while_inflight(tmp_path, monkeypatch):
    manager = BrowserSessionManager(tmp_path / ".state")
    entered = asyncio.Event()
    release = asyncio.Event()
    session = BrowserSessionState(
        session_id="session",
        playwright=object(),
        context=object(),
        browser=None,
        browser_name="chromium",
        profile_id=None,
        created_at=0,
        last_used_at=0,
    )
    manager._cleanup_pending[session.session_id] = session

    async def blocking_close(_session):
        entered.set()
        await release.wait()
        raise RuntimeError("close failed")

    monkeypatch.setattr(manager, "_close_state", blocking_close)
    close = asyncio.create_task(manager.close("session"))
    await entered.wait()
    assert manager._cleanup_pending == {}
    assert list(manager._closing_sessions) == ["session"]

    monkeypatch.setattr(browser_sessions, "_MAX_SESSIONS", 1)
    with pytest.raises(ValueError, match="at most 1 browser sessions"):
        await manager.start()

    release.set()
    with pytest.raises(RuntimeError, match="close failed"):
        await close
    assert manager._closing_sessions == {}
    assert list(manager._cleanup_pending) == ["session"]


@pytest.mark.asyncio
async def test_cleanup_attempts_all_sessions_and_retains_failed_one(tmp_path, monkeypatch):
    manager = BrowserSessionManager(tmp_path / ".state")
    calls: list[str] = []

    class Context:
        def __init__(self, name: str, *, fail: bool = False):
            self.name = name
            self.fail = fail

        async def close(self):
            calls.append(f"context:{self.name}")
            if self.fail:
                raise RuntimeError("first close failed")

    class Browser:
        def __init__(self, name: str):
            self.name = name

        async def close(self):
            calls.append(f"browser:{self.name}")

    class Playwright:
        def __init__(self, name: str):
            self.name = name

        async def stop(self):
            calls.append(f"playwright:{self.name}")

    def session(name: str, *, fail: bool = False) -> BrowserSessionState:
        return BrowserSessionState(
            session_id=name,
            playwright=Playwright(name),
            context=Context(name, fail=fail),
            browser=Browser(name),
            browser_name="chromium",
            profile_id=None,
            created_at=0,
            last_used_at=0,
        )

    manager._sessions = {"first": session("first", fail=True), "second": session("second")}
    assert await manager._cleanup_idle() == 1
    assert "context:second" in calls
    assert "browser:second" in calls
    assert "playwright:second" in calls
    assert manager._sessions == {}
    assert list(manager._cleanup_pending) == ["first"]

    # A quarantined cleanup failure must not block unrelated browser operations.
    listed = await manager.manage(action="list")
    assert listed == {"sessions": [], "cleanup_pending": ["first"]}

    monkeypatch.setattr(browser_sessions, "_MAX_SESSIONS", 1)
    with pytest.raises(ValueError, match="at most 1 browser sessions"):
        await manager.start()

    with pytest.raises(RuntimeError, match="first close failed"):
        await manager._cleanup_idle(force=True)
    assert list(manager._cleanup_pending) == ["first"]


@pytest.mark.asyncio
async def test_cleanup_inflight_sessions_stay_within_capacity(tmp_path, monkeypatch):
    manager = BrowserSessionManager(tmp_path / ".state")
    entered = asyncio.Event()
    release = asyncio.Event()
    session = BrowserSessionState(
        session_id="pending",
        playwright=object(),
        context=object(),
        browser=None,
        browser_name="chromium",
        profile_id=None,
        created_at=0,
        last_used_at=0,
    )
    manager._cleanup_pending[session.session_id] = session

    async def blocking_close(_session):
        entered.set()
        await release.wait()
        raise RuntimeError("close failed")

    monkeypatch.setattr(manager, "_close_state", blocking_close)
    cleanup = asyncio.create_task(manager._cleanup_idle(force=True))
    await entered.wait()
    assert list(manager._closing_sessions) == ["pending"]

    monkeypatch.setattr(browser_sessions, "_MAX_SESSIONS", 1)
    with pytest.raises(ValueError, match="at most 1 browser sessions"):
        await manager.start()

    release.set()
    with pytest.raises(RuntimeError, match="close failed"):
        await cleanup
    assert manager._closing_sessions == {}
    assert list(manager._cleanup_pending) == ["pending"]


@pytest.mark.asyncio
async def test_cleanup_moves_candidates_to_closing_without_capacity_gap(tmp_path):
    manager = BrowserSessionManager(tmp_path / ".state")
    gap = asyncio.Event()
    probed = asyncio.Event()
    observed_capacity = []

    class InterleavingLock:
        def __init__(self):
            self._lock = asyncio.Lock()
            self._exits = 0

        async def __aenter__(self):
            await self._lock.acquire()
            return self

        async def __aexit__(self, *_args):
            self._lock.release()
            self._exits += 1
            if self._exits == 1:
                gap.set()
                await probed.wait()

    session = BrowserSessionState(
        session_id="stale",
        playwright=object(),
        context=object(),
        browser=None,
        browser_name="chromium",
        profile_id=None,
        created_at=0,
        last_used_at=0,
    )
    manager._sessions[session.session_id] = session
    manager._lock = InterleavingLock()

    async def probe_capacity():
        await gap.wait()
        async with manager._lock:
            observed_capacity.append(
                len(manager._sessions)
                + len(manager._cleanup_pending)
                + len(manager._closing_sessions)
                + manager._starting_sessions
            )
        probed.set()

    async def close_state(_session):
        return None

    manager._close_state = close_state
    probe = asyncio.create_task(probe_capacity())
    assert await manager._cleanup_idle(force=True) == 1
    await probe
    assert observed_capacity == [1]


@pytest.mark.asyncio
async def test_cleanup_cancellation_retains_earlier_failures(tmp_path, monkeypatch):
    manager = BrowserSessionManager(tmp_path / ".state")
    second_entered = asyncio.Event()

    def session(name: str) -> BrowserSessionState:
        return BrowserSessionState(
            session_id=name,
            playwright=object(),
            context=object(),
            browser=None,
            browser_name="chromium",
            profile_id=None,
            created_at=0,
            last_used_at=0,
        )

    manager._sessions = {"first": session("first"), "second": session("second")}

    async def close_state(state):
        if state.session_id == "first":
            raise RuntimeError("first close failed")
        second_entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(manager, "_close_state", close_state)
    cleanup = asyncio.create_task(manager._cleanup_idle(force=True))
    await second_entered.wait()
    cleanup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cleanup

    assert manager._sessions == {}
    assert manager._closing_sessions == {}
    assert set(manager._cleanup_pending) == {"first", "second"}


@pytest.mark.asyncio
async def test_start_failure_stops_playwright(tmp_path, monkeypatch):
    manager = BrowserSessionManager(tmp_path / ".state")
    stopped = {"value": False}

    class BrokenBrowserType:
        async def launch(self, **_kwargs):
            raise RuntimeError("launch failed")

    class FakePlaywright:
        chromium = BrokenBrowserType()
        firefox = BrokenBrowserType()
        webkit = BrokenBrowserType()

        async def stop(self):
            stopped["value"] = True

    class Starter:
        async def start(self):
            return FakePlaywright()

    monkeypatch.setattr("playwright.async_api.async_playwright", lambda: Starter())
    with pytest.raises(RuntimeError, match="launch failed"):
        await manager.start()
    assert stopped["value"] is True


@pytest.mark.asyncio
async def test_idle_session_closes_without_another_browser_call(tmp_path, monkeypatch):
    browser = _stub_browser(monkeypatch)
    clock = [1000.0]
    monkeypatch.setattr(browser_sessions.time, "time", lambda: clock[0])
    manager = BrowserSessionManager(tmp_path / ".state")
    try:
        await manager.start()
        cleanup_task = manager._cleanup_task
        assert cleanup_task is not None
        clock[0] += browser_sessions._IDLE_TIMEOUT_S + 1
        # No list/snapshot/act/cleanup call triggers this close.
        await asyncio.wait_for(browser.closed.wait(), timeout=1)
        await asyncio.wait_for(cleanup_task, timeout=1)
        assert browser.calls == ["context", "browser", "playwright"]
        assert manager._sessions == {}
        assert manager._cleanup_task is None
    finally:
        await manager.aclose()


@pytest.mark.asyncio
async def test_cleanup_task_is_lazy_unique_and_restarts_after_empty(tmp_path, monkeypatch):
    browser = _stub_browser(monkeypatch)
    manager = BrowserSessionManager(tmp_path / ".state")
    assert manager._cleanup_task is None
    try:
        first, second = await asyncio.gather(manager.start(), manager.start())
        cleanup_task = manager._cleanup_task
        assert cleanup_task is not None
        await manager.close(first["session_id"])
        assert manager._cleanup_task is cleanup_task
        await manager.close(second["session_id"])
        await asyncio.wait_for(cleanup_task, timeout=1)
        assert manager._cleanup_task is None
        third = await manager.start()
        assert manager._cleanup_task is not None
        assert manager._cleanup_task is not cleanup_task
        assert len(browser.calls) == 6
        await manager.close(third["session_id"])
        await _eventually(lambda: manager._cleanup_task is None)
    finally:
        await manager.aclose()


@pytest.mark.parametrize("operation", ["snapshot", "act"])
@pytest.mark.asyncio
async def test_idle_cleanup_protects_running_and_queued_operations(
    tmp_path, monkeypatch, operation
):
    browser = _stub_browser(monkeypatch)
    clock = [1000.0]
    monkeypatch.setattr(browser_sessions.time, "time", lambda: clock[0])
    manager = BrowserSessionManager(tmp_path / ".state")
    entered, release = asyncio.Event(), asyncio.Event()

    async def capture(*_args):
        entered.set()
        await release.wait()
        return []

    async def action(_session, page, *_args):
        entered.set()
        await release.wait()
        return {}, page

    monkeypatch.setattr(manager, "_capture_interactive_elements", capture)
    monkeypatch.setattr(manager, "_run_action", action)
    tasks = []
    try:
        started = await manager.start()
        session_id = started["session_id"]
        state = manager._sessions[session_id]

        def invoke():
            if operation == "snapshot":
                return manager.snapshot(session_id, include_text=False, screenshot=False)
            return manager.act(session_id, [{"action": "wait"}])

        tasks.append(asyncio.create_task(invoke()))
        await entered.wait()
        tasks.append(asyncio.create_task(invoke()))
        await _eventually(lambda: state.active_operations == 2)
        clock[0] += browser_sessions._IDLE_TIMEOUT_S + 1
        await asyncio.sleep(0.04)
        assert not browser.closed.is_set()
        assert state.active_operations == 2
        release.set()
        await asyncio.gather(*tasks)
        assert state.active_operations == 0
        assert state.last_used_at == clock[0]
        await asyncio.sleep(0.04)
        assert not browser.closed.is_set()
        clock[0] += browser_sessions._IDLE_TIMEOUT_S + 1
        await asyncio.wait_for(browser.closed.wait(), timeout=1)
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await manager.aclose()


@pytest.mark.asyncio
async def test_idle_cleanup_protects_registered_start_until_summary_finishes(tmp_path, monkeypatch):
    browser = _stub_browser(monkeypatch)
    clock = [1000.0]
    monkeypatch.setattr(browser_sessions.time, "time", lambda: clock[0])
    manager = BrowserSessionManager(tmp_path / ".state")
    entered, release = asyncio.Event(), asyncio.Event()
    original_summary = manager._session_summary

    async def summary(session, **kwargs):
        entered.set()
        await release.wait()
        return await original_summary(session, **kwargs)

    monkeypatch.setattr(manager, "_session_summary", summary)
    start = asyncio.create_task(manager.start())
    try:
        await entered.wait()
        clock[0] += browser_sessions._IDLE_TIMEOUT_S + 1
        await asyncio.sleep(0.04)
        assert not browser.closed.is_set()
        release.set()
        started = await start
        assert manager._sessions[started["session_id"]].last_used_at == clock[0]
        await asyncio.sleep(0.04)
        assert not browser.closed.is_set()
    finally:
        release.set()
        await asyncio.gather(start, return_exceptions=True)
        await manager.aclose()


@pytest.mark.asyncio
async def test_background_cleanup_retries_a_failed_explicit_close(tmp_path, monkeypatch):
    browser = _stub_browser(monkeypatch)
    manager = BrowserSessionManager(tmp_path / ".state")
    original_close = manager._close_state
    attempts = []

    async def fail_once(session):
        attempts.append(session.session_id)
        if len(attempts) == 1:
            raise RuntimeError("transient close failure")
        await original_close(session)

    monkeypatch.setattr(manager, "_close_state", fail_once)
    try:
        started = await manager.start()
        with pytest.raises(RuntimeError, match="transient close failure"):
            await manager.close(started["session_id"])
        assert started["session_id"] in manager._cleanup_pending
        await asyncio.wait_for(browser.closed.wait(), timeout=1)
        await _eventually(lambda: manager._cleanup_task is None)
        assert attempts == [started["session_id"], started["session_id"]]
        assert manager._cleanup_pending == {}
        assert browser.calls == ["context", "browser", "playwright"]
    finally:
        await manager.aclose()


@pytest.mark.asyncio
async def test_aclose_waits_for_busy_session_and_is_idempotent(tmp_path, monkeypatch):
    browser = _stub_browser(monkeypatch)
    manager = BrowserSessionManager(tmp_path / ".state")
    entered, release = asyncio.Event(), asyncio.Event()

    async def action(_session, page, *_args):
        entered.set()
        await release.wait()
        return {}, page

    monkeypatch.setattr(manager, "_run_action", action)
    started = await manager.start()
    active = asyncio.create_task(manager.act(started["session_id"], [{"action": "wait"}]))
    await entered.wait()
    shutdown = asyncio.create_task(manager.aclose())
    try:
        await _eventually(lambda: manager._closed)
        assert not shutdown.done()
        assert not browser.closed.is_set()
        with pytest.raises(RuntimeError, match="manager is closed"):
            await manager.start()
        release.set()
        await active
        await asyncio.wait_for(shutdown, timeout=1)
        await manager.aclose()
        assert manager._cleanup_task is None
        assert browser.calls == ["context", "browser", "playwright"]
    finally:
        release.set()
        await asyncio.gather(active, shutdown, return_exceptions=True)
        await manager.aclose()


@pytest.mark.asyncio
async def test_aclose_rejects_an_inflight_launch_before_registration(tmp_path, monkeypatch):
    browser = _stub_browser(monkeypatch)
    manager = BrowserSessionManager(tmp_path / ".state")
    entered, release = asyncio.Event(), asyncio.Event()
    original_page = manager._ensure_page

    async def page(session):
        entered.set()
        await release.wait()
        return await original_page(session)

    monkeypatch.setattr(manager, "_ensure_page", page)
    start = asyncio.create_task(manager.start())
    await entered.wait()
    shutdown = asyncio.create_task(manager.aclose())
    try:
        await _eventually(lambda: manager._closed)
        assert not shutdown.done()
        release.set()
        with pytest.raises(RuntimeError, match="manager is closed"):
            await start
        await asyncio.wait_for(shutdown, timeout=1)
        assert manager._sessions == {}
        assert manager._starting_sessions == 0
        assert browser.calls == ["context", "browser", "playwright"]
    finally:
        release.set()
        await asyncio.gather(start, shutdown, return_exceptions=True)
        await manager.aclose()


@pytest.mark.asyncio
async def test_close_all_managers_attempts_other_owners_and_preserves_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(browser_sessions, "_BROWSER_MANAGERS", {})
    first = get_browser_session_manager(tmp_path / "first")
    second = get_browser_session_manager(tmp_path / "second")
    calls = []

    async def failed_close():
        calls.append("first-failed")
        raise RuntimeError("close failure")

    async def successful_close():
        calls.append("second")

    monkeypatch.setattr(first, "aclose", failed_close)
    monkeypatch.setattr(second, "aclose", successful_close)
    with pytest.raises(RuntimeError, match="close failure"):
        await close_browser_session_managers()
    assert calls == ["first-failed", "second"]
    assert list(browser_sessions._BROWSER_MANAGERS.values()) == [first]

    async def retry():
        calls.append("first-retried")

    monkeypatch.setattr(first, "aclose", retry)
    await close_browser_session_managers()
    assert browser_sessions._BROWSER_MANAGERS == {}
    replacement = get_browser_session_manager(tmp_path / "first")
    assert replacement is not first
    await close_browser_session_managers()
