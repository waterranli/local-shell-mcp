from __future__ import annotations

import asyncio
import json
from collections import Counter

import pytest
from starlette.exceptions import HTTPException
from starlette.requests import Request

from local_shell_mcp import human_ui as ui
from local_shell_mcp.auth import Principal
from local_shell_mcp.settings import get_settings


@pytest.fixture(autouse=True)
def isolated_ui_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_SHELL_MCP_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("LOCAL_SHELL_MCP_STATE_DIR", str(tmp_path / ".state"))
    monkeypatch.setenv("LOCAL_SHELL_MCP_AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("LOCAL_SHELL_MCP_AUTH_MODE", "oauth")
    monkeypatch.setenv("LOCAL_SHELL_MCP_REMOTE_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _request(subject="alice", *, scopes="shell:read remote:use", live_id=None):
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/ui/terminals",
            "headers": [],
            "query_string": b"machine=node",
        }
    )
    claims = {"auth": "oauth", "scope": scopes}
    if live_id is not None:
        claims.update({"auth": "live-channel", "live_id": live_id})
    request.state.principal = Principal(None, subject, claims)
    return request


async def _snapshot(request=None, *, machine="node", tool="shell_list", args=None):
    return await ui._ui_status_snapshot(
        request or _request(), machine, lambda: None, tool, args or {}
    )


async def test_concurrent_status_polls_share_fetch_and_return_independent_snapshots(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def dispatch(*args):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return {"sessions": [{"session_id": "shared"}]}

    monkeypatch.setattr(ui, "_machine_dispatch", dispatch)
    waiters = [asyncio.create_task(_snapshot()) for _ in range(20)]
    await started.wait()
    release.set()
    snapshots = await asyncio.gather(*waiters)
    assert calls == 1
    snapshots[0]["sessions"][0]["session_id"] = "changed by one consumer"
    assert snapshots[1]["sessions"][0]["session_id"] == "shared"
    assert (await _snapshot())["sessions"][0]["session_id"] == "shared"
    assert calls == 1


async def test_cancelled_waiter_cannot_cancel_shared_status_fetch(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0
    cancelled = False

    async def dispatch(*args):
        nonlocal calls, cancelled
        calls += 1
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled = True
            raise
        return {"sessions": []}

    monkeypatch.setattr(ui, "_machine_dispatch", dispatch)
    first = asyncio.create_task(_snapshot())
    await started.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    second = asyncio.create_task(_snapshot())
    release.set()
    assert await second == {"sessions": []}
    assert calls == 1
    assert cancelled is False


async def test_failed_refresh_is_shared_and_retried_only_after_short_backoff(monkeypatch):
    clock = [0.0]
    calls = 0

    async def dispatch(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("worker temporarily unavailable")
        return {"sessions": []}

    monkeypatch.setattr(ui, "_ui_status_now", lambda: clock[0])
    monkeypatch.setattr(ui, "_machine_dispatch", dispatch)
    errors = await asyncio.gather(*(_snapshot() for _ in range(20)), return_exceptions=True)
    assert all(isinstance(error, RuntimeError) for error in errors)
    assert len({id(error) for error in errors}) == 20
    assert calls == 1
    clock[0] = ui._UI_STATUS_FAILURE_TTL_S - 0.01
    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        await _snapshot()
    assert calls == 1
    clock[0] += 0.02
    assert await _snapshot() == {"sessions": []}
    assert calls == 2


async def test_successful_status_expires_and_different_query_parameters_do_not_merge(monkeypatch):
    clock = [0.0]
    calls = 0

    async def dispatch(machine, local_call, tool, args):
        nonlocal calls
        calls += 1
        return {"revision": calls, "include_finished": args["include_finished"]}

    monkeypatch.setattr(ui, "_ui_status_now", lambda: clock[0])
    monkeypatch.setattr(ui, "_machine_dispatch", dispatch)
    assert (await _snapshot(tool="job_list", args={"include_finished": True}))["revision"] == 1
    assert (await _snapshot(tool="job_list", args={"include_finished": False}))["revision"] == 2
    clock[0] = ui._UI_STATUS_TTL_S - 0.01
    assert (await _snapshot(tool="job_list", args={"include_finished": True}))["revision"] == 1
    clock[0] += 0.02
    assert (await _snapshot(tool="job_list", args={"include_finished": True}))["revision"] == 3


@pytest.mark.skipif(not hasattr(asyncio, "eager_task_factory"), reason="Requires Python 3.12+")
async def test_completed_eager_fetch_does_not_bypass_cache_expiry(monkeypatch):
    clock = [0.0]
    calls = 0

    async def dispatch(*args):
        nonlocal calls
        calls += 1
        return {"revision": calls}

    monkeypatch.setattr(ui, "_ui_status_now", lambda: clock[0])
    monkeypatch.setattr(ui, "_machine_dispatch", dispatch)
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()
    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        assert await _snapshot() == {"revision": 1}
        clock[0] += ui._UI_STATUS_TTL_S + 0.01
        assert await _snapshot() == {"revision": 2}
    finally:
        loop.set_task_factory(original_factory)


async def test_cached_status_is_isolated_by_subject_scopes_and_live_channel(monkeypatch):
    calls = 0

    async def dispatch(*args):
        nonlocal calls
        calls += 1
        return {"private_snapshot": calls}

    monkeypatch.setattr(ui, "_machine_dispatch", dispatch)
    alice = _request("alice")
    bob = _request("bob")
    assert await _snapshot(alice) == {"private_snapshot": 1}
    assert await _snapshot(bob) == {"private_snapshot": 2}
    assert await _snapshot(alice) == {"private_snapshot": 1}
    operator = _request("alice", scopes="shell:read shell:execute remote:use")
    assert await _snapshot(operator) == {"private_snapshot": 3}
    assert await _snapshot(_request("alice", live_id="first")) == {"private_snapshot": 4}
    assert await _snapshot(_request("alice", live_id="second")) == {"private_snapshot": 5}
    assert calls == 5


async def test_ui_route_checks_scope_even_after_an_authorized_cache_hit(monkeypatch):
    calls = 0

    async def dispatch(*args):
        nonlocal calls
        calls += 1
        return {"sessions": []}

    monkeypatch.setattr(ui, "_machine_dispatch", dispatch)
    assert (await ui.api_terminals(_request())).status_code == 200
    denied = await ui.api_terminals(_request(scopes="remote:use"))
    assert denied.status_code == 403
    assert calls == 1


async def test_mutations_are_never_cached_and_invalidate_status_for_all_subjects(monkeypatch):
    revision = 0
    calls = Counter()

    async def remote(machine, tool, args, timeout=None):
        nonlocal revision
        calls[tool] += 1
        if tool == "shell_start":
            revision += 1
        return {"sessions": [{"session_id": str(revision)}]}

    monkeypatch.setattr(ui, "_remote_call", remote)
    for subject in ["alice", "bob"]:
        assert (await _snapshot(_request(subject)))["sessions"][0]["session_id"] == "0"
    for _ in range(2):
        await ui._machine_dispatch("node", lambda: None, "shell_start", {"cwd": "."})
        await ui._machine_dispatch("node", lambda: None, "run_shell_tool", {"command": "true"})
        await ui._machine_dispatch("node", lambda: None, "write_file", {"path": "example"})
    for subject in ["alice", "bob"]:
        assert (await _snapshot(_request(subject)))["sessions"][0]["session_id"] == "2"
    assert calls == {"shell_list": 4, "shell_start": 2, "run_shell_tool": 2, "write_file": 2}
    with pytest.raises(ValueError, match="Only terminal and job status"):
        await _snapshot(tool="run_shell_tool", args={"command": "true"})


async def test_mutation_racing_inflight_status_fetch_cannot_repopulate_old_snapshot(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    revision = 0
    reads = 0

    async def remote(machine, tool, args, timeout=None):
        nonlocal revision, reads
        if tool == "shell_start":
            revision += 1
            return {"session_id": str(revision)}
        reads += 1
        captured_revision = revision
        if reads == 1:
            started.set()
            await release.wait()
        return {"sessions": [{"session_id": str(captured_revision)}]}

    monkeypatch.setattr(ui, "_remote_call", remote)
    waiting = asyncio.create_task(_snapshot())
    await started.wait()
    await ui._machine_dispatch("node", lambda: None, "shell_start", {"cwd": "."})
    release.set()
    assert (await waiting)["sessions"][0]["session_id"] == "1"
    assert reads == 2


async def test_continuous_mutations_bound_refresh_attempts_and_apply_failure_backoff(monkeypatch):
    clock = [0.0]
    calls = 0

    async def dispatch(machine, *args):
        nonlocal calls
        calls += 1
        ui._invalidate_ui_status(machine)
        return {"sessions": []}

    monkeypatch.setattr(ui, "_ui_status_now", lambda: clock[0])
    monkeypatch.setattr(ui, "_machine_dispatch", dispatch)
    with pytest.raises(HTTPException) as exc:
        await asyncio.wait_for(_snapshot(), 1)
    assert exc.value.status_code == 503
    assert calls == 2
    ui._invalidate_ui_status("node")
    with pytest.raises(HTTPException):
        await _snapshot()
    assert calls == 2
    clock[0] += ui._UI_STATUS_FAILURE_TTL_S + 0.01
    with pytest.raises(HTTPException):
        await _snapshot()
    assert calls == 4


async def test_mutation_cannot_bypass_failure_backoff(monkeypatch):
    calls = 0

    async def dispatch(machine, *args):
        nonlocal calls
        calls += 1
        ui._invalidate_ui_status(machine)
        raise RuntimeError("worker temporarily unavailable")

    monkeypatch.setattr(ui, "_machine_dispatch", dispatch)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="temporarily unavailable"):
            await _snapshot()
        ui._invalidate_ui_status("node")
    assert calls == 1


async def test_terminal_attachment_validation_bypasses_ui_status_cache(monkeypatch):
    sessions = [{"session_id": "old-session", "backend": "native"}]
    reads = 0

    async def remote(*args):
        nonlocal reads
        reads += 1
        return {"sessions": list(sessions)}

    monkeypatch.setattr(ui, "_remote_call", remote)
    assert (await _snapshot())["sessions"] == sessions
    sessions.clear()
    with pytest.raises(ValueError, match="Persistent terminal session not found"):
        await ui._spawn_shell_process("node", "old-session", 80, 24)
    assert reads == 2


async def test_cache_capacity_bounds_pending_fetches_and_evicts_completed_entries(monkeypatch):
    monkeypatch.setattr(ui, "_UI_STATUS_MAX_ENTRIES", 2)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def dispatch(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            started.set()
        await release.wait()
        return {"sessions": []}

    monkeypatch.setattr(ui, "_machine_dispatch", dispatch)
    waiters = [asyncio.create_task(_snapshot(machine=name)) for name in ["first", "second"]]
    await started.wait()
    try:
        with pytest.raises(HTTPException) as exc:
            await _snapshot(machine="third")
        assert exc.value.status_code == 503
        assert calls == 2
    finally:
        release.set()
        await asyncio.gather(*waiters)
    assert await _snapshot(machine="third") == {"sessions": []}
    assert len(ui._ui_status_entries()) == 2
    assert calls == 3


async def test_dashboard_and_terminal_list_polling_share_the_same_status_fetches(monkeypatch):
    calls = Counter()

    async def dispatch(machine, local_call, tool, args):
        calls[tool] += 1
        await asyncio.sleep(0)
        return {"sessions": []} if tool == "shell_list" else {"jobs": [], "counts": {}}

    monkeypatch.setattr(ui, "_machine_dispatch", dispatch)
    monkeypatch.setattr(ui, "_machine_rows", lambda: {"machines": [], "counts": {}})
    monkeypatch.setattr(ui, "_local_system_snapshot", lambda: {})
    monkeypatch.setattr(ui, "query_audit", lambda **kwargs: {"entries": [], "total_matched": 0})
    responses = await asyncio.gather(
        *(ui.api_dashboard(_request()) for _ in range(10)),
        *(ui.api_terminals(_request()) for _ in range(10)),
    )
    assert all(response.status_code == 200 for response in responses)
    assert all(json.loads(response.body)["ok"] for response in responses)
    assert calls == {"shell_list": 1, "job_list": 1}
