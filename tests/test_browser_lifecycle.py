from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager

import pytest
from starlette.applications import Starlette

from local_shell_mcp import http_app, remote, remote_worker_installer, remote_worker_service, tools
from local_shell_mcp.settings import get_settings


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_SHELL_MCP_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("LOCAL_SHELL_MCP_STATE_DIR", str(tmp_path / ".state"))
    monkeypatch.setenv("LOCAL_SHELL_MCP_AUTH_MODE", "none")
    monkeypatch.setenv("LOCAL_SHELL_MCP_UI_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.parametrize("error", [None, RuntimeError, asyncio.CancelledError])
async def test_worker_exit_closes_browsers_before_releasing_worker_lock(monkeypatch, error):
    calls = []

    @contextmanager
    def worker_lock():
        calls.append("lock")
        try:
            yield
        finally:
            calls.append("unlock")

    async def run(*args):
        calls.append("run")
        if error is not None:
            raise error()

    async def close():
        calls.append("close")

    monkeypatch.setattr(remote_worker_service, "worker_run_lock", worker_lock)
    monkeypatch.setattr(remote, "_run_worker_locked", run)
    monkeypatch.setattr(remote, "close_browser_session_managers", close)
    if error is None:
        await remote.run_worker("https://example.test", "invite")
    else:
        with pytest.raises(error):
            await remote.run_worker("https://example.test", "invite")
    assert calls == ["lock", "run", "close", "unlock"]


async def test_worker_upgrade_closes_browsers_before_exec(monkeypatch):
    calls = []

    async def close():
        calls.append("close")

    monkeypatch.setattr(
        remote_worker_installer, "install_or_update_runtime", lambda server: {"version": "5.1.1"}
    )
    monkeypatch.setattr(
        remote_worker_service, "refresh_installed_service_definition", lambda: calls.append("refresh")
    )
    monkeypatch.setattr(remote, "close_browser_session_managers", close)
    monkeypatch.setattr(remote, "_reexec_updated_worker_runtime", lambda: calls.append("exec"))
    await remote._upgrade_worker_runtime("https://example.test", "5.1.1")
    assert calls == ["refresh", "close", "exec"]


@pytest.mark.parametrize("error", [None, RuntimeError, asyncio.CancelledError])
async def test_stdio_owner_closes_browsers_on_exit(monkeypatch, isolated_settings, error):
    calls = []

    async def run(self):
        calls.append("run")
        if error is not None:
            raise error()

    async def close():
        calls.append("close")

    monkeypatch.setattr(tools.FastMCP, "run_stdio_async", run)
    monkeypatch.setattr(tools, "close_browser_session_managers", close)
    mcp = tools.build_mcp()
    if error is None:
        await mcp.run_stdio_async()
    else:
        with pytest.raises(error):
            await mcp.run_stdio_async()
    assert calls == ["run", "close"]


async def test_mcp_client_exit_and_rejected_app_reentry_do_not_close_browsers(
    monkeypatch, isolated_settings
):
    calls = []
    mcp = tools.build_mcp()

    async def close():
        assert mcp.session_manager._task_group is None
        calls.append("close")

    monkeypatch.setattr(tools, "close_browser_session_managers", close)
    app = mcp.streamable_http_app()
    duplicate_app = mcp.streamable_http_app()
    async with app.router.lifespan_context(app):
        # Low-level MCP lifespans belong to individual transport/client sessions.
        async with mcp._mcp_server.lifespan(mcp._mcp_server):
            pass
        assert calls == []
        with pytest.raises(RuntimeError, match="only be called once"):
            async with duplicate_app.router.lifespan_context(duplicate_app):
                pytest.fail("reused SDK session manager started")
        assert calls == []
    assert calls == ["close"]


@pytest.mark.parametrize("fail", [False, True])
async def test_mcp_app_lifespan_preserves_state_and_exception(monkeypatch, isolated_settings, fail):
    calls = []
    state = {"existing": object()}

    @asynccontextmanager
    async def original_lifespan(app):
        calls.append("startup")
        try:
            yield state
        finally:
            calls.append("shutdown")

    async def close():
        calls.append("close")

    monkeypatch.setattr(
        tools.FastMCP, "streamable_http_app", lambda self: Starlette(lifespan=original_lifespan)
    )
    monkeypatch.setattr(tools, "close_browser_session_managers", close)
    app = tools.build_mcp().streamable_http_app()

    async def use_app():
        async with app.router.lifespan_context(app) as actual_state:
            assert actual_state is state
            if fail:
                raise RuntimeError("application failure")

    if fail:
        with pytest.raises(RuntimeError, match="application failure"):
            await use_app()
    else:
        await use_app()
    assert calls == ["startup", "shutdown", "close"]


async def test_sequential_mcp_application_instances_each_close_once(monkeypatch, isolated_settings):
    calls = []

    async def close():
        calls.append("close")

    monkeypatch.setattr(tools, "close_browser_session_managers", close)
    for _ in range(2):
        app = tools.build_mcp().streamable_http_app()
        async with app.router.lifespan_context(app):
            pass
    assert calls == ["close", "close"]


@pytest.mark.parametrize("fail", [False, True])
async def test_rest_lifespan_preserves_router_hooks_and_exception(
    monkeypatch, isolated_settings, fail
):
    calls = []

    async def startup():
        calls.append("startup")

    async def shutdown():
        calls.append("shutdown")

    async def close():
        calls.append("close")

    monkeypatch.setattr(http_app, "close_browser_session_managers", close)
    app = http_app.build_http_app()
    app.router.on_startup.append(startup)
    app.router.on_shutdown.append(shutdown)

    async def use_app():
        async with app.router.lifespan_context(app):
            if fail:
                raise RuntimeError("application failure")

    if fail:
        with pytest.raises(RuntimeError, match="application failure"):
            await use_app()
    else:
        await use_app()
    assert calls == ["startup", "shutdown", "close"]


async def test_rest_failed_startup_does_not_close_process_browsers(monkeypatch, isolated_settings):
    calls = []

    async def startup():
        raise RuntimeError("startup failed")

    async def close():
        calls.append("close")

    monkeypatch.setattr(http_app, "close_browser_session_managers", close)
    app = http_app.build_http_app()
    app.router.on_startup.append(startup)
    with pytest.raises(RuntimeError, match="startup failed"):
        async with app.router.lifespan_context(app):
            pytest.fail("failed startup entered lifespan")
    assert calls == []
