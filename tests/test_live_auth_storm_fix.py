from __future__ import annotations

from fastapi import HTTPException
import pytest
from starlette.requests import Request

from local_shell_mcp import auth
from local_shell_mcp.settings import Settings
from local_shell_mcp.live_channel import (
    LIVE_MAX_SECONDARY_CREDENTIALS_PER_CHANNEL,
    LiveChannelManager,
)


def _make_request(path: str, token: str) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [(b"authorization", f"Bearer {token}".encode("utf-8"))],
        "client": ("10.0.0.1", 12345),
        "scheme": "https",
        "server": ("mcp.ranlima.com", 443),
    }
    return Request(scope)


def test_app_reattach_bounds_secondary_credentials_per_channel() -> None:
    manager = LiveChannelManager()
    channel, primary_token = manager.open(
        subject="local-user",
        scopes=("exec",),
        machine="local",
        cwd=".",
    )

    reattach_tokens: list[str] = []
    for _ in range(LIVE_MAX_SECONDARY_CREDENTIALS_PER_CHANNEL + 10):
        reattached, token = manager.open(
            subject="local-user",
            scopes=("exec",),
            live_id=channel.live_id,
            app_reattach=True,
            machine="local",
            cwd=".",
        )
        assert reattached.live_id == channel.live_id
        reattach_tokens.append(token)

    assert len(manager._credentials) == 1 + LIVE_MAX_SECONDARY_CREDENTIALS_PER_CHANNEL
    assert manager.authenticate(primary_token) is not None
    for expired_token in reattach_tokens[:-LIVE_MAX_SECONDARY_CREDENTIALS_PER_CHANNEL]:
        assert manager.authenticate(expired_token) is None
    for active_token in reattach_tokens[-LIVE_MAX_SECONDARY_CREDENTIALS_PER_CHANNEL:]:
        assert manager.authenticate(active_token) is not None


def test_stale_opaque_live_token_rejects_without_oauth_audit_spam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(auth, "get_settings", lambda: Settings(auth_mode="oauth"))
    monkeypatch.setattr(
        auth,
        "audit",
        lambda event, **kwargs: audit_events.append((event, kwargs)),
    )

    with pytest.raises(HTTPException) as exc_info:
        auth.verify_request(_make_request("/api/live/snapshot", "stale-opaque-live-token"))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid live workspace credential"
    assert audit_events == []


def test_malformed_jwt_on_live_route_still_falls_through_to_oauth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(auth, "get_settings", lambda: Settings(auth_mode="oauth"))
    monkeypatch.setattr(
        auth,
        "audit",
        lambda event, **kwargs: audit_events.append((event, kwargs)),
    )

    with pytest.raises(HTTPException) as exc_info:
        auth.verify_request(_make_request("/api/ui/bootstrap", "bad.jwt.token"))

    assert exc_info.value.status_code == 401
    assert any(event == "oauth_auth_failed" for event, _ in audit_events)
