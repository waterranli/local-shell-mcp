"""Regression tests for durable remote-result delivery limits and failures."""

import asyncio
import json
import time

import pytest

from local_shell_mcp import remote
from local_shell_mcp.settings import get_settings


@pytest.fixture
def outbox(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_SHELL_MCP_WORKER_RESULT_OUTBOX_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_SHELL_MCP_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("LOCAL_SHELL_MCP_STATE_DIR", str(tmp_path / ".state"))
    monkeypatch.setattr(remote, "_WORKER_OUTBOX_POLL_INTERVAL_S", 0.005)
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


@pytest.mark.parametrize("limit", ["items", "bytes"])
def test_full_outbox_rejects_new_results_but_allows_existing_result_update(
    outbox, monkeypatch, limit
):
    original = {"job_id": "one", "data": "finished"}
    path = remote._worker_store_result_outbox(original)
    if limit == "items":
        monkeypatch.setattr(remote, "REMOTE_WORKER_RESULT_OUTBOX_MAX_ITEMS", 1)
    else:
        monkeypatch.setattr(remote, "REMOTE_WORKER_RESULT_OUTBOX_MAX_BYTES", path.stat().st_size)
    with pytest.raises(RuntimeError, match="outbox"):
        remote._worker_store_result_outbox({"job_id": "two", "data": "new"})
    updated = {**original, "lifecycle": {"result_submit_started_at": 123}}
    assert remote._worker_store_result_outbox(updated) == path
    assert json.loads(path.read_text(encoding="utf-8")) == updated
    assert list(outbox.glob("*.json")) == [path]
    assert not list(outbox.glob("*.tmp.*"))


def test_outbox_requires_job_id_before_creating_file(outbox):
    with pytest.raises(ValueError, match="job id"):
        remote._worker_store_result_outbox({"data": "orphan"})
    assert list(outbox.iterdir()) == []


@pytest.mark.parametrize("failure", ["read", "submit", "timing"])
async def test_sender_preserves_result_and_releases_claim_after_failure(
    outbox, monkeypatch, failure
):
    path = remote._worker_store_result_outbox({"job_id": "one", "lifecycle": {}})
    if failure == "read":
        path.write_text("[]", encoding="utf-8")
    reached_failure = asyncio.Event()
    logs = []
    in_progress = set()

    def log(operation, error, delay):
        logs.append((operation, str(error)))
        reached_failure.set()

    async def submit(*args, **kwargs):
        if failure == "submit":
            raise ConnectionError("controller offline")
        await asyncio.Event().wait()

    def store_result(result):
        raise OSError("disk unavailable")

    monkeypatch.setattr(remote, "_worker_log_retry", log)
    monkeypatch.setattr(remote, "_submit_worker_result_with_heartbeat", submit)
    if failure == "timing":
        monkeypatch.setattr(remote, "_worker_store_result_outbox", store_result)
    sender = asyncio.create_task(
        remote._worker_result_outbox_sender("https://controller.test", {}, 1, in_progress)
    )
    try:
        await asyncio.wait_for(reached_failure.wait(), 2)
    finally:
        sender.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sender
    assert path.exists()
    assert not in_progress
    expected = {
        "read": "read result outbox",
        "submit": "result outbox submit",
        "timing": "update result outbox timing",
    }
    assert logs[0][0] == expected[failure]


async def test_sender_discards_result_rejected_by_controller(outbox, monkeypatch):
    path = remote._worker_store_result_outbox({"job_id": "expired"})
    discarded = []
    in_progress = set()

    async def submit(*args, **kwargs):
        return {"data": {"accepted": False}}

    monkeypatch.setattr(remote, "_submit_worker_result_with_heartbeat", submit)
    monkeypatch.setattr(remote, "audit", lambda event, **data: discarded.append((event, data)))
    sender = asyncio.create_task(
        remote._worker_result_outbox_sender("https://controller.test", {}, 1, in_progress)
    )

    async def wait_until_drained():
        while path.exists() or in_progress:
            await asyncio.sleep(0.01)

    try:
        await asyncio.wait_for(wait_until_drained(), 2)
    finally:
        sender.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sender
    assert discarded == [("remote_result_outbox_discarded", {"job_id": "expired"})]
    assert not path.exists()
    assert not in_progress


@pytest.mark.parametrize("tool", ["shell_start", "run_shell_tool"])
async def test_spooled_result_retains_reset_generation_and_mutation_preservation(
    outbox, monkeypatch, tool
):
    async def execute(*args):
        return {"done": True}

    monkeypatch.setattr(remote, "_execute_worker_job_with_heartbeat", execute)
    await remote._run_worker_job(
        {"id": "reset-result", "tool": tool, "args": {}, "reset_generation": 3},
        "https://controller.test",
        {},
        0.005,
    )
    path = next(outbox.glob("*.json"))
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["reset_generation"] == 3
    assert bool(persisted.get("preserve_across_reset")) == (tool == "shell_start")
    attempts = 0

    def post(url, payload, headers=None, timeout=None):
        nonlocal attempts
        if url.endswith("/result"):
            attempts += 1
            # Keep the request in flight long enough to receive a reset ACK.
            time.sleep(0.03)
            if attempts < 3:
                raise RuntimeError("temporary result upload failure")
            return {"ok": True, "data": {"accepted": True}}
        return {"ok": True, "data": {"accepted": True, "reset_generation": 4}}

    monkeypatch.setattr(remote, "_worker_post_json", post)
    monkeypatch.setattr(remote, "_WORKER_RETRY_INITIAL_DELAY_S", 0.02)
    monkeypatch.setattr(remote, "_WORKER_RETRY_MAX_DELAY_S", 0.02)
    in_progress = set()
    sender = asyncio.create_task(
        remote._worker_result_outbox_sender("https://controller.test", {}, 0.005, in_progress)
    )

    async def wait_until_drained():
        while path.exists() or in_progress:
            await asyncio.sleep(0.01)

    try:
        await asyncio.wait_for(wait_until_drained(), 3)
    finally:
        sender.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sender
    assert attempts == 3 if tool == "shell_start" else attempts < 3


async def _cancel_senders(*tasks):
    for task in tasks:
        task.cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(result, asyncio.CancelledError) for result in results)


async def test_single_result_attempt_does_not_enter_forever_retry(monkeypatch):
    calls = []

    async def post(*args):
        calls.append(args)
        raise ConnectionError("offline")

    async def forever(*args):
        pytest.fail("spooled delivery must not monopolize a sender with forever retries")

    monkeypatch.setattr(remote, "_worker_post_json_cancellable", post)
    monkeypatch.setattr(remote, "_worker_post_json_forever", forever)
    with pytest.raises(ConnectionError, match="offline"):
        await remote._submit_worker_result_with_heartbeat(
            {"job_id": "one"}, "https://controller.test", {}, 1, single_attempt=True
        )
    assert len(calls) == 1
    assert calls[0][3] == remote._worker_result_request_timeout_s({"job_id": "one"})


async def test_spool_failure_retains_synchronous_forever_fallback(outbox, monkeypatch):
    submissions = []

    async def execute(*args):
        return {"done": True}

    def store(*args):
        raise OSError("disk full")

    async def forever(*args):
        submissions.append(args)
        return {"data": {"accepted": True}}

    monkeypatch.setattr(remote, "_execute_worker_job_with_heartbeat", execute)
    monkeypatch.setattr(remote, "_worker_store_result_outbox", store)
    monkeypatch.setattr(remote, "_worker_post_json_forever", forever)
    await remote._run_worker_job(
        {"id": "one", "tool": "run_shell_tool"}, "https://controller.test", {}, 1
    )
    assert len(submissions) == 1
    assert submissions[0][-1] == "submit result"


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"ok": True},
        {"data": []},
        {"data": {"accepted": 0}},
        {"data": {"accepted": "true"}},
        {"data": {"cancelled": "true"}},
        {"ok": False, "data": {"accepted": True}},
    ],
)
async def test_unacknowledged_result_is_retained(outbox, monkeypatch, response):
    path = remote._worker_store_result_outbox({"job_id": "one"})
    rejected = asyncio.Event()
    in_progress = set()

    async def submit(*args, **kwargs):
        assert kwargs["single_attempt"] is True
        return response

    monkeypatch.setattr(remote, "_submit_worker_result_with_heartbeat", submit)
    monkeypatch.setattr(remote, "_worker_log_retry", lambda *args: rejected.set())
    sender = asyncio.create_task(
        remote._worker_result_outbox_sender("https://controller.test", {}, 1, in_progress)
    )
    try:
        await asyncio.wait_for(rejected.wait(), 2)
    finally:
        await _cancel_senders(sender)
    assert path.exists()
    assert not in_progress


@pytest.mark.parametrize("data", [{"accepted": True}, {"cancelled": True}])
async def test_explicit_acknowledgement_removes_result(outbox, monkeypatch, data):
    path = remote._worker_store_result_outbox({"job_id": "one"})
    in_progress = set()

    async def submit(*args, **kwargs):
        return {"ok": True, "data": data}

    monkeypatch.setattr(remote, "_submit_worker_result_with_heartbeat", submit)
    monkeypatch.setattr(remote, "audit", lambda *args, **kwargs: None)
    sender = asyncio.create_task(
        remote._worker_result_outbox_sender("https://controller.test", {}, 1, in_progress)
    )
    try:
        async with asyncio.timeout(2):
            while path.exists() or in_progress:
                await asyncio.sleep(0.005)
    finally:
        await _cancel_senders(sender)
    assert not path.exists()


async def test_acknowledged_result_is_retained_when_unlink_fails(outbox, monkeypatch):
    path = remote._worker_store_result_outbox({"job_id": "one"})
    failed = asyncio.Event()
    in_progress = set()
    unlink = remote.Path.unlink

    async def submit(*args, **kwargs):
        return {"data": {"accepted": True}}

    def fail_unlink(target, *args, **kwargs):
        if target == path:
            raise OSError("disk unavailable")
        return unlink(target, *args, **kwargs)

    monkeypatch.setattr(remote, "_submit_worker_result_with_heartbeat", submit)
    monkeypatch.setattr(remote.Path, "unlink", fail_unlink)
    monkeypatch.setattr(remote, "_worker_log_retry", lambda *args: failed.set())
    sender = asyncio.create_task(
        remote._worker_result_outbox_sender("https://controller.test", {}, 1, in_progress)
    )
    try:
        await asyncio.wait_for(failed.wait(), 2)
    finally:
        await _cancel_senders(sender)
    assert path.exists()
    assert not in_progress


async def test_two_failed_uploads_do_not_starve_a_new_result(outbox, monkeypatch):
    old_paths = [
        remote._worker_store_result_outbox({
            "job_id": f"old-{index}", "preserve_across_reset": True,
            "lifecycle": {"result_spooled_at": remote._utc() - 7200},
        })
        for index in range(2)
    ]
    started = asyncio.Event()
    delivered = asyncio.Event()
    attempts = []

    async def submit(result, *args, **kwargs):
        assert kwargs["single_attempt"] is True
        job_id = result["job_id"]
        attempts.append(job_id)
        if job_id.startswith("old"):
            if len(attempts) == 2:
                started.set()
            await started.wait()
            # A permanent HTTP failure must cool down, rather than stop a sender.
            raise remote.WorkerHttpError("https://controller.test/remote/result", 403, "forbidden")
        delivered.set()
        return {"ok": True, "data": {"accepted": True}}

    monkeypatch.setattr(remote, "_submit_worker_result_with_heartbeat", submit)
    monkeypatch.setattr(remote, "_worker_log_retry", lambda *args: None)
    in_progress = set()
    schedule = remote._WorkerResultOutboxSchedule()
    senders = [asyncio.create_task(remote._worker_result_outbox_sender(
        "https://controller.test", {}, 1, in_progress, schedule
    )) for _ in range(2)]
    try:
        await asyncio.wait_for(started.wait(), 2)
        new_path = remote._worker_store_result_outbox({
            "job_id": "new-interactive", "lifecycle": {"result_spooled_at": remote._utc()},
        })
        await asyncio.wait_for(delivered.wait(), 0.5)
        async with asyncio.timeout(2):
            while new_path.exists() or in_progress:
                await asyncio.sleep(0.005)
    finally:
        await _cancel_senders(*senders)
    assert all(path.exists() for path in old_paths)
    assert attempts.count("old-0") == attempts.count("old-1") == 1
    assert attempts[-1] == "new-interactive"
    assert not in_progress


async def test_backlog_progresses_during_a_continuous_stream_of_fresh_results(outbox):
    old_path = remote._worker_store_result_outbox({
        "job_id": "protected", "preserve_across_reset": True,
        "lifecycle": {"result_spooled_at": remote._utc() - 7200},
    })
    schedule = remote._WorkerResultOutboxSchedule()
    in_progress = set()
    chosen = []
    for index in range(remote._WORKER_OUTBOX_FRESH_BURST + 1):
        remote._worker_store_result_outbox({
            "job_id": f"fresh-{index}", "lifecycle": {"result_spooled_at": remote._utc()},
        })
        path = await schedule.claim(outbox, in_progress)
        chosen.append(path)
        in_progress.remove(path.name)
        # A successful fresh upload leaves another fresh result ready next turn.
        if path != old_path:
            path.unlink()
    assert chosen[-1] == old_path
    assert all(path != old_path for path in chosen[:-1])
    assert old_path.exists()


async def test_shared_claim_and_backoff_prevent_duplicate_sender_attempts(outbox, monkeypatch):
    path = remote._worker_store_result_outbox({"job_id": "one"})
    schedule = remote._WorkerResultOutboxSchedule()
    in_progress = set()
    assert await schedule.claim(outbox, in_progress) == path
    assert await schedule.claim(outbox, in_progress) is None
    assert await schedule.failed(path.name) == 1
    in_progress.remove(path.name)
    assert await schedule.claim(outbox, in_progress) is None
    assert await schedule.failed(path.name) == 2
    # Expiring the one shared cooldown makes the file eligible exactly once.
    schedule.retry_at[path.name] = 0
    assert await schedule.claim(outbox, in_progress) == path
    assert await schedule.claim(outbox, in_progress) is None


async def test_old_retry_precedes_newly_discovered_backlog(outbox):
    old_path = remote._worker_store_result_outbox({
        "job_id": "protected", "preserve_across_reset": True,
        "lifecycle": {"result_spooled_at": remote._utc() - 7200},
    })
    schedule = remote._WorkerResultOutboxSchedule()
    in_progress = set()
    assert await schedule.claim(outbox, in_progress) == old_path
    await schedule.failed(old_path.name)
    in_progress.clear()
    newer_path = remote._worker_store_result_outbox({
        "job_id": "newer-backlog", "lifecycle": {"result_spooled_at": remote._utc() - 3600},
    })
    # While the protected result cools down, other entries make progress.
    assert await schedule.claim(outbox, in_progress) == newer_path
    in_progress.clear()
    schedule.retry_at[old_path.name] = 0
    remote._worker_store_result_outbox({
        "job_id": "newest-backlog", "lifecycle": {"result_spooled_at": remote._utc() - 1800},
    })
    # A continuous influx of previously untried backlog cannot starve the retry.
    assert await schedule.claim(outbox, in_progress) == old_path


@pytest.mark.parametrize("bad_json", ["[" * 2000 + "]" * 2000,
    '{"lifecycle":{"result_spooled_at":' + "1" + "0" * 400 + "}}"])
async def test_extreme_metadata_failure_is_isolated_to_one_file(outbox, bad_json):
    corrupt = remote._worker_store_result_outbox({"job_id": "bad"})
    corrupt.write_text(bad_json, encoding="utf-8")
    good = remote._worker_store_result_outbox({
        "job_id": "good", "lifecycle": {"result_spooled_at": remote._utc()},
    })
    schedule = remote._WorkerResultOutboxSchedule()
    assert await schedule.claim(outbox, set()) == good
    assert corrupt.exists()


async def test_spool_time_is_cached_and_mtime_rewrite_does_not_change_priority(
    outbox, monkeypatch
):
    old_path = remote._worker_store_result_outbox({
        "job_id": "old", "lifecycle": {"result_spooled_at": remote._utc() - 7200},
    })
    schedule = remote._WorkerResultOutboxSchedule()
    in_progress = set()
    assert await schedule.claim(outbox, in_progress) == old_path
    in_progress.clear()
    original_time = schedule.spooled_at[old_path.name]
    # A retry's timing update changes mtime, but never immutable queue priority.
    remote._worker_store_result_outbox({
        "job_id": "old", "lifecycle": {"result_spooled_at": remote._utc()},
    })
    remote._worker_store_result_outbox({
        "job_id": "fresh", "lifecycle": {"result_spooled_at": remote._utc()},
    })
    reads = []
    read_text = remote.Path.read_text

    def tracked_read(path, *args, **kwargs):
        reads.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(remote.Path, "read_text", tracked_read)
    assert await schedule.claim(outbox, in_progress) != old_path
    assert schedule.spooled_at[old_path.name] == original_time
    assert old_path not in reads


async def test_corrupt_entry_does_not_prevent_good_result_delivery(outbox, monkeypatch):
    corrupt = remote._worker_store_result_outbox({"job_id": "bad"})
    corrupt.write_text("{", encoding="utf-8")
    good = remote._worker_store_result_outbox({"job_id": "good"})
    submitted = []

    async def submit(result, *args, **kwargs):
        submitted.append(result["job_id"])
        return {"data": {"accepted": True}}

    monkeypatch.setattr(remote, "_submit_worker_result_with_heartbeat", submit)
    monkeypatch.setattr(remote, "_worker_log_retry", lambda *args: None)
    in_progress = set()
    sender = asyncio.create_task(
        remote._worker_result_outbox_sender("https://controller.test", {}, 1, in_progress)
    )
    try:
        async with asyncio.timeout(2):
            while good.exists() or in_progress:
                await asyncio.sleep(0.005)
    finally:
        await _cancel_senders(sender)
    assert corrupt.exists()
    assert submitted == ["good"]


async def test_cancelling_sender_during_upload_retains_result(outbox, monkeypatch):
    path = remote._worker_store_result_outbox({"job_id": "one"})
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def post(*args):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(remote, "_worker_post_json_cancellable", post)
    in_progress = set()
    sender = asyncio.create_task(
        remote._worker_result_outbox_sender("https://controller.test", {}, 1, in_progress)
    )
    await asyncio.wait_for(started.wait(), 2)
    await _cancel_senders(sender)
    assert stopped.is_set()
    assert path.exists()
    assert not in_progress
