"""Runtime-status writes are refused when a test process resolves the production home.

LOCAL PATCH (PATCH-023): tests that clear ``os.environ`` (e.g. ``patch.dict(os.environ,
{}, clear=True)``) lose the conftest's sandboxed HERMES_HOME, so a status write from
code under test resolves the operator's real ``~/.hermes/gateway_state.json`` and
stamps the pytest pid over the live gateway's record. Mirrors the state.db guard
(``hermes_state._ensure_test_isolation``) at the single runtime-status writer; the guard itself
lives in ``hermes_state_guard`` (see tests/test_runtime_file_guard.py).
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway import status


@pytest.fixture
def fake_real_root(tmp_path, monkeypatch):
    """Point the guard's notion of the real platform root at a scratch dir."""
    root = tmp_path / "real-home" / ".hermes"
    root.mkdir(parents=True)
    import hermes_state_guard

    monkeypatch.setattr(hermes_state_guard, "_LAUNCH_PLATFORM_STATE_ROOT", root.resolve())
    return root


def _drain():
    """Wait for the background writer; a refused write settles as not-persisted."""
    status.flush_runtime_status(timeout=5.0)


def test_env_wiped_status_write_never_reaches_production_file(fake_real_root, monkeypatch):
    live = fake_real_root / "gateway_state.json"
    live.write_text(json.dumps({"gateway_state": "running", "pid": 4242}))
    before = live.read_text()
    with patch.dict(os.environ, {"HERMES_HOME": str(fake_real_root)}, clear=True):
        status.publish_runtime_status(platform="feishu", platform_state="connected")
        _drain()
    assert live.read_text() == before


def test_sync_status_write_to_production_reports_not_persisted(fake_real_root, monkeypatch):
    live = fake_real_root / "gateway_state.json"
    monkeypatch.setenv("HERMES_HOME", str(fake_real_root))
    assert status.write_runtime_status(gateway_state="starting", wait_timeout=2.0) is False
    assert not live.exists()


def test_profile_runtime_status_under_production_root_is_refused(fake_real_root, monkeypatch):
    profile = fake_real_root / "profiles" / "coder"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    status.publish_runtime_status(gateway_state="starting")
    _drain()
    assert not (profile / "gateway_state.json").exists()


def test_sandboxed_home_still_persists(fake_real_root, tmp_path, monkeypatch):
    sandbox = tmp_path / "sandbox-home"
    sandbox.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(sandbox))
    assert status.write_runtime_status(gateway_state="running", wait_timeout=2.0) is True
    assert json.loads((sandbox / "gateway_state.json").read_text())["gateway_state"] == "running"


def test_guard_inactive_outside_test_context(fake_real_root, monkeypatch):
    """Production processes (no pytest env, ancestor, or loaded pytest) write normally."""
    import hermes_state_guard

    monkeypatch.setattr(hermes_state_guard, "in_runtime_file_test_context", lambda: False)
    monkeypatch.setenv("HERMES_HOME", str(fake_real_root))
    assert status.write_runtime_status(gateway_state="running", wait_timeout=2.0) is True
    assert json.loads((fake_real_root / "gateway_state.json").read_text())["gateway_state"] == "running"


def test_explicit_bypass_env_allows_write(fake_real_root, monkeypatch):
    import hermes_state_guard

    monkeypatch.setenv(hermes_state_guard._STATE_DB_GUARD_BYPASS_ENV, "1")
    monkeypatch.setenv("HERMES_HOME", str(fake_real_root))
    assert status.write_runtime_status(gateway_state="running", wait_timeout=2.0) is True


def test_env_wiped_write_refused_when_home_differs_from_passwd_home(tmp_path, monkeypatch):
    """The path is resolved on the caller thread while the env is wiped (no HOME, so the
    passwd home wins), but the guard runs later on the writer thread after the env is
    restored to a HOME that differs from the passwd home. The passwd home's production
    file must still be protected."""
    import pwd
    import threading

    passwd_home = tmp_path / "passwd-home"
    (passwd_home / ".hermes").mkdir(parents=True)
    other_home = tmp_path / "other-home"
    other_home.mkdir()
    real_getpwuid = pwd.getpwuid

    def fake_getpwuid(uid):
        entry = real_getpwuid(uid)
        return pwd.struct_passwd((*entry[:5], str(passwd_home), *entry[6:]))

    monkeypatch.setattr(pwd, "getpwuid", fake_getpwuid)
    monkeypatch.setenv("HOME", str(other_home))
    env_restored = threading.Event()
    real_merge = status._merge_over_on_disk

    def merge_after_env_restore(path, payload):
        assert env_restored.wait(timeout=5.0)
        return real_merge(path, payload)

    monkeypatch.setattr(status, "_merge_over_on_disk", merge_after_env_restore)
    with patch.dict(os.environ, {}, clear=True):
        status.publish_runtime_status(gateway_state="starting")
    env_restored.set()
    _drain()
    assert not (passwd_home / ".hermes" / "gateway_state.json").exists()


# ── Sibling runtime identity files in the same HERMES_HOME (PATCH-023 round 3) ────────────────

@pytest.fixture
def env_wiped_to_real_root(fake_real_root, monkeypatch):
    """The incident shape for every identity file: the code under test resolves the real root."""
    monkeypatch.setenv("HERMES_HOME", str(fake_real_root))
    return fake_real_root


def test_planned_stop_marker_is_not_written_into_production(env_wiped_to_real_root):
    import hermes_state_guard

    with pytest.raises(hermes_state_guard.RuntimeFileIsolationError):
        status.write_planned_stop_marker(4242)
    assert not (env_wiped_to_real_root / ".gateway-planned-stop.json").exists()


def test_takeover_marker_is_not_written_into_production(env_wiped_to_real_root):
    import hermes_state_guard

    with pytest.raises(hermes_state_guard.RuntimeFileIsolationError):
        status.write_takeover_marker(4242, target_start_time=None)
    assert not (env_wiped_to_real_root / ".gateway-takeover.json").exists()


def test_pid_file_is_not_written_into_production(env_wiped_to_real_root):
    import hermes_state_guard

    with pytest.raises(hermes_state_guard.RuntimeFileIsolationError):
        status.write_pid_file()
    assert not (env_wiped_to_real_root / "gateway.pid").exists()


def test_runtime_lock_is_not_claimed_or_rewritten_in_production(env_wiped_to_real_root):
    import hermes_state_guard

    lock = env_wiped_to_real_root / "gateway.lock"
    lock.write_text(json.dumps({"pid": 4242}))
    before = lock.read_text()
    with pytest.raises(hermes_state_guard.RuntimeFileIsolationError):
        status.acquire_gateway_runtime_lock()
    assert lock.read_text() == before
    assert not status.owns_gateway_runtime_lock()


@pytest.mark.parametrize("clear, name", [
    (lambda: status.clear_takeover_marker(), ".gateway-takeover.json"),
    (lambda: status.clear_planned_stop_marker(), ".gateway-planned-stop.json"),
])
def test_production_markers_are_not_deleted(env_wiped_to_real_root, clear, name):
    import hermes_state_guard

    marker = env_wiped_to_real_root / name
    marker.write_text("{}")
    with pytest.raises(hermes_state_guard.RuntimeFileIsolationError):
        clear()
    assert marker.exists()


def test_sibling_identity_files_still_work_in_a_sandbox_home(fake_real_root, tmp_path, monkeypatch):
    sandbox = tmp_path / "sandbox-home"
    sandbox.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(sandbox))
    assert status.write_planned_stop_marker(4242) is True
    status.clear_planned_stop_marker()
    assert not (sandbox / ".gateway-planned-stop.json").exists()
    assert status.acquire_gateway_runtime_lock() is True
    try:
        status.write_pid_file()
        assert json.loads((sandbox / "gateway.pid").read_text())["pid"] == os.getpid()
    finally:
        status.remove_pid_file()
        status.release_gateway_runtime_lock()


def test_remove_pid_file_leaves_a_production_pid_file_alone(env_wiped_to_real_root):
    """``remove_pid_file`` runs from atexit and swallows errors; it must still never unlink the
    live gateway's PID file, even one that (by PID reuse) names this test process."""
    pid_file = env_wiped_to_real_root / "gateway.pid"
    pid_file.write_text(json.dumps({"pid": os.getpid()}))
    status.remove_pid_file()
    assert pid_file.exists()


# ── Housekeeping deleters on read paths (PATCH-023 round 5) ─────────────────────────────────
# These run inside read APIs (liveness probes, marker consume), so a refusal must leave the file
# in place WITHOUT raising into the caller.

def _stale_marker():
    return json.dumps({"target_pid": 1, "written_at": "2000-01-01T00:00:00+00:00"})


def test_get_running_pid_cleanup_leaves_production_identity_files(env_wiped_to_real_root, monkeypatch):
    pid_file = env_wiped_to_real_root / "gateway.pid"
    lock = env_wiped_to_real_root / "gateway.lock"
    pid_file.write_text(json.dumps({"pid": 999999, "hermes_home": str(env_wiped_to_real_root)}))
    lock.write_text(json.dumps({"pid": 999999}))
    monkeypatch.setattr(status, "_pid_exists", lambda pid: False)
    assert status.get_running_pid() is None
    assert pid_file.exists() and lock.exists()


@pytest.mark.parametrize("consume, name", [
    (lambda: status.consume_planned_stop_marker_for_self(), ".gateway-planned-stop.json"),
    (lambda: status.consume_takeover_marker_for_self(), ".gateway-takeover.json"),
])
def test_consuming_a_production_marker_leaves_it_in_place(env_wiped_to_real_root, consume, name):
    marker = env_wiped_to_real_root / name
    marker.write_text(_stale_marker())
    assert consume() is False
    assert marker.exists()


def test_consuming_a_fresh_foreign_production_marker_leaves_it_in_place(env_wiped_to_real_root):
    """The live gateway's own planned-stop marker: deleting it makes an operator-planned stop look
    like an unexpected exit that the service manager revives."""
    from datetime import datetime, timezone

    marker = env_wiped_to_real_root / ".gateway-planned-stop.json"
    marker.write_text(json.dumps({
        "target_pid": os.getpid() + 1, "written_at": datetime.now(timezone.utc).isoformat()}))
    assert status.consume_planned_stop_marker_for_self() is False
    assert marker.exists()


def test_housekeeping_unlink_still_cleans_a_sandbox_home(fake_real_root, tmp_path, monkeypatch):
    sandbox = tmp_path / "sandbox-home"
    sandbox.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(sandbox))
    marker = sandbox / ".gateway-planned-stop.json"
    marker.write_text(_stale_marker())
    assert status.consume_planned_stop_marker_for_self() is False
    assert not marker.exists()
    (sandbox / "gateway.pid").write_text(json.dumps({"pid": 999999}))
    (sandbox / "gateway.lock").write_text(json.dumps({"pid": 999999}))
    monkeypatch.setattr(status, "_pid_exists", lambda pid: False)
    assert status.get_running_pid() is None
    assert not (sandbox / "gateway.pid").exists() and not (sandbox / "gateway.lock").exists()
