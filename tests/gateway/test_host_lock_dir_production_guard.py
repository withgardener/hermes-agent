"""Machine-wide gateway lock/rendezvous files are never touched from a test process.

LOCAL PATCH (PATCH-023): token-scoped locks (``acquire_scoped_lock``) and the host rendezvous
record/lock/token live in ``$HERMES_GATEWAY_LOCK_DIR`` else ``$XDG_STATE_HOME/hermes/gateway-locks``
else ``~/.local/state/hermes/gateway-locks`` -- outside HERMES_HOME. The hermetic conftest points
``HERMES_GATEWAY_LOCK_DIR`` at a per-test dir, but a test that clears ``os.environ`` falls back to
the operator's live lock dir, where the running gateway's ``host-gateway.json`` and platform-token
locks live.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import hermes_state_guard as guard
from gateway import host_rendezvous, status


@pytest.fixture
def live_lock_dir(tmp_path, monkeypatch):
    """Stand-in for the operator's live lock dir (the default this pytest launched with)."""
    lock_dir = tmp_path / "launch-home" / ".local" / "state" / "hermes" / "gateway-locks"
    lock_dir.mkdir(parents=True)
    monkeypatch.setattr(guard, "_LAUNCH_HOST_LOCK_DIR", lock_dir.resolve())
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(lock_dir))
    return lock_dir


def test_scoped_lock_is_not_claimed_in_the_live_lock_dir(live_lock_dir):
    with pytest.raises(guard.RuntimeFileIsolationError):
        status.acquire_scoped_lock("feishu-app-id", "cli_test")
    assert list(live_lock_dir.iterdir()) == []


def test_a_live_scoped_lock_is_not_released_or_swept(live_lock_dir):
    lock = status._get_scope_lock_path("feishu-app-id", "cli_test")
    lock.write_text(json.dumps({"pid": os.getpid()}))
    with pytest.raises(guard.RuntimeFileIsolationError):
        status.release_scoped_lock("feishu-app-id", "cli_test")
    with pytest.raises(guard.RuntimeFileIsolationError):
        status.release_all_scoped_locks()
    assert lock.exists()


def test_host_record_is_not_published_into_the_live_lock_dir(live_lock_dir):
    with pytest.raises(guard.RuntimeFileIsolationError):
        host_rendezvous.publish_record("gateway", host="127.0.0.1", port=1, token="t" * 32)
    assert list(live_lock_dir.iterdir()) == []


def test_host_lock_is_not_claimed_in_the_live_lock_dir(live_lock_dir):
    with pytest.raises(guard.RuntimeFileIsolationError):
        host_rendezvous.claim_host_lock("gateway")
    assert not host_rendezvous.owns_host_lock("gateway")
    assert list(live_lock_dir.iterdir()) == []


@pytest.mark.parametrize("remove", [host_rendezvous.clear_record, host_rendezvous.discard_dead_record])
def test_live_host_record_is_not_removed(live_lock_dir, remove):
    record = live_lock_dir / "host-gateway.json"
    record.write_text(json.dumps({"role": "gateway", "pid": os.getpid()}))
    with pytest.raises(guard.RuntimeFileIsolationError):
        remove("gateway")
    assert record.exists()


def test_env_wiped_default_lock_dir_resolves_to_the_protected_launch_dir(live_lock_dir, monkeypatch):
    """The incident shape: no HERMES_GATEWAY_LOCK_DIR / XDG_STATE_HOME -> HOME-derived default."""
    home = live_lock_dir.parents[3]
    with patch.dict(os.environ, {"HOME": str(home)}, clear=True):
        with pytest.raises(guard.RuntimeFileIsolationError):
            status.acquire_scoped_lock("feishu-app-id", "cli_test")
    assert list(live_lock_dir.iterdir()) == []


def test_a_test_owned_lock_dir_still_works(live_lock_dir, tmp_path, monkeypatch):
    own = tmp_path / "per-test-locks"
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(own))
    acquired, _ = status.acquire_scoped_lock("feishu-app-id", "cli_test")
    assert acquired
    status.release_scoped_lock("feishu-app-id", "cli_test")
    assert host_rendezvous.publish_record("gateway", host="127.0.0.1", port=1, token="t" * 32)
    host_rendezvous.clear_record("gateway")
    assert not (own / "host-gateway.json").exists()


def test_lock_dir_roots_include_the_passwd_home_default(tmp_path, monkeypatch):
    import pwd

    passwd_home = tmp_path / "passwd-home"
    real = pwd.getpwuid
    monkeypatch.setattr(pwd, "getpwuid", lambda uid: pwd.struct_passwd(
        (*real(uid)[:5], str(passwd_home), *real(uid)[6:])))
    expected = (passwd_home / ".local" / "state" / "hermes" / "gateway-locks").resolve()
    assert expected in guard.host_lock_dir_production_roots()


def test_launch_lock_dir_honors_an_absolute_xdg_state_home_only():
    assert guard._default_host_lock_dir({"XDG_STATE_HOME": "/x/state", "HOME": "/h"}) == \
        Path("/x/state/hermes/gateway-locks")
    assert guard._default_host_lock_dir({"XDG_STATE_HOME": "rel", "HOME": "/h"}) == \
        Path("/h/.local/state/hermes/gateway-locks")


def test_an_explicit_lock_dir_override_is_never_the_launch_dir(tmp_path, monkeypatch):
    """Children spawned by tests inherit HERMES_TEST_ISOLATION *and* the per-test
    HERMES_GATEWAY_LOCK_DIR; treating that override as "the live dir" made every such child
    refuse its own sandbox. Only the XDG/HOME default is the operator's live dir."""
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "per-test-locks"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    assert guard._launch_host_lock_dir() == (
        tmp_path / "home" / ".local" / "state" / "hermes" / "gateway-locks").resolve()


# ── Host update→restart obligation record (PATCH-023 round 5) ───────────────────────────────

def test_host_obligation_is_not_written_amended_or_cleared_in_the_live_lock_dir(live_lock_dir):
    from hermes_cli import update_host_obligation as ob

    assert ob.write_host_obligation(expected_sha="abc") is False
    assert list(live_lock_dir.iterdir()) == []
    record = live_lock_dir / ob.HOST_OBLIGATION_NAME
    record.write_text(json.dumps({"version": 1, "expected_sha": "live"}))
    before = record.read_text()
    ob.amend_host_obligation(note="x")
    ob.mark_host_restart_completed("abc")
    ob.clear_host_obligation()
    assert record.read_text() == before


def test_host_obligation_still_round_trips_in_a_test_owned_lock_dir(live_lock_dir, tmp_path, monkeypatch):
    from hermes_cli import update_host_obligation as ob

    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "per-test-locks"))
    assert ob.write_host_obligation(expected_sha="abc") is True
    ob.mark_host_restart_completed("abc")
    assert ob.host_restart_already_completed("abc")
    ob.clear_host_obligation()
    assert not ob.host_obligation_present()
