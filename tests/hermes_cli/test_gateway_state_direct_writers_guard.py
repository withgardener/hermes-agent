"""Out-of-process ``gateway_state.json`` writers honor the runtime-file production guard.

LOCAL PATCH (PATCH-023): container boot seeding, s6 desired-state persistence and the
migration compensator stamp ``gateway_state.json`` directly instead of going through the
gateway's runtime-status writer, so each must refuse a test-context write to a production root.
"""

import json
from pathlib import Path

import pytest

import hermes_state_guard as guard
from hermes_cli import container_boot, gateway_migrate, service_manager


@pytest.fixture
def launch_root(tmp_path, monkeypatch):
    """Stand-in for the operator's live root (the root this pytest process launched with)."""
    root = tmp_path / "launch-home" / ".hermes"
    root.mkdir(parents=True)
    monkeypatch.setattr(guard, "_LAUNCH_PLATFORM_STATE_ROOT", root.resolve())
    return root


@pytest.fixture
def sandbox_root(tmp_path):
    root = tmp_path / "sandbox-root"
    root.mkdir()
    return root


def _seed(path: Path, record: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path.read_text(encoding="utf-8")


def test_container_boot_legacy_seed_refuses_a_production_root(launch_root, monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_NO_SUPERVISE", raising=False)
    with pytest.raises(guard.RuntimeFileIsolationError):
        container_boot._maybe_migrate_legacy_gateway_run_state(
            launch_root, container_argv=("gateway", "run"), dry_run=False)
    assert not (launch_root / "gateway_state.json").exists()


def test_container_boot_legacy_seed_still_writes_a_sandbox_root(launch_root, sandbox_root, monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_NO_SUPERVISE", raising=False)
    assert container_boot._maybe_migrate_legacy_gateway_run_state(
        sandbox_root, container_argv=("gateway", "run"), dry_run=False) == "running"
    assert json.loads((sandbox_root / "gateway_state.json").read_text())["desired_state"] == "running"


@pytest.mark.parametrize("service, rel", [
    ("gateway-default", Path("gateway_state.json")),
    ("gateway-coder", Path("profiles") / "coder" / "gateway_state.json"),
])
def test_s6_desired_state_refuses_a_production_root(launch_root, monkeypatch, service, rel):
    target = launch_root / rel
    before = _seed(target, {"gateway_state": "running", "pid": 4242})
    monkeypatch.setenv("HERMES_HOME", str(launch_root))
    with pytest.raises(guard.RuntimeFileIsolationError):
        service_manager._write_gateway_desired_state(service, "stopped")
    assert target.read_text(encoding="utf-8") == before
    assert not target.with_suffix(target.suffix + ".tmp").exists()


def test_s6_desired_state_still_writes_a_sandbox_root(launch_root, sandbox_root, monkeypatch):
    target = sandbox_root / "gateway_state.json"
    _seed(target, {"gateway_state": "running"})
    monkeypatch.setenv("HERMES_HOME", str(sandbox_root))
    service_manager._write_gateway_desired_state("gateway-default", "stopped")
    assert json.loads(target.read_text())["desired_state"] == "stopped"


def test_migration_compensator_refuses_a_production_root(launch_root):
    target = launch_root / "gateway_state.json"
    before = _seed(target, {"gateway_state": "running", "pid": 4242, "served_profiles": ["coder"],
                            "platforms": {"coder:feishu": {"state": "connected"}}})
    with pytest.raises(guard.RuntimeFileIsolationError):
        gateway_migrate._reconcile_standalone_runtime(launch_root, {"coder"})
    assert target.read_text(encoding="utf-8") == before


def test_migration_compensator_still_clears_a_sandbox_root(launch_root, sandbox_root):
    target = sandbox_root / "gateway_state.json"
    _seed(target, {"served_profiles": ["coder"], "platforms": {"coder:feishu": {}, "feishu": {}}})
    gateway_migrate._reconcile_standalone_runtime(sandbox_root, {"coder"})
    data = json.loads(target.read_text())
    assert data["served_profiles"] == [] and data["platforms"] == {"feishu": {}}
