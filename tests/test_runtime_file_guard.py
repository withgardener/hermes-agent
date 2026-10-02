"""Shared production-root guard for runtime identity files (``gateway_state.json``).

LOCAL PATCH (PATCH-023): every writer of a runtime identity file resolves its target from
HERMES_HOME / HOME / passwd at CALL time, so a test that wipes or rewrites the env can aim it at
the operator's live ``~/.hermes``. The guard decides "production" from roots a test cannot move:
the root this process launched with, the passwd home, the sudo invoker's home, and the custom
pre-sandbox root the hermetic conftest exports.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

import hermes_constants
import hermes_state_guard as guard


@pytest.fixture
def launch_root(tmp_path, monkeypatch):
    """Stand-in for the root this pytest process launched with (the operator's ~/.hermes)."""
    root = tmp_path / "launch-home" / ".hermes"
    root.mkdir(parents=True)
    monkeypatch.setattr(guard, "_LAUNCH_PLATFORM_STATE_ROOT", root.resolve())
    return root


def _refused(path: Path) -> bool:
    try:
        guard.ensure_runtime_file_test_isolation(path)
    except guard.RuntimeFileIsolationError:
        return True
    return False


def test_root_and_profile_runtime_files_under_launch_root_are_refused(launch_root):
    assert _refused(launch_root / "gateway_state.json")
    assert _refused(launch_root / "profiles" / "coder" / "gateway_state.json")


def test_scratch_paths_deeper_under_the_root_are_not_production(launch_root):
    assert not _refused(launch_root / "cache" / "scratch" / "wt" / "gateway_state.json")


def test_a_test_owned_home_is_not_production(launch_root, tmp_path, monkeypatch):
    """A test that points HOME at a tmp dir must keep writing there (no over-blocking)."""
    test_home = tmp_path / "test-home"
    (test_home / ".hermes").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(test_home))
    assert not _refused(test_home / ".hermes" / "gateway_state.json")


def test_launch_root_stays_protected_after_home_is_rewritten(launch_root, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "elsewhere"))
    assert _refused(launch_root / "gateway_state.json")
    with patch.dict(os.environ, {}, clear=True):
        assert _refused(launch_root / "gateway_state.json")


@pytest.mark.linux_only
def test_passwd_home_is_protected_when_it_differs_from_the_launch_home(
        launch_root, tmp_path, monkeypatch):
    import pwd

    passwd_home = tmp_path / "passwd-home"
    (passwd_home / ".hermes").mkdir(parents=True)
    real_getpwuid = pwd.getpwuid
    monkeypatch.setattr(pwd, "getpwuid", lambda uid: pwd.struct_passwd(
        (*real_getpwuid(uid)[:5], str(passwd_home), *real_getpwuid(uid)[6:])))
    assert _refused(passwd_home / ".hermes" / "gateway_state.json")
    assert _refused(launch_root / "gateway_state.json")


@pytest.mark.linux_only
def test_passwd_lookup_failure_still_protects_the_launch_root(launch_root, monkeypatch):
    import pwd

    def no_entry(uid):
        raise KeyError(uid)

    monkeypatch.setattr(pwd, "getpwuid", no_entry)
    with patch.dict(os.environ, {}, clear=True):
        assert _refused(launch_root / "gateway_state.json")


def test_sudo_invoker_home_is_protected(launch_root, tmp_path, monkeypatch):
    invoker_root = tmp_path / "invoker" / ".hermes"
    invoker_root.mkdir(parents=True)
    monkeypatch.setattr(hermes_constants, "sudo_invoker_default_home", lambda: invoker_root)
    assert _refused(invoker_root / "gateway_state.json")
    assert _refused(invoker_root / "profiles" / "ops" / "gateway_state.json")


def test_extra_deny_roots_cover_a_custom_production_root(launch_root, tmp_path, monkeypatch):
    custom = tmp_path / "opt" / "data"
    custom.mkdir(parents=True)
    monkeypatch.setattr(guard, "_RUNTIME_FILE_GUARD_EXTRA_DENY_ROOTS", (custom,))
    assert _refused(custom / "gateway_state.json")
    assert _refused(custom / "profiles" / "coder" / "gateway_state.json")


def test_bypass_flag_and_bypass_env_allow_the_write(launch_root, monkeypatch):
    monkeypatch.setattr(guard, "_RUNTIME_FILE_GUARD_BYPASS", True)
    assert not _refused(launch_root / "gateway_state.json")
    monkeypatch.setattr(guard, "_RUNTIME_FILE_GUARD_BYPASS", False)
    monkeypatch.setenv(guard._STATE_DB_GUARD_BYPASS_ENV, "1")
    assert not _refused(launch_root / "gateway_state.json")


def test_outside_test_context_the_guard_never_resolves_roots(launch_root, monkeypatch):
    monkeypatch.setattr(guard, "in_runtime_file_test_context", lambda: False)

    def must_not_run():
        raise AssertionError("production path must return before root resolution")

    monkeypatch.setattr(guard, "runtime_file_production_roots", must_not_run)
    guard.ensure_runtime_file_test_isolation(launch_root / "gateway_state.json")


def test_in_process_pytest_counts_as_test_context_without_env(monkeypatch):
    """The incident shape: env wiped, no pytest ancestor, but this process IS pytest."""
    monkeypatch.setattr(guard, "_has_pytest_ancestor", lambda: False)
    with patch.dict(os.environ, {}, clear=True):
        assert guard._in_test_context() is False
        assert guard.in_runtime_file_test_context() is True


@pytest.mark.live_system_guard_bypass
def test_live_system_guard_bypass_marker_disarms_the_runtime_file_guard():
    assert guard._RUNTIME_FILE_GUARD_BYPASS is True


def test_unmarked_tests_run_with_the_runtime_file_guard_armed():
    assert guard._RUNTIME_FILE_GUARD_BYPASS is False


def _run_probe_pytest(tmp_path, probe_body: str, env: dict) -> subprocess.CompletedProcess:
    """Run ``probe_body`` as a test in a FRESH pytest that loads the real tests/conftest.py as a
    plugin. The probe lives under tmp_path, so nothing is ever written into the source tree."""
    repo = Path(__file__).resolve().parents[1]
    probe = tmp_path / "probe" / "test_probe.py"
    probe.parent.mkdir(parents=True)
    probe.write_text(textwrap.dedent(probe_body), encoding="utf-8")
    env = {k: v for k, v in env.items() if not k.startswith("PYTEST_")}
    env["PYTHONPATH"] = str(repo)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "tests.conftest",
         f"--rootdir={probe.parent}", "-c", os.devnull, str(probe)],
        cwd=repo, env=env, capture_output=True, text=True, timeout=180)


def test_conftest_denies_a_custom_pre_sandbox_root_in_a_fresh_pytest(tmp_path):
    """A pytest launched with a custom (non-default) HERMES_HOME treats that root as production
    for runtime files, like the state.db guard does."""
    custom = tmp_path / "custom-root"
    custom.mkdir()
    env = dict(os.environ, HERMES_HOME=str(custom))
    result = _run_probe_pytest(tmp_path, f"""
        from pathlib import Path
        import hermes_state_guard as guard

        def test_custom_root_is_denied():
            assert Path({str(custom)!r}).resolve() in guard.runtime_file_production_roots()
    """, env)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


def test_conftest_maps_a_custom_profile_home_to_its_root_for_both_guards(tmp_path):
    """A custom ``<root>/profiles/<name>`` launch home protects the whole custom root, for the
    runtime-file guard AND the state.db guard (one shared derivation)."""
    root = tmp_path / "opt-data"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    env = dict(os.environ, HERMES_HOME=str(profile))
    result = _run_probe_pytest(tmp_path, f"""
        from pathlib import Path
        import hermes_state
        import hermes_state_guard as guard

        ROOT = Path({str(root)!r}).resolve()

        def test_runtime_file_guard_denies_the_custom_root():
            assert ROOT in guard.runtime_file_production_roots()

        def test_state_db_guard_denies_the_custom_root_and_sibling_profiles():
            assert ROOT in hermes_state._STATE_DB_GUARD_EXTRA_DENY_ROOTS
            for target in (ROOT / "state.db", ROOT / "profiles" / "other" / "state.db"):
                try:
                    hermes_state._ensure_test_isolation(target)
                except RuntimeError:
                    continue
                raise AssertionError(f"state.db guard allowed {{target}}")
    """, env)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


def test_launch_root_is_frozen_at_conftest_import_even_without_hermes_home(tmp_path):
    """``scripts/run_tests.sh`` launches pytest under ``env -i`` (no HERMES_HOME). The launch root
    must still come from the HOME pytest STARTED with, not from whatever HOME the first test that
    happens to import the guard has set."""
    launch_home = tmp_path / "launch-home"
    launch_home.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "HERMES_HOME"}
    env["HOME"] = str(launch_home)
    result = _run_probe_pytest(tmp_path, f"""
        import os
        from pathlib import Path

        # A test MODULE that rewrites HOME at import (collection time, before any fixture).
        os.environ["HOME"] = {str(tmp_path / "module-owned-home")!r}

        def test_launch_root_is_still_the_launch_home():
            import hermes_state_guard as guard
            assert guard._LAUNCH_PLATFORM_STATE_ROOT == (Path({str(launch_home)!r}) / ".hermes").resolve()
            assert guard._LAUNCH_HOST_LOCK_DIR == (
                Path({str(launch_home)!r}) / ".local" / "state" / "hermes" / "gateway-locks").resolve()
    """, env)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


@pytest.mark.macos_only
def test_case_variant_of_the_launch_root_is_refused_on_case_insensitive_volumes(launch_root):
    variant = Path(str(launch_root).replace("launch-home", "LAUNCH-HOME"))
    assert variant != launch_root and variant.exists()
    assert _refused(variant / "gateway_state.json")
    assert _refused(variant / "Profiles" / "coder" / "gateway_state.json")


@pytest.mark.windows_only
def test_case_variant_of_the_launch_root_is_refused_on_windows(launch_root):
    variant = Path(str(launch_root).replace("launch-home", "LAUNCH-HOME"))
    assert _refused(variant / "gateway_state.json")
    assert _refused(variant / "PROFILES" / "coder" / "gateway_state.json")


@pytest.mark.linux_only
def test_a_bind_mounted_alias_of_the_launch_root_is_refused(launch_root, tmp_path):
    """Path.resolve() does not see through bind mounts (nor case folding on macOS/Windows): the
    guard compares directory identity, not spelling."""
    alias = tmp_path / "alias"
    alias.mkdir()
    probe = textwrap.dedent(f"""
        import sys
        from pathlib import Path
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        import hermes_state_guard as guard
        guard._LAUNCH_PLATFORM_STATE_ROOT = Path({str(launch_root.resolve())!r})
        try:
            guard.ensure_runtime_file_test_isolation(Path({str(alias)!r}) / "gateway_state.json")
        except guard.RuntimeFileIsolationError:
            print("REFUSED")
        else:
            print("ALLOWED")
    """)
    script = tmp_path / "probe.py"
    script.write_text(probe, encoding="utf-8")
    cmd = (f"mount --bind {launch_root} {alias} && "
           f"HERMES_TEST_ISOLATION=1 {sys.executable} {script}")
    try:
        result = subprocess.run(["unshare", "-m", "-r", "--propagation", "private", "sh", "-c", cmd],
                                capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        pytest.skip("unshare(1) not installed")
    if result.returncode != 0 and "REFUSED" not in result.stdout and "ALLOWED" not in result.stdout:
        pytest.skip(f"cannot create a private mount namespace here: {result.stderr.strip()[:200]}")
    assert result.stdout.strip() == "REFUSED", result.stdout + result.stderr
