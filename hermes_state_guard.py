"""Live-DB test-isolation guard and the per-process "last init error" record.
Every SessionDB construction resolves its path through _ensure_test_isolation
so a pytest-context process (env OR ancestry) can never open a production
state.db; env-based so subprocess children are protected too."""

import os
import sys
import threading
import weakref
from pathlib import Path
from typing import Any, Optional

try:  # Hard dependency, but tolerate scaffold-phase imports before pip install.
    import psutil
except ImportError:  # pragma: no cover - stripped/scaffold installs only
    psutil = None  # type: ignore[assignment]

# Field evidence: pytest fixture rows landed in the production state.db and a
# pytest-spawned child flipped the journal mode under the live WAL writer.

#: Env twin of ``_STATE_DB_GUARD_BYPASS`` for child processes (a module global
#: cannot cross a process boundary, and ancestry arms the guard there).
_STATE_DB_GUARD_BYPASS_ENV = "HERMES_STATE_DB_GUARD_BYPASS"


def _real_platform_state_root() -> Optional[Path]:
    """The REAL platform-default Hermes root. Avoids ``Path.home()`` /
    ``hermes_constants`` (tests monkeypatch Path.home to a tempdir); ``expanduser``
    reads HOME/passwd, which the conftest never rewrites."""
    try:
        home = Path(os.path.expanduser("~"))
        if sys.platform == "win32":
            base = os.environ.get("LOCALAPPDATA", "").strip()
            root = Path(base) / "hermes" if base else home / "AppData" / "Local" / "hermes"
        else:
            root = home / ".hermes"
        return root.resolve()
    except Exception:
        return None


#: Exported by the hermetic conftest alongside the HERMES_HOME redirect. Unlike
#: PYTEST_* it is OURS and inherits by default, so a child carrying it that
#: resolves a production DB is by definition an isolation escape.
# : Env marker exported by the hermetic test conftest at the same moment it : redirects ``HERMES_HOME`` to
# the per-session tmp isolation root. Unlike ``PYTEST_*`` (owned by pytest, and : routinely scrubbed by
# tests that rebuild a child environment), this marker : is OURS: it declares "this process tree is running
# under Hermes test : isolation", and it inherits into subprocess children by default — so a : child that
# received the patched ``HERMES_HOME`` also received the marker, : and a child that resolves a production DB
# while carrying it is, by : definition, an isolation escape (#82770).
_TEST_ISOLATION_MARKER_ENV = "HERMES_TEST_ISOLATION"


def _running_under_pytest() -> bool:
    """True when this process (or a parent test process) is a pytest run."""
    return bool(
        os.environ.get("PYTEST_CURRENT_TEST")
        or os.environ.get("PYTEST_VERSION")
        or os.environ.get(_TEST_ISOLATION_MARKER_ENV)
    )


#: pytest launcher names, matched against each argv token's *basename* so
#: ``/tmp/pytest-of-dev/...`` paths cannot false-positive.
_PYTEST_LAUNCHER_NAMES = frozenset({"pytest", "py.test", "pytest.exe", "py.test.exe"})

#: Memoised ancestry answer: the tree above us doesn't change; keep the hot path free.
_PYTEST_ANCESTOR: Optional[bool] = None


def _process_looks_like_pytest(proc: Any) -> bool:
    """True when *proc*'s command line is a pytest invocation. Unreadable cmdline
    => not pytest: guessing the other way would refuse production opens."""
    try:
        cmdline = proc.cmdline() or []
    except Exception:
        return False
    for arg in cmdline:
        try:
            # Split on both separators on every host so the answer is platform-independent.
            name = str(arg).strip('"').strip("'").replace("\\", "/").rsplit("/", 1)[-1].lower()
        except Exception:
            continue
        if name in _PYTEST_LAUNCHER_NAMES:
            return True
    return False


def _has_pytest_ancestor() -> bool:
    """True when an ancestor process is a pytest run: a child spawned with a
    rebuilt env loses PYTEST_* and the HERMES_HOME redirect together, ancestry
    survives that. Fails open without psutil / on walk errors.

    ``_running_under_pytest`` reads ``PYTEST_*`` env vars, which a child spawned with a rebuilt environment
    loses at the same moment it loses the ``HERMES_HOME`` redirect: that child aims at the production DB
    *and* disarms the guard in one step (#82770). Ancestry is the one test-context signal that survives an
    env rebuild, so it backs the env check up.
    """
    global _PYTEST_ANCESTOR
    if _PYTEST_ANCESTOR is not None:
        return _PYTEST_ANCESTOR
    found = False
    if psutil is not None:
        try:
            found = any(_process_looks_like_pytest(p) for p in psutil.Process().parents())
        except Exception:
            found = False
    _PYTEST_ANCESTOR = found
    return found


def _in_test_context() -> bool:
    """Test run by environment or ancestry (memoised; env checked first)."""
    return _running_under_pytest() or _has_pytest_ancestor()


def _is_production_state_db(resolved: Path, root: Path) -> bool:
    """*resolved* is ``<root>/state.db`` or ``<root>/profiles/<name>/state.db``;
    deeper scratch paths (repo worktrees) are deliberately NOT matched."""
    if resolved.parent == root:
        return True
    try:
        parts = resolved.relative_to(root).parts
    except ValueError:
        return False
    return len(parts) == 3 and parts[0] == "profiles"


# ── Runtime identity files (``gateway_state.json``) ─────────────────────────────────────────
# LOCAL PATCH (PATCH-023). Tests that wipe/rewrite ``os.environ`` re-aim every call-time path
# resolver (HERMES_HOME → HOME → passwd) at the operator's live root, and a pytest pid then lands
# in the live gateway's status record. "Production" is therefore judged by roots a test cannot
# move: the root at import time (tests/conftest.py imports this module first thing, before any
# test body can rewrite HOME), the passwd home, the sudo invoker's home, and the conftest-exported
# custom pre-sandbox root. The CURRENT HOME is deliberately not a root: a test that points HOME
# at a tmp dir owns that tree.

#: Platform-default root this process launched with (captured once, before any test body runs).
_LAUNCH_PLATFORM_STATE_ROOT: Optional[Path] = _real_platform_state_root()
#: Conftest-managed knobs, mirroring ``hermes_state``'s state.db guard knobs.
_RUNTIME_FILE_GUARD_BYPASS = False
_RUNTIME_FILE_GUARD_EXTRA_DENY_ROOTS: tuple = ()


class RuntimeFileIsolationError(RuntimeError):
    """A test-context process tried to write a runtime identity file under a production root."""


def in_runtime_file_test_context() -> bool:
    """Env/ancestry test context, OR this very process being pytest: a test that wipes
    ``os.environ`` drops every PYTEST_* signal and ancestry only walks parents."""
    return _in_test_context() or "_pytest" in sys.modules


def custom_pre_sandbox_root(pre_sandbox_home: str) -> Optional[Path]:
    """The custom production root a pytest was launched against, or None when the launch
    HERMES_HOME is unset or the platform default (already a production root). A custom
    ``<root>/profiles/<name>`` home yields ``<root>`` so its sibling profiles are covered too.
    Used by the hermetic conftest for both the state.db and the runtime-file deny lists."""
    if not pre_sandbox_home.strip():
        return None
    try:
        home = Path(pre_sandbox_home).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    root = home.parent.parent if home.parent.name == "profiles" else home
    launch = _LAUNCH_PLATFORM_STATE_ROOT
    if launch is not None and _same_directory(root, launch):
        return None
    return root


def runtime_file_production_roots() -> set:
    roots = set()
    candidates = [_LAUNCH_PLATFORM_STATE_ROOT, *_RUNTIME_FILE_GUARD_EXTRA_DENY_ROOTS]
    if sys.platform != "win32":
        try:
            import pwd

            candidates.append(Path(pwd.getpwuid(os.getuid()).pw_dir) / ".hermes")
        except (ImportError, KeyError, OSError):
            pass  # no passwd entry (containers with arbitrary uids): the launch root still holds
    try:
        import hermes_constants

        candidates.append(hermes_constants.sudo_invoker_default_home())
    except ImportError:
        pass
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            roots.add(Path(candidate).expanduser().resolve())
        except (OSError, RuntimeError, ValueError):
            continue
    return roots


def _same_directory(left: Path, right: Path) -> bool:
    """Directory identity, not spelling: ``resolve()`` sees neither bind mounts nor case folding
    (macOS/Windows), so an aliased spelling of a production root must still match."""
    if left == right:
        return True
    try:
        return os.path.samefile(left, right)
    except OSError:  # either side missing: fall back to the platform's case rules
        return os.path.normcase(str(left)) == os.path.normcase(str(right))


def _runtime_file_is_under_root(resolved: Path, root: Path) -> bool:
    """``<root>/<file>`` or ``<root>/profiles/<name>/<file>``; deeper scratch paths do not match."""
    home = resolved.parent
    if _same_directory(home, root):
        return True
    return (os.path.normcase(home.parent.name) == os.path.normcase("profiles")
            and _same_directory(home.parent.parent, root))


# Machine-wide gateway lock dir (token-scoped locks + host rendezvous record/lock/token). It lives
# outside HERMES_HOME (``$HERMES_GATEWAY_LOCK_DIR`` / ``$XDG_STATE_HOME`` / ``~/.local/state``), so an
# env-wiping test falls back to the operator's live dir. Same frozen-root rule as above.
_HOST_LOCK_DIR_PARTS = ("hermes", "gateway-locks")


def _default_host_lock_dir(env) -> Path:
    """``gateway.status._get_lock_dir`` minus the override: XDG (absolute only) else HOME."""
    xdg = env.get("XDG_STATE_HOME") or ""
    base = Path(xdg) if os.path.isabs(xdg) else Path(env.get("HOME") or os.path.expanduser("~")) / ".local" / "state"
    return base.joinpath(*_HOST_LOCK_DIR_PARTS)


def _launch_host_lock_dir() -> Optional[Path]:
    """The DEFAULT lock dir at launch. An explicit ``HERMES_GATEWAY_LOCK_DIR`` is deliberately
    ignored: under pytest it is the conftest's per-test sandbox, inherited by spawned children."""
    try:
        return _default_host_lock_dir(os.environ).resolve()
    except Exception:
        return None


#: Lock dir this process launched with (captured with the launch root, before any test body).
_LAUNCH_HOST_LOCK_DIR: Optional[Path] = _launch_host_lock_dir()


def host_lock_dir_production_roots() -> set:
    dirs = [_LAUNCH_HOST_LOCK_DIR]
    if sys.platform != "win32":
        try:
            import pwd

            dirs.append(Path(pwd.getpwuid(os.getuid()).pw_dir).joinpath(".local", "state", *_HOST_LOCK_DIR_PARTS))
        except (ImportError, KeyError, OSError):
            pass
    try:
        import hermes_constants

        invoker = hermes_constants.sudo_invoker_default_home()
        if invoker is not None:
            dirs.append(invoker.parent.joinpath(".local", "state", *_HOST_LOCK_DIR_PARTS))
    except ImportError:
        pass
    roots = set()
    for candidate in dirs:
        if candidate is None:
            continue
        try:
            roots.add(Path(candidate).expanduser().resolve())
        except (OSError, RuntimeError, ValueError):
            continue
    return roots


def ensure_host_lock_file_test_isolation(path: Path) -> None:
    """Raise when a test-context process is about to create, rewrite or delete a file directly
    inside a production gateway lock dir. Production returns immediately."""
    if (_RUNTIME_FILE_GUARD_BYPASS or os.environ.get(_STATE_DB_GUARD_BYPASS_ENV)
            or not in_runtime_file_test_context()):
        return
    resolved = Path(path).expanduser().resolve()
    for root in host_lock_dir_production_roots():
        if _same_directory(resolved.parent, root) or _same_directory(resolved, root):
            raise RuntimeFileIsolationError(
                f"live-system guard: test attempted to touch production gateway lock file {resolved} "
                f"(live lock dir {root}). Point HERMES_GATEWAY_LOCK_DIR at a temporary dir, mark the "
                "test @pytest.mark.live_system_guard_bypass, or export "
                f"{_STATE_DB_GUARD_BYPASS_ENV}=1 in a spawned child.")


def ensure_runtime_file_test_isolation(path: Path) -> None:
    """Raise when a test-context process is about to write ``path`` under a production root
    (``<root>/<file>`` or ``<root>/profiles/<name>/<file>``). Production returns immediately."""
    if (_RUNTIME_FILE_GUARD_BYPASS or os.environ.get(_STATE_DB_GUARD_BYPASS_ENV)
            or not in_runtime_file_test_context()):
        return
    resolved = Path(path).expanduser().resolve()
    for root in runtime_file_production_roots():
        if _runtime_file_is_under_root(resolved, root):
            raise RuntimeFileIsolationError(
                f"live-system guard: test attempted to write production runtime file {resolved} "
                f"(under real Hermes root {root}). Use a temporary HERMES_HOME, mark the test "
                f"@pytest.mark.live_system_guard_bypass, or export {_STATE_DB_GUARD_BYPASS_ENV}=1 "
                "in a spawned child (a child launched with HOME=<tmp> treats <tmp>/.hermes as its "
                "launch root, so it needs that export too).")


# Test-only SessionDB instance registry. Under the hermetic suite every
# successfully constructed SessionDB is added to this WeakSet so the autouse
# teardown in tests/conftest.py (_close_leaked_session_dbs) can close whatever
# a test forgot to close. Dozens of tests build SessionDB() directly and never
# close it; each instance holds a writer connection plus pooled readers, and a
# single-process run over tests/hermes_cli/ accumulated 16-25 GB RSS (OOM
# incident 20260816). The per-file runner masks this in CI; the registry fixes
# the class at the source instead of patching ~40 test files.
#
# Population is gated on the isolation marker (exported by tests/conftest.py
# before any test module imports). Production processes never populate it;
# do not "simplify" the gate away. WeakSet membership never pins an instance.
_test_instance_registry: "weakref.WeakSet[Any]" = weakref.WeakSet()


def _register_test_instance(db: Any) -> None:
    """Track *db* for suite-level teardown closing (test-isolation runs only)."""
    if os.environ.get(_TEST_ISOLATION_MARKER_ENV):
        try:
            _test_instance_registry.add(db)
        except Exception:  # pragma: no cover — registry must never break init
            pass


# Last SessionDB() init error, per-process; surfaced by /resume-style slash
# commands so users know WHY. Only SessionDB.__init__ writes it.
_last_init_error: Optional[str] = None
_last_init_error_lock = threading.Lock()


def _set_last_init_error(msg: Optional[str]) -> None:
    """Record (or clear with None) the most recent init failure. __init__ never
    clears on success: a concurrent open would erase the cause another thread's
    /resume is about to format."""
    global _last_init_error
    with _last_init_error_lock:
        _last_init_error = msg


def get_last_init_error() -> Optional[str]:
    """Most recent state.db init failure (None if none/never attempted)."""
    return _last_init_error
