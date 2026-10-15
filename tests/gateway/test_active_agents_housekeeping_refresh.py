"""LOCAL PATCH-025: session housekeeping re-publishes the live active-work count.

``active_agents`` is otherwise written only at chat-turn boundaries; cron work that finishes after
the last turn left a stale non-zero value that external drain gates waited on forever.
"""

import asyncio
import time

from gateway.run_watchers import GatewaySessionWatchersMixin


class _Runner(GatewaySessionWatchersMixin):
    def __init__(self, work_count: int):
        self.work_count = work_count
        self.published: list[int] = []
        self._last_session_store_prune_ts = time.time()

    def _active_work_count(self) -> int:
        return self.work_count

    def _persist_active_agents(self) -> None:
        self.published.append(self._active_work_count())

    def _sweep_idle_cached_agents(self) -> int:
        return 0

    def _sweep_agent_cache_under_pressure(self) -> None:
        return None


def test_housekeeping_republishes_count_after_out_of_turn_work_ends():
    runner = _Runner(work_count=1)  # a cron job still running at the last turn boundary
    asyncio.run(runner._session_housekeeping())
    runner.work_count = 0  # cron finished; no chat turn follows
    asyncio.run(runner._session_housekeeping())
    assert runner.published == [1, 0]


def test_housekeeping_continues_when_publish_fails():
    runner = _Runner(work_count=0)
    swept = []

    def _boom():
        raise RuntimeError("status writer unavailable")

    runner._persist_active_agents = _boom
    runner._sweep_idle_cached_agents = lambda: swept.append(True) or 0
    asyncio.run(runner._session_housekeeping())
    assert swept == [True]
