# SPDX-License-Identifier: MIT
"""Verify the step counter (**a wrong count is worse than no count**).

Progress that lies is worse than a heartbeat, so what is pinned here is the
honesty of the report rather than its prettiness:

- The **first and last steps always arrive**, whatever the throttle says.
- A loop with an unknown length reports `step` and **no `total`**, so nothing
  downstream can compute a percentage out of thin air.
- If a loop turns out to be longer than its total said, **the total is dropped**
  rather than reported as 17/15.
- The tqdm replacement **counts even when the bar is disabled**, which is how
  upstream runs it.
- The scheduler hook takes its total from the scheduler itself, and installing
  it twice does not double-count.

Run it with any python 3.12 (no torch, no numpy)::

    <venv>\\Scripts\\python.exe .\\tests\\test_steps.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis import steps  # noqa: E402


class Sink:
    """Collects what the runner would have sent."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def __call__(self, stage: str, message: str = "", **extra: Any) -> None:
        self.events.append({"stage": stage, "message": message, **extra})


def _counter(sink: Sink, min_interval: float = 0.0) -> steps.StepCounter:
    counter = steps.StepCounter(min_interval_sec=min_interval)
    counter.bind(sink, "sampling")
    return counter


def test_every_step_is_reported_when_unthrottled() -> None:
    """With no throttle, each step produces one event carrying step and total."""
    sink = Sink()
    counter = _counter(sink)
    counter.begin(3)
    for _ in range(3):
        counter.advance()
    assert [e["step"] for e in sink.events] == [1, 2, 3], sink.events
    assert all(e["total"] == 3 for e in sink.events), sink.events
    assert all(e["stage"] == "sampling" for e in sink.events)


def test_the_first_and_last_step_beat_the_throttle() -> None:
    """A long throttle still lets the ends through: they prove start and finish."""
    sink = Sink()
    counter = _counter(sink, min_interval=3600.0)
    counter.begin(5)
    for _ in range(5):
        counter.advance()
    assert [e["step"] for e in sink.events] == [1, 5], sink.events


def test_an_unknown_length_reports_no_total() -> None:
    """**No total means no percentage.** The step number still goes out."""
    sink = Sink()
    counter = _counter(sink)
    counter.begin(None)
    counter.advance()
    counter.advance()
    assert [e["step"] for e in sink.events] == [1, 2], sink.events
    assert all("total" not in e for e in sink.events), sink.events


def test_overrunning_the_total_drops_it() -> None:
    """A loop longer than its total stops claiming one instead of reporting 3/2."""
    sink = Sink()
    counter = _counter(sink)
    counter.begin(2)
    for _ in range(3):
        counter.advance()
    assert sink.events[-1]["step"] == 3, sink.events
    assert "total" not in sink.events[-1], sink.events


def test_binding_clears_the_previous_count() -> None:
    """A second request starts at one, not where the last one stopped."""
    sink = Sink()
    counter = _counter(sink)
    counter.begin(2)
    counter.advance()
    counter.bind(sink, "sampling")
    counter.begin(2)
    counter.advance()
    assert sink.events[-1]["step"] == 1, sink.events


def test_nothing_is_sent_without_a_sink() -> None:
    """An unbound counter is silent rather than raising."""
    counter = steps.StepCounter(min_interval_sec=0.0)
    counter.bind(None, "sampling")
    counter.begin(2)
    counter.advance()


# --- the tqdm replacement ----------------------------------------------------


class FakeTqdm:
    """Enough of tqdm to test against: it iterates and honours `disable`."""

    def __init__(self, iterable: Any = None, total: Any = None, disable: bool = False) -> None:
        self.iterable = iterable
        self.total = total if total is not None else len(iterable)
        self.disable = disable
        self.drawn = 0

    def __iter__(self) -> Any:
        for item in self.iterable:
            if not self.disable:
                self.drawn += 1
            yield item


def test_tqdm_counts_even_when_the_bar_is_disabled() -> None:
    """**Upstream disables its bars.** Counting must not depend on drawing."""
    sink = Sink()
    counter = _counter(sink)
    counting = steps.counting_tqdm(counter, FakeTqdm)
    bar = counting([10, 20, 30], disable=True)
    assert list(bar) == [10, 20, 30]
    assert [e["step"] for e in sink.events] == [1, 2, 3], sink.events
    assert all(e["total"] == 3 for e in sink.events), sink.events
    assert bar.drawn == 0, "the bar drew despite disable=True"


def test_patching_a_module_is_idempotent() -> None:
    """Installing twice leaves one layer, not two."""

    class Module:
        tqdm = FakeTqdm

    sink = Sink()
    counter = _counter(sink)
    steps.count_tqdm(Module, counter)
    first = Module.tqdm
    steps.count_tqdm(Module, counter)
    assert Module.tqdm is first, "the replacement was wrapped again"
    assert list(Module.tqdm([1, 2])) == [1, 2]
    assert [e["step"] for e in sink.events] == [1, 2], sink.events


# --- the diffusers scheduler hook --------------------------------------------


class FakeScheduler:
    """Enough of a diffusers scheduler, **including the parameter names**.

    Real pipelines decide what to pass by inspecting these names, so the fake
    carries the ones upstream actually looks for (`sigmas` on `set_timesteps`,
    `eta` on `step`).
    """

    def __init__(self) -> None:
        self.timesteps: list[int] = []
        self.stepped = 0

    def set_timesteps(self, num: int, sigmas: Any = None) -> None:
        self.timesteps = list(range(num))

    def step(self, sample: Any = None, eta: float = 0.0) -> str:
        self.stepped += 1
        return "sample"


def test_the_hook_keeps_the_scheduler_signatures() -> None:
    """**Upstream reads the parameter names and refuses to run without them.**

    `retrieve_timesteps` asks whether `set_timesteps` takes `sigmas`, and the
    denoising loop asks whether `step` takes `eta`. A wrapper that reports only
    `*args, **kwargs` makes upstream raise
    "does not support custom sigmas schedules" (seen on 2026-09-02).
    """
    import inspect

    scheduler = FakeScheduler()
    steps.count_scheduler(scheduler, _counter(Sink()))
    set_params = set(inspect.signature(scheduler.set_timesteps).parameters)
    step_params = set(inspect.signature(scheduler.step).parameters)
    assert "sigmas" in set_params, set_params
    assert "num" in set_params, set_params
    assert "eta" in step_params, step_params


def test_the_scheduler_supplies_the_total() -> None:
    """The count follows what upstream asked the scheduler for."""
    sink = Sink()
    counter = _counter(sink)
    scheduler = FakeScheduler()
    steps.count_scheduler(scheduler, counter)
    scheduler.set_timesteps(4)
    for _ in range(4):
        assert scheduler.step() == "sample", "the original return value was lost"
    assert scheduler.stepped == 4
    assert [e["step"] for e in sink.events] == [1, 2, 3, 4], sink.events
    assert all(e["total"] == 4 for e in sink.events), sink.events


def test_installing_the_scheduler_hook_twice_does_not_double_count() -> None:
    """It is installed before every generation, so it must be idempotent."""
    sink = Sink()
    counter = _counter(sink)
    scheduler = FakeScheduler()
    steps.count_scheduler(scheduler, counter)
    steps.count_scheduler(scheduler, counter)
    scheduler.set_timesteps(2)
    scheduler.step()
    assert [e["step"] for e in sink.events] == [1], sink.events


def test_a_second_run_restarts_the_count() -> None:
    """set_timesteps resets, so the next generation starts from one."""
    sink = Sink()
    counter = _counter(sink)
    scheduler = FakeScheduler()
    steps.count_scheduler(scheduler, counter)
    scheduler.set_timesteps(2)
    scheduler.step()
    scheduler.step()
    scheduler.set_timesteps(2)
    scheduler.step()
    assert sink.events[-1]["step"] == 1, sink.events


# --- our own loops -----------------------------------------------------------


def test_counted_passes_the_items_through_unchanged() -> None:
    """Wrapping a loop must not change what it yields."""
    sink = Sink()
    counter = _counter(sink)
    assert list(steps.counted(["a", "b"], counter)) == ["a", "b"]
    assert [e["step"] for e in sink.events] == [1, 2], sink.events
    assert all(e["total"] == 2 for e in sink.events), sink.events


def test_counted_handles_a_generator() -> None:
    """A generator has no length, so it reports steps without a total."""
    sink = Sink()
    counter = _counter(sink)
    assert list(steps.counted((x for x in [1, 2]), counter)) == [1, 2]
    assert all("total" not in e for e in sink.events), sink.events


def main() -> int:
    """Run every test."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

