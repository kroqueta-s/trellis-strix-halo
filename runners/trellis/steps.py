# SPDX-License-Identifier: MIT
"""Report **how far through a loop** the runner is, without touching upstream code.

The heartbeat says the runner is alive. It does not say how far along it is, and
on this machine that gap is expensive: the first texture bake ran 2,509 s with
nothing but heartbeats, and there was no way to tell progress from a hang.

What is reported here is **counted, never estimated**. A loop that can be
counted sends `step` and `total`; a loop whose length is unknown sends `step`
alone; anything else sends neither. **No ETA and no overall percentage**: on
this hardware the same loop ran 167 s per step on the first run and 14.7 s per
step afterwards, so any prediction built on a stored constant would be wrong by
an order of magnitude exactly when it mattered.

Two hooks cover every loop that matters, both installed at launch time:

- `count_scheduler` for diffusers pipelines. `set_timesteps` fixes the total and
  resets the count; each `scheduler.step` advances it. **This works even when
  upstream disables its progress bar**, which Hunyuan3D's texture stage does
  (`set_progress_bar_config(disable=True)` in `multiview_utils.py`).
- `count_tqdm` for loops that are written by hand around `tqdm`, which is how
  TRELLIS and Hi3DGen sample. The replacement **counts even when the bar is
  disabled**, because `disable=True` only suppresses drawing.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any

# How often a running loop may report. Steps here take seconds (3 s for the
# shape stage, 15 s for the texture stage), so this throttle almost never bites;
# it exists so that a fast loop cannot flood the protocol channel.
_MIN_INTERVAL_SEC = 1.0

# The progress sink a runner passes in: `progress(stage, message, step=, total=)`.
Progress = Callable[..., None]


class StepCounter:
    """Count one loop at a time and report each step.

    One instance belongs to one pipeline and is **rebound per request**, because
    the sink carries the request id. Installing a hook is separate from binding,
    so hooks are installed once and reused.

    **The first and last steps are always reported**, whatever the throttle says:
    the first proves the loop started, the last proves it finished.
    """

    def __init__(self, min_interval_sec: float = _MIN_INTERVAL_SEC) -> None:
        self._min_interval = min_interval_sec
        self._progress: Progress | None = None
        self._stage = ""
        self._message = ""
        self._step = 0
        self._total: int | None = None
        self._last_emit = 0.0

    def bind(self, progress: Progress | None, stage: str, message: str = "") -> None:
        """Point the counter at one request's sink and name the stage.

        Call this immediately before the loop runs. It also clears any count left
        over from an earlier request.
        """
        self._progress = progress
        self._stage = stage
        self._message = message
        self._step = 0
        self._total = None
        self._last_emit = 0.0

    def begin(self, total: int | None = None) -> None:
        """Start a new loop, with `total` steps if that is known."""
        self._step = 0
        self._total = total if total is None or total > 0 else None
        self._last_emit = 0.0

    def advance(self, count: int = 1) -> None:
        """Move `count` steps on and report if the throttle allows."""
        self._step += count
        if self._total is not None and self._step > self._total:
            # The loop is longer than the total said. **Stop claiming a
            # denominator** rather than report 17/15; a scheduler whose `step`
            # fires more than once per timestep would do this.
            self._total = None
        now = time.monotonic()
        last = self._step == self._total
        if not last and self._step > 1 and now - self._last_emit < self._min_interval:
            return
        self._last_emit = now
        self._emit()

    def _emit(self) -> None:
        if self._progress is None:
            return
        extra: dict[str, Any] = {"step": self._step}
        if self._total is not None:
            extra["total"] = self._total
        self._progress(self._stage, self._message, **extra)


class _NullCounter(StepCounter):
    """A counter that reports nothing (used when no sink is bound)."""

    def _emit(self) -> None:
        return


def count_scheduler(scheduler: Any, counter: StepCounter) -> None:
    """Count a diffusers denoising loop by shadowing the scheduler's methods.

    The wrappers are set **on the instance**, so nothing global changes and no
    other pipeline in the process is affected. Calling this again on the same
    scheduler does nothing, so it is safe to call before every generation.

    `set_timesteps` decides the total: after upstream calls it, the scheduler's
    own `timesteps` is the authority on how many steps the loop will take.
    Reading it there means the count follows whatever upstream actually asked
    for, instead of a number copied out of upstream and left to drift.

    **Both wrappers keep the original signature** (`functools.wraps` sets
    `__wrapped__`, which `inspect.signature` follows). This is not cosmetic:
    diffusers decides what to pass by looking at the parameter names, in
    `retrieve_timesteps` (`"sigmas" in ... set_timesteps`) and around the
    denoising loop (`"eta" in ... step`). A wrapper taking `*args, **kwargs`
    reports no parameters at all, and upstream then refuses to run.

    Args:
        scheduler: The pipeline's scheduler instance.
        counter: Where the steps go.
    """
    if getattr(scheduler, "_hearth_counted", False):
        return

    original_set = scheduler.set_timesteps
    original_step = scheduler.step

    @functools.wraps(original_set)
    def set_timesteps(*args: Any, **kwargs: Any) -> Any:
        result = original_set(*args, **kwargs)
        timesteps = getattr(scheduler, "timesteps", None)
        try:
            total = len(timesteps) if timesteps is not None else None
        except TypeError:
            total = None
        counter.begin(total)
        return result

    @functools.wraps(original_step)
    def step(*args: Any, **kwargs: Any) -> Any:
        result = original_step(*args, **kwargs)
        counter.advance()
        return result

    scheduler.set_timesteps = set_timesteps
    scheduler.step = step
    scheduler._hearth_counted = True


def counting_tqdm(counter: StepCounter, original: Any) -> Any:
    """Build a `tqdm` replacement that reports to `counter`.

    **The count does not depend on the bar being enabled.** Upstream commonly
    writes `tqdm(..., disable=not verbose)`, and real tqdm skips its own
    bookkeeping entirely in that case, so hooking `update` would report nothing.
    Iteration is intercepted instead, which happens either way.

    The bar still draws exactly as upstream asked, so behaviour on a terminal is
    unchanged.

    Args:
        counter: Where the steps go.
        original: The real `tqdm` class to subclass.

    Returns:
        A class that can replace `tqdm` in a module.
    """

    class CountingTqdm(original):  # type: ignore[misc, valid-type]
        def __iter__(self) -> Iterator[Any]:
            total = getattr(self, "total", None)
            counter.begin(int(total) if isinstance(total, int) else None)
            for item in super().__iter__():
                counter.advance()
                yield item

    return CountingTqdm


def count_tqdm(module: Any, counter: StepCounter) -> None:
    """Replace one module's `tqdm` with a counting one.

    **The module's own attribute is replaced, not the tqdm package.** Modules
    bind `tqdm` at import time (`from tqdm import tqdm`), so patching the
    package afterwards would miss them, and patching it beforehand would catch
    unrelated loops such as hub downloads. Naming the module that holds the loop
    keeps this to the one loop that is meant.

    Calling this again on the same module does nothing.

    Args:
        module: The module holding the loop (for example upstream's sampler).
        counter: Where the steps go.
    """
    current = getattr(module, "tqdm", None)
    if current is None or getattr(current, "_hearth_counting", False):
        return
    replacement = counting_tqdm(counter, current)
    replacement._hearth_counting = True
    module.tqdm = replacement


def counted(
    iterable: Iterable[Any], counter: StepCounter, total: int | None = None
) -> Iterator[Any]:
    """Count a loop written here (no upstream involved).

    Args:
        iterable: What to iterate.
        counter: Where the steps go.
        total: The length, when `len()` does not work on `iterable`.

    Yields:
        The items of `iterable`, unchanged.
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None
    counter.begin(total)
    for item in iterable:
        counter.advance()
        yield item
