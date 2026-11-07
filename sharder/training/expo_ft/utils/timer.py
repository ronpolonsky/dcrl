import time
from collections import defaultdict
from contextlib import contextmanager


class Timer:
    """Span timer in the style of hil-serl's timer_utils.

    Use `with timer.context("key"):` for block-scoped spans, or tick/tock for
    spans that don't nest cleanly. `get_times_ms(reset=True)` returns per-key
    average durations in milliseconds; with one tick/tock per key between
    resets this is just the per-span duration.

    Not thread-safe: durations measured on worker threads should be computed
    there and passed back as plain numbers.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = defaultdict(int)
        self.times = defaultdict(float)
        self.start_times = {}

    def tick(self, key):
        if key in self.start_times:
            raise ValueError(f"Timer is already ticking for key: {key}")
        self.start_times[key] = time.time()

    def tock(self, key):
        if key not in self.start_times:
            raise ValueError(f"Timer is not ticking for key: {key}")
        self.counts[key] += 1
        self.times[key] += time.time() - self.start_times[key]
        del self.start_times[key]

    @contextmanager
    def context(self, key):
        self.tick(key)
        try:
            yield
        finally:
            self.tock(key)

    def get_times_ms(self, reset=True):
        ret = {key: 1000.0 * self.times[key] / self.counts[key] for key in self.counts}
        if reset:
            self.reset()
        return ret
