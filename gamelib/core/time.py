import threading
import time


class Clock:
    """Simple tool for keeping time."""

    def __init__(self, rate=60):
        """Initialize the clock.

        Parameters
        ----------
        rate : int | float
            Sample rate in times per second.
        """

        now = time.time()
        self._previous_tick = now
        self.internal_dt = 1 / rate  # number of seconds between ticks
        self.next = now + self.internal_dt

    @staticmethod
    def now():
        """Convenience method."""

        return time.time()

    def tick(self, rate=None):
        """Block until the next tick occurs. Returns the time since previous
        tick.

        Parameters
        ----------
        rate : int, optional
            Override the sample rate assigned in __init__. This effect will not
            persist past this tick.

        Returns
        -------
        float:
            The number of seconds since the previous tick() call.
        """

        dt = 1 / rate if rate else self.internal_dt
        prev_time = self._previous_tick
        remaining_time = prev_time + dt - time.time()

        if remaining_time < 0:
            time_now = time.time()
        else:
            time.sleep(remaining_time)
            time_now = time.time()

        self._previous_tick = time_now
        self.next = time_now + dt
        return time_now - prev_time

    def remaining(self, *, now=None):
        """Calculates how much time is remaining until next tick.

        Parameters
        ----------
        now : float, optional
            Keyword argument to pass in the current time instead of
            calculating it in the function call. Can be calculated with
            Timer.now().

        Returns
        -------
        float:
            The number of seconds remaining until the next tick. Can be
            negative.
        """

        return self.next - (now or time.time())


class Schedule:
    """This class manages a group of clocks linked to callbacks."""

    def __init__(self, *function_timings, threaded=False):
        """Create a mapping of clocks to callback functions.

        Parameters
        ----------
        *function_timings : tuple[float | int, callable]
            (number of seconds between calls, callback)
            ex: (5, some_function) means: call some_function every 5 seconds.
        threaded : bool, optional
            If true callbacks will be executed in a thread. Use this to
            schedule long tasks to prevent blocking the main loop.
        """

        self._callbacks = dict()
        self._once = set()
        self._threaded = threaded

        for frequency, callback in function_timings:
            self.add(callback, frequency)

    def update(self):
        """Checks the Schedule for expired clocks and calls the registered
        callback functions."""

        now = time.time()
        clocks = sorted(
            [c for c in self._callbacks.keys() if c.remaining(now=now) < 0],
            key=lambda c: c.remaining(now=now),
        )
        for c in clocks:
            callback = self._get_callback(c)
            if self._threaded:
                threading.Thread(target=callback, daemon=True).start()
            else:
                callback()
            c.tick()

    def add(self, callback, frequency):
        """Adds a callback to this schedule.

        Parameters
        ----------
        callback : Callable
        frequency : int | float
            How often to call the function.
        """

        clock = Clock(1 / frequency)
        self._callbacks[clock] = callback

    def remove(self, callback):
        """Removes a callback from this schedule, safe to call if callback is
        not actually registered.

        Parameters
        ----------
        callback : Callable
        """

        for clock, cb in self._callbacks.copy().items():
            if cb is callback:
                self._callbacks.pop(clock)

    def once(self, callback, delay):
        """Register a callback to be called only once.

        Parameters
        ----------
        callback : Callable
        delay : int | float
            How many seconds to wait. Can be negative to occur on next update.
        """

        self.add(callback, delay)
        self._once.add(callback)

    def _get_callback(self, clock):
        """Find a callback for the given clock."""

        cb = self._callbacks[clock]
        if cb in self._once:
            self._callbacks.pop(clock)
            self._once.remove(cb)
        return cb
