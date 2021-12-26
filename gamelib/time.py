import threading
import time


class Timer:
    """Simple tool for keeping time."""

    def __init__(self, rate=60):
        """Initialize the timer.

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
            Override the sample rate assigned in __init__. This effect wont
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
    """This class manages a group of timers linked to callbacks."""

    def __init__(self, *function_timings, threaded=False):
        """Create a mapping of timers to callback functions.

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
            self.add(frequency, callback)

    def update(self):
        """Checks the Schedule for expired timers and calls the registered
        callback functions."""

        now = time.time()
        timers = sorted(
            [t for t in self._callbacks.keys() if t.remaining(now=now) < 0],
            key=lambda t: t.remaining(now=now),
        )
        for t in timers:
            callback = self._get_callback(t)
            if self._threaded:
                threading.Thread(target=callback, daemon=True).start()
            else:
                callback()
            t.tick()

    def add(self, frequency, callback):
        """Adds a callback to this schedule.

        Parameters
        ----------
        frequency : int | float
            How often to call the function.
        callback : Callable
        """

        timer = Timer(1 / frequency)
        self._callbacks[timer] = callback

    def remove(self, callback):
        """Removes a callback from this schedule, safe to call if callback is
        not actually registered.

        Parameters
        ----------
        callback : Callable
        """

        for timer, cb in self._callbacks.copy().items():
            if cb is callback:
                self._callbacks.pop(timer)

    def once(self, wait, callback):
        """Register a callback to be called only once.

        Parameters
        ----------
        wait : int | float
            How many seconds to wait. Can be negative to occur on next update.
        callback : Callable
        """

        self.add(wait, callback)
        self._once.add(callback)

    def _get_callback(self, timer):
        """Find a callback for the given timer."""

        cb = self._callbacks[timer]
        if cb in self._once:
            self._callbacks.pop(timer)
            self._once.remove(cb)
        return cb
