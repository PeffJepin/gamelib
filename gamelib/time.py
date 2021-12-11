import threading
import time


class Timer:
    """Simple tool for keeping time."""

    def __init__(self, rate=60):
        """Initialize the timer.

        Parameters
        ----------
        rate : int | float
            sample rate in times per second
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
            override the sample rate assigned in __init__

        Returns
        -------
        float
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
        """Calculates how much time is remaining until next tick. If checking
        many timers use `now` to pass in a shared time for many timers.

        Parameters
        ----------
        now : float, optional
            Keyword argument to pass in the current time instead of
            calculating it in the function call.
            Can be calculated with Timer.now().

        Returns
        -------
        float
            The number of seconds remaining until the next tick. Can be
            negative.
        """

        return self.next - (now or time.time())


class Schedule:
    """This class manages a group of timers linked to callbacks."""

    def __init__(self, *mappings, threaded=False):
        """Create a mapping of timers to callback functions.

        Parameters
        ----------
        *mappings : tuple[float | int, callable]
            (number of seconds between calls, callback)
            ex: (5, some_function) means: some_function() every 5 seconds.
        threaded : bool, optional
            If true callbacks will be executed in a thread. Use this to
            schedule long tasks to prevent blocking update.
        """

        self._callbacks = dict()
        self._once = set()
        self._threaded = threaded

        for dt, callback in mappings:
            self.add(dt, callback)

    def update(self):
        """Calls back functions for each timer tick and ticks the timers."""

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

    def _get_callback(self, timer):
        """Find a callback for the given timer."""

        cb = self._callbacks[timer]
        if cb in self._once:
            self._callbacks.pop(timer)
            self._once.remove(cb)
        return cb

    def add(self, frequency, callback):
        """Adds a callback to this schedule.

        Parameters
        ----------
        frequency : int | float
            How often to call the function.
        callback : callable
        """

        timer = Timer(1 / frequency)
        self._callbacks[timer] = callback

    def remove(self, callback):
        """Removes a callback if its found in the callbacks dict."""

        for timer, cb in self._callbacks.copy().items():
            if cb is callback:
                self._callbacks.pop(timer)

    def once(self, dt, callback):
        """Register a callback to be called only once.

        Parameters
        ----------
        dt : int | float
            How long to wait. Can be negative to occur on next update.
        callback : callable
        """

        self.add(dt, callback)
        self._once.add(callback)
