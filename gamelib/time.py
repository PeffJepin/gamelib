import threading
import time


class Timer:
    default_rate = 60

    def __init__(self, rate=None):
        now = time.time()
        self._previous_tick = now
        self.internal_dt = 1 / (rate or self.default_rate)
        self.next = now + self.internal_dt

    @staticmethod
    def now():
        return time.time()

    def tick(self, rate=None):
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
        return self.next - (now or time.time())


class Schedule:
    def __init__(self, *schedule, threaded=False, max_workers=None):
        self._callbacks = dict()
        self._once = set()
        self._threaded = threaded
        self._max_workers = max_workers

        for dt, callback in schedule:
            self.add(dt, callback)

    def update(self):
        now = time.time()
        timers = sorted(
            [t for t in self._callbacks.keys() if t.remaining(now=now) < 0],
            key=lambda t: t.remaining(now=now)
        )
        for t in timers:
            callback = self._get_callback(t)
            if self._threaded:
                threading.Thread(target=callback, daemon=True).start()
            else:
                callback()
            t.tick()

    def _get_callback(self, timer):
        cb = self._callbacks[timer]
        if cb in self._once:
            self._callbacks.pop(timer)
            self._once.remove(cb)
        return cb

    def add(self, dt, callback):
        timer = Timer(1 / dt)
        self._callbacks[timer] = callback

    def remove(self, callback):
        for timer, cb in self._callbacks.copy().items():
            if cb is callback:
                self._callbacks.pop(timer)

    def once(self, dt, callback):
        self.add(dt, callback)
        self._once.add(callback)