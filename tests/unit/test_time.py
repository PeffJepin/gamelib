import time
import pytest

from gamelib.core.time import Clock, Schedule
from tests.conftest import RecordedCallback


def assert_almost_equal(t1, t2):
    diff = t2 - t1
    assert pytest.approx(0, abs=0.001) == diff


def do_for(t, func):
    start = time.time()
    while time.time() - start < t:
        func()


@pytest.mark.ignore_github
class TestTimer:
    def test_tick_sets_next_time(self):
        timer = Clock()
        rate = 60

        timer.tick(rate)

        assert_almost_equal(time.time() + 1 / rate, timer.next)

    def test_tick_sleeps_until_next(self):
        timer = Clock()
        timer.tick(60)

        next = timer.next
        timer.tick(60)

        assert_almost_equal(time.time(), next)

    def test_remaining_no_args(self):
        timer = Clock()

        timer.tick(30)
        time.sleep(0.01)

        assert_almost_equal(1 / 30 - 0.01, timer.remaining())

    def test_remaining_given_now_as_arg(self):
        timer = Clock()

        timer.tick(60)
        remaining = timer.remaining(now=time.time() + 0.01)

        assert_almost_equal(1 / 60 - 0.01, remaining)

    def test_now(self):
        assert_almost_equal(time.time(), Clock.now())

    def test_default_tick_rate(self):
        timer1 = Clock(120)
        timer2 = Clock()

        timer1.tick()
        assert_almost_equal(time.time() + 1 / 120, timer1.next)

        timer2.tick()
        assert_almost_equal(time.time() + timer2.internal_dt, timer2.next)


@pytest.mark.ignore_github
class TestSchedule:
    def test_scheduled_callbacks(self):
        cbs = [RecordedCallback() for _ in range(5)]
        schedule = Schedule(
            (0.001, cbs[0]),
            (0.002, cbs[1]),
            (0.003, cbs[2]),
            (0.004, cbs[3]),
            (0.005, cbs[4]),
        )

        do_for(0.1, schedule.update)

        expected = [100, 50, 33, 25, 20]
        actual = [cb.called for cb in cbs]
        diffs = [abs(v1 - v2) for v1, v2 in zip(expected, actual)]
        print(diffs)
        assert all(d < 4 for d in diffs)

    def test_remove_from_schedule(self, recorded_callback):
        schedule = Schedule((0.001, recorded_callback))
        schedule.remove(recorded_callback)

        do_for(0.01, schedule.update)

        assert not recorded_callback.called

    def test_adding_to_a_schedule(self, recorded_callback):
        schedule = Schedule()
        schedule.add(recorded_callback, 0.001)

        do_for(0.01, schedule.update)

        assert 7 <= recorded_callback.called <= 12

    def test_schedule_once(self, recorded_callback):
        schedule = Schedule()
        schedule.once(recorded_callback, -1)

        schedule.update()
        assert recorded_callback.called == 1

        schedule.update()
        assert recorded_callback.called == 1

    def test_threaded(self, recorded_callback):
        schedule = Schedule((0.01, recorded_callback), threaded=True)
        schedule.once(lambda: time.sleep(0.09), -1)

        do_for(0.1, schedule.update)

        assert recorded_callback.called > 5
