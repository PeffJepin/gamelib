import multiprocessing as mp
from contextlib import contextmanager

import numpy as np
import pytest

from src.gamelib import events
from src.gamelib.events import SystemStop
from src.gamelib.system import System, UpdateComplete, PublicAttribute


class TestSystem:
    def test_handles_events_with_functions_marked_by_handler_decorator(
        self, pipe_reader
    ):
        with self.system_tester(ExampleSystem) as conn:
            conn.send(ExampleEvent(5))
            value = pipe_reader(conn)
            assert 5 == value

    def test_process_automatically_handles_update_event(self, pipe_reader):
        with self.system_tester(ExampleSystem) as conn:
            conn.send(events.Update())
            value = pipe_reader(conn)
            assert 123 == value

    def test_process_shuts_down_gracefully_on_stop_event(self):
        a, b = mp.Pipe()
        system = ExampleSystem(b)
        system.start()

        a.send(SystemStop())
        system.join(5)

        assert system.exitcode == 0

    def test_posts_update_complete_event_after_updating(self, pipe_reader):
        with self.system_tester(ExampleSystem) as conn:
            conn.send(events.Update())
            res1 = pipe_reader(conn)
            res2 = pipe_reader(conn)
            assert 123 == res1 and UpdateComplete(ExampleSystem) == res2

    def test_event_derived_from_system_Event_gets_sent_through_pipe_when_published(
        self, pipe_reader
    ):
        with self.system_tester(SomeSystem) as conn:
            conn.send(events.Update())
            res = pipe_reader(conn)
            assert res == SomeEvent()

    def test_public_attribute_access_before_init(self):
        with pytest.raises(FileNotFoundError):
            arr = ExampleComponent.nums

    def test_public_attribute_access_after_init(self):
        with self.system_tester(ExampleSystem) as conn:
            assert all(ExampleComponent.nums[:] == 0)

    @contextmanager
    def system_tester(self, sys_type):
        System.MAX_ENTITIES = 16
        a, b = mp.Pipe()
        sys = sys_type(b)
        sys.start()
        try:
            yield a
        finally:
            a.send(SystemStop())
            sys.join(10)
            if sys.exitcode is None:
                sys.kill()
                assert False  # system not joining is indicative of an error


class ExampleEvent(events.Event):
    __slots__ = ["value"]

    value: int


class ExampleSystem(System):
    @events.handler(ExampleEvent)
    def _example_handler(self, event: ExampleEvent):
        self._conn.send(event.value)

    def update(self):
        self._conn.send(123)


class ExampleComponent(ExampleSystem.Component):
    nums = PublicAttribute(np.uint8)


class SomeSystem(System):
    def update(self):
        self._message_bus.post_event(SomeEvent())


class SomeEvent(SomeSystem.Event):
    pass
