import multiprocessing as mp
from contextlib import contextmanager

import numpy as np
import pytest

from src.gamelib import events, SystemStop, Update
from src.gamelib.system import System, UpdateComplete, PublicAttribute


class TestSystem:
    def test_events_are_pooled_until_update(self, pipe_reader):
        with self.system_tester(ExampleSystem) as conn:
            conn.send(ExampleEvent(10))
            assert pipe_reader(conn, 0.1) is None
            conn.send(Update())
            assert pipe_reader(conn) is not None

    def test_event_resolution_order(self, pipe_reader):
        with self.system_tester(ExampleSystem) as conn:
            conn.send(ExampleEvent(10))
            conn.send(ExampleEvent(15))
            conn.send(Update())

            responses = [pipe_reader(conn, 3) for _ in range(4)]
            assert ['updated', 10, 15, UpdateComplete(ExampleSystem)] == responses

    def test_process_automatically_handles_update_event(self, pipe_reader):
        with self.system_tester(ExampleSystem) as conn:
            conn.send(Update())
            value = pipe_reader(conn)
            assert 'updated' == value

    def test_process_shuts_down_gracefully_on_stop_event(self):
        a, b = mp.Pipe()
        system = ExampleSystem(b)
        system.start()

        a.send(SystemStop())
        system.join(5)

        assert system.exitcode == 0

    def test_posts_update_complete_event_after_updating(self, pipe_reader):
        with self.system_tester(ExampleSystem) as conn:
            conn.send(Update())
            res1 = pipe_reader(conn)
            res2 = pipe_reader(conn)
            assert 'updated' == res1 and UpdateComplete(ExampleSystem) == res2

    def test_event_derived_from_system_Event_gets_sent_through_pipe_when_published(
        self, pipe_reader
    ):
        with self.system_tester(SomeSystem) as conn:
            conn.send(Update())
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
    @events.eventhandler(ExampleEvent)
    def _example_handler(self, event: ExampleEvent):
        self._conn.send(event.value)

    def update(self):
        self._conn.send('updated')


class ExampleComponent(ExampleSystem.Component):
    nums = PublicAttribute(np.uint8)


class SomeSystem(System):
    def update(self):
        self._message_bus.post_event(SomeEvent())


class SomeEvent(SomeSystem.Event):
    pass
