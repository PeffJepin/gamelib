from contextlib import contextmanager

import numpy as np
import pytest

from src.gamelib import events, SystemStop, Update
from src.gamelib.events import eventhandler, Event
from src.gamelib.system import SystemUpdateComplete, PublicAttribute, System
from ..conftest import PatchedSystem


class TestSystem:
    def test_events_are_pooled_until_update(self, pipe_reader):
        with self.system_tester(ExampleSystem) as conn:
            conn.send((ExampleEvent(10), None))
            assert pipe_reader(conn, timeout=0.1) is None

            conn.send((Update(), None))
            assert pipe_reader(conn) is not None

    def test_event_resolution_order(self, pipe_reader):
        with self.system_tester(ExampleSystem) as conn:
            conn.send((ExampleEvent(10), None))
            conn.send((ExampleEvent(15), None))
            conn.send((Update(), None))

            responses = pipe_reader(conn, n=4)
            assert [
                "updated",  # update() function runs once Update event comes
                10,  # first sent event is handled
                15,  # second sent event is handled
                (SystemUpdateComplete(ExampleSystem), None),  # update complete response after all else
            ] == responses

    def test_process_automatically_handles_update_event(self, pipe_reader):
        with self.system_tester(ExampleSystem) as conn:
            conn.send((Update(), None))
            value = pipe_reader(conn)
            assert "updated" == value

    def test_process_shuts_down_gracefully_on_stop_event(self):
        System.SHARED_BLOCK = ExampleSystem.make_test_shm_block()
        conn, process = ExampleSystem.run_in_process(max_entities=10)
        conn.send((SystemStop(), None))
        process.join(5)
        assert process.exitcode == 0

    def test_posts_update_complete_event_after_updating(self, pipe_reader):
        with self.system_tester(ExampleSystem) as conn:
            conn.send((Update(), None))

            responses = pipe_reader(conn, n=2)
            expected = ["updated", (SystemUpdateComplete(ExampleSystem), None)]
            assert expected == responses

    def test_public_attribute_access_before_init(self):
        with pytest.raises(Exception):
            arr = ExampleComponent.nums

    def test_public_attribute_access_after_init(self):
        with self.system_tester(ExampleSystem):
            assert all(ExampleComponent.nums[:] == 0)

    def test_keyed_event_between_processes(self, pipe_reader):
        with self.system_tester(ExampleSystem) as conn:
            conn.send((Event(), "KEYED_TEST"))
            conn.send((Update(), None))

            expected = [
                "updated",
                (Event(), "KEYED_RESPONSE"),
                (SystemUpdateComplete(ExampleSystem), None),
            ]
            responses = pipe_reader(conn, n=3)
            assert expected == responses

    @contextmanager
    def system_tester(self, sys_type, max_entities=100):
        System.set_shared_block(sys_type.make_test_shm_block())
        conn, process = sys_type.run_in_process(max_entities)
        try:
            yield conn
        finally:
            process.join()


class ExampleEvent(events.Event):
    __slots__ = ["value"]

    value: int


class ExampleSystem(PatchedSystem):
    @eventhandler(ExampleEvent)
    def _example_handler(self, event: ExampleEvent):
        self._conn.send(event.value)

    @eventhandler(Event.KEYED_TEST)
    def _test_interprocess_keyed_event(self, event):
        self.post_event(Event(), key="KEYED_RESPONSE")

    def update(self):
        self._conn.send("updated")


class ExampleComponent(ExampleSystem.Component):
    nums = PublicAttribute(np.uint8)
