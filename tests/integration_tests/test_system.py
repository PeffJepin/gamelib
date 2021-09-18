import multiprocessing as mp
from dataclasses import dataclass

from src.gamelib import events
from src.gamelib.system import System, StopEvent, UpdateComplete


class TestSystem:
    def test_handles_events_with_functions_marked_by_handler_decorator(self, pipe_reader):
        a, b = mp.Pipe()
        system = ExampleSystem(b)
        system.start()

        a.send(ExampleEvent(5))
        value = pipe_reader(a)
        system.kill()

        assert 5 == value

    def test_process_automatically_handles_update_event(self, pipe_reader):
        a, b = mp.Pipe()
        system = ExampleSystem(b)
        system.start()

        a.send(events.Update())
        value = pipe_reader(a)
        system.kill()

        assert 123 == value

    def test_process_shuts_down_gracefully_on_stop_event(self):
        a, b = mp.Pipe()
        system = ExampleSystem(b)
        system.start()

        a.send(StopEvent())
        system.join(5)

        assert system.exitcode == 0

    def test_posts_update_complete_event_after_updating(self, pipe_reader):
        a, b = mp.Pipe()
        system = ExampleSystem(b)
        system.start()

        a.send(events.Update())
        res1 = pipe_reader(a)
        res2 = pipe_reader(a)
        system.kill()

        assert 123 == res1 and UpdateComplete(ExampleSystem) == res2

    def test_event_derived_from_system_Event_gets_sent_through_pipe_when_published(self, pipe_reader):
        a, b = mp.Pipe()
        system = SomeSystem(b)
        system.start()

        a.send(events.Update())
        res = pipe_reader(a)

        system.kill()
        assert res == SomeEvent()


@dataclass
class ExampleEvent(events.Event):
    value: int


class ExampleSystem(System):
    @events.handler(ExampleEvent)
    def _example_handler(self, event: ExampleEvent):
        self._conn.send(event.value)

    def update(self):
        self._conn.send(123)


class SomeSystem(System):
    def update(self):
        self._message_bus.post_event(SomeEvent())


class SomeEvent(SomeSystem.Event):
    pass
