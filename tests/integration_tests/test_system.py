from contextlib import contextmanager

import pytest

from src.gamelib import SystemStop, Update, events
from src.gamelib.ecs import _EcsGlobals
from src.gamelib.events import eventhandler, Event
from src.gamelib.ecs.component import Component
from src.gamelib.ecs.system import (
    SystemUpdateComplete,
    System,
    SystemRunner,
)


class TestSystemInProcess:
    def test_automatically_handles_update(self, recorded_callback):
        with self.system_tester(ExampleSystem):
            recorded_callback.register(Response)

            events.post(Update())
            recorded_callback.await_called(1)

            assert recorded_callback.event == Response("updated")

    def test_process_shuts_down_gracefully_on_stop_event(self):
        with self.system_tester(ExampleSystem) as runner:
            events.post(SystemStop())
            runner.join(5)
            assert runner.exitcode == 0

    def test_posts_update_complete_event_after_updating(self, recorded_callback):
        with self.system_tester(ExampleSystem):
            recorded_callback.register(SystemUpdateComplete)
            events.post(Update())

            recorded_callback.await_called(1)
            assert SystemUpdateComplete(ExampleSystem) == recorded_callback.event

    def test_keyed_event_between_processes(self, recorded_callback):
        with self.system_tester(ExampleSystem):
            recorded_callback.register(Event.TEST_KEYED)
            events.post(Event(), key="TEST_KEYED")

            recorded_callback.await_called(1)

    def test_creating_a_component_in_main_process(self, recorded_callback):
        with self.system_tester(ExampleSystem):
            recorded_callback.register(Response)
            new_instance = Component1(1, 2, entity=0)

            events.post(Event(), key="CREATE_COMPONENT_TEST")

            recorded_callback.await_called(1)
            assert recorded_callback.event == Response(new_instance.values)

    def test_destroying_a_component_in_the_main_process(self, recorded_callback):
        with self.system_tester(ExampleSystem):
            recorded_callback.register(Response)
            new_instance = Component1(1, 2, entity=0)
            Component1.destroy(0)

            events.post(Event(), key="DESTROY_COMPONENT_TEST")

            recorded_callback.await_called(1)
            assert recorded_callback.event == Response(None)

    def test_synchronization_using_type(self):
        with self.system_tester(SyncTestingSystem) as runner:
            instance = Component1(0, 0, entity=0)

            events.post(Update())
            for _ in range(1_000):
                with Component1:
                    instance.val1 += 1

            runner.join(1)
            assert instance.val1 == 2_000

    def test_synchronization_using_instance(self):
        with self.system_tester(SyncTestingSystem) as runner:
            instance = Component1(0, 0, entity=0)

            events.post(Update())
            for _ in range(1_000):
                with instance:
                    instance.val1 += 1

            runner.join(1)
            assert instance.val1 == 2_000

    @contextmanager
    def system_tester(self, system, max_entities=100):
        _EcsGlobals.max_entities = max_entities
        runner = SystemRunner(system)
        try:
            runner.start()
            yield runner
        finally:
            runner.kill()


class Response(Event):
    __slots__ = ["value"]


class Component1(Component):
    val1: int
    val2: float


class ExampleSystem(System):
    COMPONENTS = (Component1,)

    @eventhandler(Event.KEYED_TEST)
    def _test_interprocess_keyed_event(self, _):
        self.raise_event(Event(), key="KEYED_RESPONSE")

    @eventhandler(Event.CREATE_COMPONENT_TEST)
    def _component_created_response(self, _):
        response = Response(Component1.get_for_entity(0).values)
        self.raise_event(response)

    @eventhandler(Event.DESTROY_COMPONENT_TEST)
    def _component_destroyed_response(self, _):
        response = Response(Component1.get_for_entity(0))
        self.raise_event(response)

    def update(self):
        self.raise_event(Response("updated"))


class SyncTestingSystem(System):
    COMPONENTS = (Component1,)

    def update(self):
        instance = Component1.get_for_entity(0)
        for _ in range(1_000):
            with Component1:
                instance.val1 += 1


@pytest.fixture(autouse=True)
def manage_component_allocation():
    # cleanup existing data
    Component1.free()
    # allocate for test invocation
    Component1.allocate()
