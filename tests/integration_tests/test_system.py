from contextlib import contextmanager
from typing import NamedTuple, Any

import pytest

from gamelib import events
from gamelib.ecs import _EcsGlobals
from gamelib.events import handler, Update, SystemStop
from gamelib.ecs.component import Component
from gamelib.ecs.system import (
    SystemUpdateComplete,
    System,
    SystemRunner,
)


class TestSystemInProcess:
    def test_automatically_handles_update(self, recorded_callback):
        with self.system_tester(ExampleSystem):
            recorded_callback.register(Response)

            events.post(Update(0))
            recorded_callback.await_called(1)

            assert recorded_callback.event == Response("updated")

    def test_process_shuts_down_gracefully_on_stop_event(self):
        with self.system_tester(ExampleSystem) as runner:
            events.post(SystemStop())
            runner.join(5)
            assert runner.exitcode == 0

    def test_posts_update_complete_event_after_updating(
        self, recorded_callback
    ):
        with self.system_tester(ExampleSystem):
            recorded_callback.register(SystemUpdateComplete)
            events.post(Update(0))

            recorded_callback.await_called(1)
            assert (
                SystemUpdateComplete(ExampleSystem) == recorded_callback.event
            )

    def test_creating_a_component_in_main_process(self, recorded_callback):
        with self.system_tester(ExampleSystem):
            recorded_callback.register(Response)
            new_instance = Component1(1, 2, entity=0)

            events.post(Command("CREATE_COMPONENT_TEST"))

            recorded_callback.await_called(1)
            assert recorded_callback.event == Response(new_instance.values)

    def test_destroying_a_component_in_the_main_process(
        self, recorded_callback
    ):
        with self.system_tester(ExampleSystem):
            recorded_callback.register(Response)
            new_instance = Component1(1, 2, entity=0)
            Component1.destroy(0)

            events.post(Command("DESTROY_COMPONENT_TEST"))

            recorded_callback.await_called(1)
            assert recorded_callback.event == Response(None)

    def test_synchronization_using_type(self):
        with self.system_tester(SyncTestingSystem) as runner:
            instance = Component1(0, 0, entity=0)

            events.post(Update(0))
            for _ in range(1_000):
                with Component1.locks:
                    instance.val1 += 1

            runner.join(1)
            assert instance.val1 == 2_000

    def test_synchronization_using_instance(self):
        with self.system_tester(SyncTestingSystem) as runner:
            instance = Component1(0, 0, entity=0)

            events.post(Update(0))
            for _ in range(1_000):
                with instance.locks:
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


class Command(NamedTuple):
    msg: str


class Response(NamedTuple):
    val: Any


class Component1(Component):
    val1: int
    val2: float


class ExampleSystem(System):
    COMPONENTS = (Component1,)

    @handler(Command)
    def _test_command_delegator(self, cmd):
        if cmd.msg == "CREATE_COMPONENT_TEST":
            self._component_created_response()
        elif cmd.msg == "DESTROY_COMPONENT_TEST":
            self._component_destroyed_response()

    def _component_created_response(self):
        response = Response(Component1.get_for_entity(0).values)
        self.raise_event(response)

    def _component_destroyed_response(self):
        response = Response(Component1.get_for_entity(0))
        self.raise_event(response)

    def update(self):
        self.raise_event(Response("updated"))


class SyncTestingSystem(System):
    COMPONENTS = (Component1,)

    def update(self):
        instance = Component1.get_for_entity(0)
        for _ in range(1_000):
            with Component1.locks:
                instance.val1 += 1


@pytest.fixture(autouse=True)
def manage_component_allocation():
    # cleanup existing data
    Component1.free()
    # allocate for test invocation
    Component1.allocate()
