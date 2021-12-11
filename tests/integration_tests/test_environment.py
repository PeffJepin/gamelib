from collections import defaultdict
from typing import NamedTuple, Any

import numpy as np
import pytest

from gamelib import events
from gamelib.events import Update
from gamelib.ecs import EntityDestroyed
from gamelib.ecs.environment import Environment
from gamelib.ecs.system import SystemUpdateComplete, System
from gamelib.ecs.component import Component


class Command(NamedTuple):
    msg: str = "default"


class Response(NamedTuple):
    val: Any = "default"


class Component1(Component):
    attr1: int
    attr2: float


class Component2(Component):
    attr1: int
    attr2: float


class _TestSystem(System):
    # system that consumes command events and returns:
    # ("ClassName", eval(Command.msg))

    # record is a counter of event types handled
    record = defaultdict(int)

    @events.handler(Command)
    def _command(self, cmd):
        self.record[Command] += 1

        try:
            val = eval(cmd.msg)
        except NameError:
            val = cmd.msg

        res = Response((self.__class__.__name__, val))
        self.raise_event(res)


class LocalSystem(_TestSystem):
    pass


class ProcessSystem(_TestSystem):
    pass


@pytest.fixture(autouse=True, scope="function")
def cleanup():
    Component1.free()
    ExampleEnvironment.record.clear()
    LocalSystem.record.clear()


class ExampleEnvironment(Environment):
    # injected for testing
    ASSET_LABELS = [f"asset{i + 1}" for i in range(6)]
    record = defaultdict(int)

    # normal implementation
    COMPONENTS = [Component1, Component2]
    LOCAL_SYSTEMS = [LocalSystem]
    PROCESS_SYSTEMS = [ProcessSystem]
    ASSETS = []

    @events.handler(Command)
    def command_handler(self, cmd):
        self.record[Command] += 1

    @events.handler(SystemUpdateComplete)
    def update_complete_counter(self, _):
        self.record[SystemUpdateComplete] += 1


class TestEnvironment:
    def test_does_not_handle_events_before_entering(self):
        env = ExampleEnvironment()
        events.post(Command())

        assert env.record[Command] == 0

    def test_handles_events_after_entering(self):
        with ExampleEnvironment() as env:
            events.post(Command())

            assert env.record[Command] == 1

    def test_does_not_handle_events_after_exiting(self):
        env = ExampleEnvironment()

        with env:
            pass
        events.post(Command())

        assert env.record[Command] == 0

    def test_local_systems_dont_handle_events_before_entering(self):
        env = ExampleEnvironment()

        events.post(Command())

        assert LocalSystem.record[Command] == 0

    def test_local_systems_begin_handling_events_after_entering(self):
        with ExampleEnvironment():
            events.post(Command())

        assert LocalSystem.record[Command] == 1

    def test_local_systems_stop_handling_events_after_exiting(self):
        env = ExampleEnvironment()
        with env:
            pass

        events.post(Command())

        assert LocalSystem.record[Command] == 0

    def test_process_systems_dont_handle_events_before_entering(
        self, recorded_callback
    ):
        env = ExampleEnvironment()
        events.subscribe(Response, recorded_callback)

        events.post(Command())

        with pytest.raises(TimeoutError):
            recorded_callback.await_called(1, timeout=0.1)

    def test_process_systems_handle_events_after_entering(
        self, recorded_callback
    ):
        events.subscribe(Response, recorded_callback)

        with ExampleEnvironment():
            events.post(Command("hello"))

        recorded_callback.await_silence()
        assert Response(("ProcessSystem", "hello")) in recorded_callback.events

    def test_process_systems_stop_handling_events_after_exiting(
        self, recorded_callback
    ):
        events.subscribe(Response, recorded_callback)
        env = ExampleEnvironment()

        with env:
            pass
        events.post(Command())

        with pytest.raises(TimeoutError):
            recorded_callback.await_called(1, timeout=0.1)

    def test_components_memory_is_not_allocated_before_loading(self):
        env = ExampleEnvironment()

        with pytest.raises(Exception):
            for component in env.COMPONENTS:
                assert component.array

    def test_component_memory_is_allocated_after_loading(self):
        with ExampleEnvironment() as env:
            for component in env.COMPONENTS:
                assert len(component.array) == env.MAX_ENTITIES

    def test_component_memory_is_not_allocated_after_exiting(self):
        env = ExampleEnvironment()

        with env:
            pass

        with pytest.raises(Exception):
            for component in env.COMPONENTS:
                assert component.array

    def test_component_data_is_shared_across_processes(
        self, recorded_callback
    ):
        events.subscribe(Response, recorded_callback)

        with ExampleEnvironment():
            Component1.array[:] = 123
            events.post(Command("Component1.array"))

            recorded_callback.await_silence()
            for res in recorded_callback.events:
                process_name, value = res.val
                assert np.all(value == Component1.array)
                return
            raise AssertionError("No response from process.")

    def test_each_system_posts_an_instance_of_system_update_complete(
        self, recorded_callback
    ):
        events.subscribe(SystemUpdateComplete, recorded_callback)

        with ExampleEnvironment() as env:
            events.post(Update(0))

            expected_number_of_calls = len(env.PROCESS_SYSTEMS) + len(
                env.LOCAL_SYSTEMS
            )
            recorded_callback.await_called(expected_number_of_calls)

    def test_creating_an_entity_within_context_manager(self):
        with ExampleEnvironment() as env:
            component = Component1(1, 2)
            entity = env.create_entity(component)

            assert entity is not None

    def test_creating_an_entity_outside_context_manager(self):
        env = ExampleEnvironment()
        component = Component1(1, 1)

        with pytest.raises(Exception):
            env.create_entity(component)

    def test_destroying_an_entity_that_exists(self):
        with ExampleEnvironment() as env:
            component = Component1(3, 4)
            entity = env.create_entity(component)

            events.post(EntityDestroyed(entity))

            assert Component1.get_for_entity(entity) is None

    def test_destroying_an_entity_that_no_longer_exists_does_not_error(self):
        with ExampleEnvironment() as env:
            component = Component1(3, 4)
            entity = env.create_entity(component)

            events.post(EntityDestroyed(entity))

    def test_entity_with_multiple_components(self):
        with ExampleEnvironment() as env:
            component1 = Component1(1, 2)
            component2 = Component2(3, 4)
            entity = env.create_entity(component1, component2)

            assert Component1.values is not None
            assert Component2.values is not None
            events.post(EntityDestroyed(entity))

            assert Component1.get_for_entity(entity) is None
            assert Component2.get_for_entity(entity) is None

    def test_entities_recycle_when_destroyed(self):
        with ExampleEnvironment() as env:
            component = Component1(1, 2)

            entity1 = env.create_entity(component)
            events.post(EntityDestroyed(entity1))
            entity2 = env.create_entity(component)

            assert entity1 == entity2

    def test_errors_upon_creating_too_many_entities(self):
        with ExampleEnvironment(max_entities=1) as env:
            component = Component1(1, 2)
            env.create_entity(component)

            with pytest.raises(Exception):
                env.create_entity(component)
