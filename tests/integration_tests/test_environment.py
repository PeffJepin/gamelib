import numpy as np
import pytest

from src.gamelib import Update, events
from src.gamelib.ecs import EntityDestroyed
from src.gamelib.ecs.environment import Environment
from src.gamelib.events import eventhandler, Event
from src.gamelib.ecs.system import SystemUpdateComplete, System
from src.gamelib.ecs.component import Component


class TestEnvironment:
    def test_does_not_handle_events_before_entering(self):
        env = ExampleEnvironment()
        events.post(Event(), "ABC")

        assert 0 == env.abc_event_handled

    def test_handles_events_after_entering(self):
        with ExampleEnvironment() as env:
            events.post(Event(), "ABC")

            assert 1 == env.abc_event_handled

    def test_does_not_handle_events_after_exiting(self):
        env = ExampleEnvironment()

        with env:
            pass
        events.post(Event(), "ABC")

        assert 0 == env.abc_event_handled

    def test_local_systems_dont_handle_events_before_entering(self, recorded_callback):
        env = ExampleEnvironment()
        events.register(Response.LOCAL_EVENT, recorded_callback)

        events.post(Event(), key="LOCAL_EVENT")

        assert not recorded_callback.called

    def test_local_systems_begin_handling_events_after_entering(
        self, recorded_callback
    ):
        events.register(Response.LOCAL_EVENT, recorded_callback)

        with ExampleEnvironment():
            events.post(Event(), key="LOCAL_EVENT")

            assert recorded_callback.called

    def test_local_systems_stop_handling_events_after_exiting(self, recorded_callback):
        events.register(Response.LOCAL_EVENT, recorded_callback)

        env = ExampleEnvironment()
        with env:
            pass

        events.post(Event(), key="LOCAL_EVENT")

        assert not recorded_callback.called

    def test_process_systems_dont_handle_events_before_entering(
        self, recorded_callback
    ):
        env = ExampleEnvironment()
        events.register(Response.BASE_PROCESS_TEST, recorded_callback)

        events.post(Event(), key="BASE_PROCESS_TEST")

        with pytest.raises(TimeoutError):
            recorded_callback.await_called(1, timeout=0.1)

    def test_process_systems_handle_events_after_entering(self, recorded_callback):
        events.register(Response.BASE_PROCESS_TEST, recorded_callback)

        with ExampleEnvironment():
            events.post(Event(), key="BASE_PROCESS_TEST")

            recorded_callback.await_called(1)

    def test_process_systems_stop_handling_events_after_exiting(
        self, recorded_callback
    ):
        events.register(Response.BASE_PROCESS_TEST, recorded_callback)
        env = ExampleEnvironment()

        with env:
            pass
        events.post(Event(), key="BASE_PROCESS_TEST")

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

    def test_component_data_is_shared_across_processes(self, recorded_callback):
        events.register(Response.INTERPROCESS_COMPONENT_DATA, recorded_callback)

        with ExampleEnvironment():
            Component1.array[:] = 123
            events.post(Event(), key="INTERPROCESS_COMPONENT_DATA")

            recorded_callback.await_called(1)
            assert np.all(recorded_callback.event.value == Component1.array)

    def test_each_system_posts_an_instance_of_system_update_complete(
        self, recorded_callback
    ):
        events.register(SystemUpdateComplete, recorded_callback)

        with ExampleEnvironment() as env:
            events.post(Update())

            expected_number_of_calls = len(env.PROCESS_SYSTEMS) + len(env.LOCAL_SYSTEMS)
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


class Response(Event):
    __slots__ = ["value"]


class Component1(Component):
    attr1: int
    attr2: float


class Component2(Component):
    attr1: int
    attr2: float


class ProcessSystem1(System):
    @eventhandler(Event.BASE_PROCESS_TEST)
    def _test_base_event_handling(self, _):
        self.raise_event(Response(""), key="BASE_PROCESS_TEST")

    @eventhandler(Event.INTERPROCESS_COMPONENT_DATA)
    def _test_components_data(self, _):
        self.raise_event(Response(Component1.array), key="INTERPROCESS_COMPONENT_DATA")


class LocalSystem1(System):
    @eventhandler(Event.LOCAL_EVENT)
    def _test_handling_events_locally(self, _):
        self.raise_event(Response(""), key="LOCAL_EVENT")


class ExampleEnvironment(Environment):
    # injected for testing
    ASSET_LABELS = [f"asset{i + 1}" for i in range(6)]
    abc_event_handled = 0
    updates_completed = 0

    # normal implementation
    COMPONENTS = [Component1, Component2]
    LOCAL_SYSTEMS = [LocalSystem1]
    PROCESS_SYSTEMS = [ProcessSystem1]
    ASSETS = []

    @eventhandler(Event.ABC)
    def abc_handler(self, _):
        self.abc_event_handled += 1

    @eventhandler(SystemUpdateComplete)
    def update_complete_counter(self, _):
        self.updates_completed += 1


@pytest.fixture(autouse=True)
def ensure_component_memory_cleanup():
    Component1.free()
