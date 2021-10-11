from contextlib import contextmanager

import numpy as np
import pytest

from src.gamelib import events, SystemStop, Update, sharedmem, Config, EntityDestroyed
from src.gamelib.events import eventhandler, Event
from src.gamelib.system import SystemUpdateComplete, System, ProcessSystem
from src.gamelib.component import PublicAttribute, ComponentCreated, ArrayAttribute


class TestSystem:
    def test_events_are_pooled_until_update(self, pipe_reader):
        with self.system_tester(ExampleSystem) as conn:
            conn.send((ExampleEvent(10), None))

            with pytest.raises(TimeoutError):
                pipe_reader(conn, timeout=0.1)

            conn.send((Update(), None))
            assert pipe_reader(conn) is not None

    def test_event_resolution_order(self, pipe_reader):
        with self.system_tester(ExampleSystem) as conn:
            conn.send((ExampleEvent(10), None))
            conn.send((ExampleEvent(15), None))
            conn.send((Update(), None))

            responses = pipe_reader(conn, n=4)
            assert [
                10,  # first sent event is handled
                15,  # second sent event is handled
                "updated",  # update() function runs after events are handled
                (
                    SystemUpdateComplete(ExampleSystem),
                    None,
                ),  # update complete response after all else
            ] == responses

    def test_process_automatically_handles_update_event(self, pipe_reader):
        with self.system_tester(ExampleSystem) as conn:
            conn.send((Update(), None))
            value = pipe_reader(conn)
            assert "updated" == value

    def test_process_shuts_down_gracefully_on_stop_event(self):
        sharedmem.allocate(ExampleSystem.shared_specs)
        conn, process = ExampleSystem.run_in_process()
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
                (Event(), "KEYED_RESPONSE"),
                "updated",
                (SystemUpdateComplete(ExampleSystem), None),
            ]
            responses = pipe_reader(conn, n=3)
            assert expected == responses

    def test_components_created_by_event(self):
        system = LocalSystem()
        event = ComponentCreated(entity_id=0, type=LocalComponent, args=(1, 2))
        events.post_event(event)

        component = system.get_component(type_=LocalComponent, entity_id=0)
        assert 0 == component.entity_id
        assert (1, 2) == component.args
        assert isinstance(component, LocalComponent)

    def test_components_destroyed_by_event(self):
        system = LocalSystem()
        event = ComponentCreated(entity_id=0, type=LocalComponent, args=(1, 2))
        events.post_event(event)
        events.post_event(EntityDestroyed(0))

        component = system.get_component(type_=LocalComponent, entity_id=0)
        assert component is None

    def test_array_attributes_are_masked_after_being_destroyed(self):
        system = LocalSystem()
        event = ComponentCreated(entity_id=0, type=LocalComponent, args=(1, 2))
        events.post_event(event)

        assert 100 == LocalComponent.arr[0]

        events.post_event(EntityDestroyed(0))
        assert not LocalComponent.arr[0]

    @contextmanager
    def system_tester(self, sys_type, max_entities=100):
        Config.MAX_ENTITIES = max_entities
        sharedmem.allocate(sys_type.shared_specs)
        conn, process = sys_type.run_in_process()
        try:
            yield conn
        finally:
            process.kill()


class TestPublicAttribute:
    def test_public_attribute_does_not_work_before_allocation(self, attr):
        with pytest.raises(Exception):
            attr = ExampleComponent.public_attr

    def test_access_can_be_made_after_allocation(self, allocated_attr):
        assert all(ExampleComponent.public_attr[:] == 0)

    def test_cannot_be_accessed_after_closed(self):
        attr = vars(ExampleComponent)["public_attr"]
        sharedmem.allocate(attr.shared_specs)
        arr = ExampleComponent.public_attr

        sharedmem.unlink()
        attr.close_view()
        with pytest.raises(Exception):
            arr = ExampleComponent.public_attr

    def test_access_is_read_only_if_component_not_in_local_components(
        self, allocated_attr
    ):
        Config.local_components = []
        with pytest.raises(ValueError):
            ExampleComponent.public_attr += 1

    def test_access_is_not_read_only_if_component_is_local(self, allocated_attr):
        Config.local_components.append(ExampleComponent)
        ExampleComponent.public_attr[:] = 100

        assert all(100 == ExampleComponent.public_attr)

    def test_access_on_type_returns_np_ndarray(self, allocated_attr):
        Config.local_components.append(ExampleComponent)

        assert isinstance(ExampleComponent.public_attr, np.ndarray)

    def test_access_on_instance_returns_index_into_array(self, allocated_attr):
        Config.local_components.append(ExampleComponent)
        ExampleComponent.public_attr[0] = 100
        ExampleComponent.public_attr[1] = 200

        c1 = ExampleComponent(0)
        c2 = ExampleComponent(1)

        assert c1.public_attr == 100
        assert c2.public_attr == 200

    def test_changes_not_reflected_until_update(self, allocated_attr):
        # get read view
        read_view = ExampleComponent.public_attr

        # close attribute and set component as local se we can get write view
        allocated_attr.close_view()
        Config.local_components.append(ExampleComponent)

        # now we get a write view because component is local
        write_view = ExampleComponent.public_attr

        write_view[:] = 123
        assert all(123 == write_view)
        assert all(0 == read_view)

        allocated_attr.update_buffer()
        assert all(123 == read_view)

    def test_array_size_dictated_by_config_max_entities(self, attr):
        Config.MAX_ENTITIES = 16
        sharedmem.allocate(attr.shared_specs)

        assert len(ExampleComponent.public_attr) == 16

    @pytest.fixture
    def attr(self):
        attr = vars(ExampleComponent)["public_attr"]
        try:
            yield attr
        finally:
            attr.close_view()

    @pytest.fixture
    def allocated_attr(self):
        attr = vars(ExampleComponent)["public_attr"]
        sharedmem.allocate(attr.shared_specs)

        try:
            yield attr
        finally:
            attr.close_view()


class ExampleEvent(events.Event):
    __slots__ = ["value"]

    value: int


class Response(Event):
    __slots__ = ["value"]


class ExampleSystem(ProcessSystem):
    @eventhandler(ExampleEvent)
    def _example_handler(self, event: ExampleEvent):
        self._conn.send(event.value)

    @eventhandler(Event.KEYED_TEST)
    def _test_interprocess_keyed_event(self, _):
        self.raise_event(Event(), key="KEYED_RESPONSE")

    def update(self):
        self._conn.send("updated")


class ExampleComponent(ExampleSystem.Component):
    nums = PublicAttribute(np.uint8)
    public_attr = PublicAttribute(int)

    def __init__(self, entity_id, *args):
        super().__init__(entity_id)
        self.args = args


class LocalSystem(System):
    pass


class LocalComponent(LocalSystem.Component):

    arr = ArrayAttribute(int)

    def __init__(self, entity_id, *args):
        super().__init__(entity_id)
        self.args = args
        self.arr = 100
