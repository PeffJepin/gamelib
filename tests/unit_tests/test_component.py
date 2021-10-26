import threading
import time

import numpy as np
import pytest

from src.gamelib import ecs
from src.gamelib.ecs import StaticGlobals
from src.gamelib.ecs.component import Component


class TestComponent:
    def test_accessing_the_underlying_array(self):
        assert isinstance(ExampleComponent.array, np.ndarray)
        assert max_entities() == len(ExampleComponent.array)

    def test_late_bind_to_entity(self):
        instance = ExampleComponent(1, 2)
        assert instance.val1 is None
        assert instance.val2 is None

        instance.bind_to_entity(0)

        assert instance.val1 == 1
        assert instance.val2 == 2

    def test_binding_to_entity_on_init(self):
        instance = ExampleComponent(1, 2, entity=0)

        assert 1 == instance.val1
        assert 2 == instance.val2
        assert 1 == ExampleComponent.array[0]["val1"]
        assert 2 == ExampleComponent.array[0]["val2"]

    def test_changing_data_with_an_instance(self):
        instance = ExampleComponent(1, 2, entity=0)
        instance.val1 = 100

        assert 100 == ExampleComponent.get_for_entity(0).val1
        assert 100 == ExampleComponent.array[0]["val1"]

    def test_retrieved_components_can_mutate_data(self):
        ExampleComponent(1, 2, entity=0)

        instance = ExampleComponent.get_for_entity(0)
        instance.val1 = 100

        assert 100 == ExampleComponent.get_for_entity(0).val1
        assert 100 == ExampleComponent.array[0]["val1"]

    def test_getting_a_component_by_entity(self):
        ExampleComponent(3, 4, entity=5)

        retrieved = ExampleComponent.get_for_entity(5)

        assert 3 == retrieved.val1 and 4 == retrieved.val2

    def test_accessing_the_data_for_a_single_attribute(self):
        for i in range(10):
            ExampleComponent(1, 2, entity=i)

        assert np.all(1 == ExampleComponent.val1)
        assert np.all(2 == ExampleComponent.val2)

    def test_mutating_data_across_a_single_attribute(self):
        for i in range(10):
            ExampleComponent(1, 2, entity=i)

        ExampleComponent.val1 += 100

        assert np.all(101 == ExampleComponent.val1)

    def test_getting_a_view_of_only_existing_components(self):
        for i in range(20):
            ExampleComponent(100, 200, entity=i)

        assert 20 == len(ExampleComponent.existing)

    def test_all_components_can_be_destroyed(self):
        for i in range(10):
            ExampleComponent(1, 4, entity=i)

        ExampleComponent.destroy_all()

        assert 0 == len(ExampleComponent.existing)

    def test_a_single_component_can_be_destroyed(self):
        ExampleComponent(1, 2, entity=5)

        ExampleComponent.destroy(5)

        assert ExampleComponent.get_for_entity(5) is None

    def test_components_can_be_enumerated(self):
        ExampleComponent(0, 0, entity=0)
        ExampleComponent(1, 1, entity=5)
        ExampleComponent(5, 5, entity=7)

        expected = [
            (0, ExampleComponent.get_for_entity(0)),
            (5, ExampleComponent.get_for_entity(5)),
            (7, ExampleComponent.get_for_entity(7)),
        ]
        assert expected == list(ExampleComponent.enumerate())

    def test_getting_the_current_active_entities(self):
        for i in range(10):
            ExampleComponent(0, 0, entity=i)

        assert list(range(10)) == list(ExampleComponent.entities)

    def test_locking_access_using_context_manager(self):
        instance = ExampleComponent(0, 0, entity=0)
        running = True

        def increment(inst):
            while running:
                inst.val1 += 1

        t = threading.Thread(target=increment, args=(instance,), daemon=True)
        t.start()

        try:
            # thread shouldn't be able to increment
            # a component instance can lock the entire array
            with instance:
                first_peek = instance.val1
                for _ in range(100):
                    assert first_peek == instance.val1

            # thread should do some increments
            time.sleep(0.001)

            # thread should be locked out again
            # the component type can lock the entire array
            with ExampleComponent:
                second_peek = instance.val1
                assert second_peek != first_peek
                for _ in range(100):
                    assert second_peek == instance.val1
        finally:
            running = False

    def test_freeing_the_shared_memory_allocation(self):
        assert len(ExampleComponent.array) == max_entities()

        ExampleComponent.free()

        with pytest.raises(Exception):
            ExampleComponent.array


@pytest.fixture(autouse=True)
def ensure_cleanup():
    ecs.reset_globals({StaticGlobals.MAX_ENTITIES: 100})
    ExampleComponent.free()
    ExampleComponent.allocate()


def max_entities():
    return ecs.get_static_global(StaticGlobals.MAX_ENTITIES)


class ExampleComponent(Component):
    val1: int
    val2: float
