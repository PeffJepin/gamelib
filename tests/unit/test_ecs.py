import numpy as np

import pytest
import threading
import time

from gamelib import ecs
from gamelib import gl

from ..conftest import assert_approx


@pytest.fixture(autouse=True)
def cleanup():
    ecs.Entity.clear()


class TestIdGenerator:
    def test_ids_are_given_sequentially(self):
        gen = ecs.IdGenerator()
        
        assert next(gen) == 0
        assert next(gen) == 1
        assert next(gen) == 2

    def test_ids_can_be_recycled(self):
        gen = ecs.IdGenerator()

        for _ in range(2):
            next(gen)

        gen.recycle(0)
        
        assert next(gen) == 0

    def test_always_lowest_possible_id(self):
        gen = ecs.IdGenerator()

        for _ in range(5):
            # 0, 1, 2, 3, 4
            next(gen)

        gen.recycle(3)
        gen.recycle(1)
        gen.recycle(2)

        assert next(gen) == 1
        assert next(gen) == 2
        assert next(gen) == 3
        assert next(gen) == 5

    def test_knows_largest_id_in_use(self):
        gen = ecs.IdGenerator()
        for _ in range(5):
            next(gen)

        assert gen.largest_active == 4
        
        gen.recycle(1)
        gen.recycle(3)
        gen.recycle(4)
        assert gen.largest_active == 2

    def test_set_state_stops_recycling_irrelevant_ids(self):
        gen = ecs.IdGenerator()
        for _ in range(5):
            next(gen)

        gen.recycle(4)
        gen.set_state(3)

        # state set to lower than 4, so its discarded
        # from the recycled ids
        assert next(gen) == 3
        assert next(gen) == 4
        assert next(gen) == 5

    def test_set_state_does_not_discard_relevant_ids(self):
        gen = ecs.IdGenerator()
        for _ in range(5):
            next(gen)

        gen.recycle(1)
        gen.set_state(3)

        # state is 3, but there are still recyclables to get through
        assert next(gen) == 1
        assert next(gen) == 3
        assert next(gen) == 4


class ExampleComponent1(ecs.Component):
    x: float
    y: float


class ExampleComponent2(ecs.Component):
    z: float
    w: float


class TestComponent:
    def test_creating_a_component(self):
        component = ExampleComponent1(123, 124)

        assert component.x == 123
        assert component.y == 124
        assert 123 in ExampleComponent1.x
        assert 124 in ExampleComponent1.y
        actual = ExampleComponent1.get_raw_arrays()[component.id]
        assert actual["x"], actual["y"] == (123, 124)

    def test_getting_a_component(self):
        component = ExampleComponent1(1.23, 4.56)

        assert ExampleComponent1.get(component.id) == component

    def test_destroying_a_component(self):
        component1 = ExampleComponent1(0.1234, 0.1234)
        component2 = ExampleComponent1(321, 321)

        assert component1 == ExampleComponent1.get(component1.id)
        assert component2 == ExampleComponent1.get(component2.id)

        component1.destroy(component1.id)
        ExampleComponent1.destroy(component2.id)

        assert ExampleComponent1.get(component1.id) is None
        assert ExampleComponent1.get(component2.id) is None

    def test_safe_to_destroy_nonexisting_component(self):
        ExampleComponent1.destroy(1_000)
        assert True  # above should not error

    def test_length_of_component(self):
        length = ExampleComponent1.num_elements
        ExampleComponent1(0, 0)

        assert ExampleComponent1.num_elements == length + 1

    def test_equality_comparison(self):
        assert ExampleComponent1(123.0, 123.0) == (123.0, 123.0)
        assert ExampleComponent1(123.0, 123.0) != (123.1, 123.1)

    def test_getting_an_existing_component_by_id(self):
        id = ExampleComponent1(519, 542).id

        looked_up = ExampleComponent1.get(id)
        assert looked_up == (519, 542)

        looked_up.x = 1024
        looked_up.y = 1025

        assert ExampleComponent1.get(id) == (1024, 1025)

    def test_id_when_created(self):
        component = ExampleComponent1(1234, 1234)

        assert component.id is not None
        assert ExampleComponent1.get(component.id) == (1234, 1234)
        assert ExampleComponent1(1234, 1234).id == component.id + 1

    def test_recycling_an_id(self):
        component1 = ExampleComponent1(0, 0)
        c1_id = component1.id
        ExampleComponent1.destroy(component1.id)
        component2 = ExampleComponent1(1, 1)

        assert component2.id == c1_id

    def test_ids_back_to_0_after_clear(self):
        for _ in range(10):
            ExampleComponent1(0, 0)
        ExampleComponent1.clear()

        assert ExampleComponent1(0, 0).id == 0

    def test_clearing_a_component(self):
        for _ in range(10):
            component = ExampleComponent1(0, 0)

        assert ExampleComponent1.num_elements >= 10

        ExampleComponent1.clear()

        assert ExampleComponent1.num_elements == 0
        assert component.x is None
        assert component.y is None

    def test_masked_after_being_destroyed(self):
        component1 = ExampleComponent1(1, 1)
        component2 = ExampleComponent1(2, 2)
        component3 = ExampleComponent1(3, 3)

        ExampleComponent1.destroy(component2.id)

        assert ExampleComponent1.get(component1.id) == component1
        assert ExampleComponent1.get(component3.id) == component3
        assert ExampleComponent1.num_elements == 2

    def test_mutating_data_by_an_instance(self):
        component = ExampleComponent1(1, 2)
        assert ExampleComponent1.get(component.id) == (1, 2)

        component.x = 132
        component.y = 1234

        assert (132, 1234) == ExampleComponent1.get(component.id)

    def test_mutating_data_by_class_array(self):
        component1 = ExampleComponent1(0, 0)
        component2 = ExampleComponent1(1, 1)

        ExampleComponent1.x += 100

        assert (100, 0) == component1
        assert (101, 1) == component2

    def test_locking_access_using_context_manager(self):
        instance = ExampleComponent1(0, 0)
        running = True

        def increment(inst):
            while running:
                inst.x += 1
                inst.y += 1

        t = threading.Thread(target=increment, args=(instance,), daemon=True)
        t.start()

        try:
            # thread shouldn't be able to increment
            # a component instance can lock the entire array
            with instance:
                first_peek = (instance.x, instance.y)
                for _ in range(100):
                    assert first_peek == (instance.x, instance.y)

            # thread should do some increments
            time.sleep(0.001)

            # thread should be locked out again
            # the component type can lock the entire array
            with ExampleComponent1:
                second_peek = (instance.x, instance.y)
                assert second_peek != first_peek
                for _ in range(100):
                    assert second_peek == (instance.x, instance.y)
        finally:
            running = False
            t.join(1)


class ExampleEntity1(ecs.Entity):
    comp1: ExampleComponent1
    comp2: ExampleComponent2


class ExampleEntity2(ecs.Entity):
    comp1: ExampleComponent1
    comp2: ExampleComponent2


class TestEntity:
    def test_entities_share_an_id_pool(self):
        entity0 = ExampleEntity1(
            comp1=ExampleComponent1(1, 2), comp2=ExampleComponent2(3, 4)
        )
        entity1 = ExampleEntity2(
            comp1=ExampleComponent1(3, 4), comp2=ExampleComponent2(5, 6)
        )
        entity2 = ExampleEntity1(
            comp1=ExampleComponent1(5, 6), comp2=ExampleComponent2(7, 8)
        )

        assert entity0.id == 0
        assert entity1.id == 1
        assert entity2.id == 2

    def test_entity_found_with_only_id(self):
        entity0 = ExampleEntity1(
            comp1=ExampleComponent1(1, 2), comp2=ExampleComponent2(3, 4)
        )
        entity1 = ExampleEntity2(
            comp1=ExampleComponent1(3, 4), comp2=ExampleComponent2(5, 6)
        )
        entity2 = ExampleEntity1(
            comp1=ExampleComponent1(5, 6), comp2=ExampleComponent2(7, 8)
        )

        get0 = ecs.Entity.get(0)
        assert isinstance(get0, ExampleEntity1)
        assert get0 == entity0

        get1 = ecs.Entity.get(1)
        assert isinstance(get1, ExampleEntity2)
        assert get1 == entity1

    def test_binding_components_with_entity(self):
        entity = ExampleEntity1(
            comp1=ExampleComponent1(1, 2), comp2=ExampleComponent2(3, 4)
        )

        assert entity.comp1.x == 1.0
        assert entity.comp1.y == 2.0
        assert entity.comp2.z == 3.0
        assert entity.comp2.w == 4.0

        c1_id = entity.comp1.id
        c2_id = entity.comp2.id
        assert entity.comp1 == ExampleComponent1.get(c1_id)
        assert entity.comp2 == ExampleComponent2.get(c2_id)

    def test_equality_comparison(self):
        entity1 = ExampleEntity1(
            comp1=ExampleComponent1(1, 2), comp2=ExampleComponent2(3, 4)
        )
        entity2 = ExampleEntity1(
            comp1=ExampleComponent1(1, 2), comp2=ExampleComponent2(3, 4)
        )
        entity3 = ExampleEntity1(
            comp1=ExampleComponent1(3, 4), comp2=ExampleComponent2(1, 2)
        )

        assert entity1 == entity2
        assert entity3 != entity1

    def test_getting_a_previously_created_entity(self):
        entity = ExampleEntity1(
            comp1=ExampleComponent1(1, 2), comp2=ExampleComponent2(3, 4)
        )

        assert ExampleEntity1.get(entity.id) == entity

    def test_destroying_an_entity(self):
        entity = ExampleEntity1(
            comp1=ExampleComponent1(1, 2), comp2=ExampleComponent2(3, 4)
        )
        id = entity.id
        id1 = entity.comp1.id
        id2 = entity.comp2.id

        ExampleEntity1.destroy(entity.id)

        assert ExampleEntity1.get(id) is None
        assert ExampleComponent1.get(id1) is None
        assert ExampleComponent2.get(id2) is None
        assert entity.comp1 is None
        assert entity.comp2 is None

    def test_destroy_entity_with_only_id(self):
        entity = ExampleEntity1(
            comp1=ExampleComponent1(1, 2), comp2=ExampleComponent2(3, 4)
        )

        ecs.Entity.destroy(entity.id)

        assert ExampleEntity1.get(entity.id) is None

    def test_safe_to_destroy_entity_that_does_not_exist(self):
        ecs.Entity.destroy(1_000_000)
        ExampleEntity1.destroy(1_000_000)
        assert True  # should not exit early from exception

    def test_access_to_masked_component_arrays(self):
        ExampleComponent1(123, 123)
        ExampleComponent1(1234, 1234)

        entity1 = ExampleEntity1(
            comp1=ExampleComponent1(1, 2), comp2=ExampleComponent2(101, 102)
        )
        entity2 = ExampleEntity1(
            comp1=ExampleComponent1(3, 4), comp2=ExampleComponent2(103, 104)
        )

        assert np.all(ExampleEntity1.comp1.x == np.array([1, 3], float))
        assert np.all(ExampleEntity1.comp1.y == np.array([2, 4], float))
        assert np.all(ExampleEntity1.comp2.z == np.array([101, 103], float))
        assert np.all(ExampleEntity1.comp2.w == np.array([102, 104], float))

    def test_internal_length(self):
        ExampleEntity1.clear()
        
        assert ExampleEntity1.existing == 0
        assert len(ExampleEntity1) > 0

    def test_auto_allocation(self):
        c1 = ExampleComponent1(1, 2)
        c2 = ExampleComponent2(1, 2)
        length = len(ExampleEntity1)
        
        for _ in range(length + 1):
            ExampleEntity1(c1, c2)

        assert len(ExampleEntity1) > length
    
    def test_auto_deallocation(self):
        c1 = ExampleComponent1(1, 2)
        c2 = ExampleComponent2(1, 2)
        length = len(ExampleEntity1)
        instances = []

        for _ in range(length + 1):
            instances.append(ExampleEntity1(c1, c2))

        grown_length = len(ExampleEntity1)
        assert grown_length > length

        for inst in instances:
            ecs.Entity.destroy(inst)

        assert len(ExampleEntity1) < grown_length
    

class GlTypeComponent(ecs.Component):
    v3: gl.vec3
    m4: gl.mat4


def test_component_arrays_with_multi_dimensional_dtype():
    m4_1 = np.array(
        ((1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6), (4, 5, 6, 7)), gl.mat4
    )
    m4_2 = m4_1 + 1

    instance1 = GlTypeComponent((1, 2, 3), m4_1)
    instance2 = GlTypeComponent((2, 3, 4), m4_2)

    assert_approx(instance1.v3, (1, 2, 3))
    assert_approx(instance2.v3, (2, 3, 4))

    assert_approx(instance1.m4, m4_1)
    assert_approx(instance2.m4, m4_2)
