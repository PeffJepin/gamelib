import numpy as np

import pytest
import time

from gamelib.ecs import base
from gamelib import gl

from ..conftest import assert_approx


@pytest.fixture(autouse=True)
def cleanup():
    base.Entity.clear()


class TestIdGenerator:
    def test_ids_are_given_sequentially(self):
        gen = base.IdGenerator()

        assert next(gen) == 0
        assert next(gen) == 1
        assert next(gen) == 2

    def test_ids_can_be_recycled(self):
        gen = base.IdGenerator()

        for _ in range(2):
            next(gen)

        gen.recycle(0)

        assert next(gen) == 0

    def test_always_lowest_possible_id(self):
        gen = base.IdGenerator()

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
        gen = base.IdGenerator()
        for _ in range(5):
            next(gen)

        assert gen.largest_active == 4

        gen.recycle(1)
        gen.recycle(3)
        gen.recycle(4)
        assert gen.largest_active == 2

    def test_set_state_stops_recycling_irrelevant_ids(self):
        gen = base.IdGenerator()
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
        gen = base.IdGenerator()
        for _ in range(5):
            next(gen)

        gen.recycle(1)
        gen.set_state(3)

        # state is 3, but there are still recyclables to get through
        assert next(gen) == 1
        assert next(gen) == 3
        assert next(gen) == 4


class Component1(base.Component):
    x: float
    y: float


class Component2(base.Component):
    z: float
    w: float


class TestComponent:
    def test_creating_a_component(self):
        component = Component1.create(123, 124)

        assert component.x == 123
        assert component.y == 124
        assert 123 in Component1.x
        assert 124 in Component1.y
        actual = Component1.view_raw_arrays()[component.id]
        assert actual["x"], actual["y"] == (123, 124)

    def test_getting_a_component(self):
        component = Component1.create(1.23, 4.56)

        assert Component1.get(component.id) == component

    def test_destroying_a_component(self):
        component1 = Component1.create(0.1234, 0.1234)
        component2 = Component1.create(321, 321)

        assert component1 == Component1.get(component1.id)
        assert component2 == Component1.get(component2.id)

        component1.destroy(component1.id)
        Component1.destroy(component2.id)

        assert Component1.get(component1.id) is None
        assert Component1.get(component2.id) is None

    def test_safe_to_destroy_nonexisting_component(self):
        Component1.destroy(1_000)
        assert True  # above should not error

    def test_length_of_component(self):
        length = len(Component1)
        Component1.create(0, 0)

        assert len(Component1) == length + 1

    def test_equality_comparison(self):
        assert Component1.create(123.0, 123.0) == (123.0, 123.0)
        assert Component1.create(123.0, 123.0) != (123.1, 123.1)

    def test_getting_an_existing_component_by_id(self):
        id = Component1.create(519, 542).id

        looked_up = Component1.get(id)
        assert looked_up == (519, 542)

        looked_up.x = 1024
        looked_up.y = 1025

        assert Component1.get(id) == (1024, 1025)

    def test_id_when_created(self):
        component = Component1.create(1234, 1234)

        assert component.id is not None
        assert Component1.get(component.id) == (1234, 1234)
        assert Component1.create(1234, 1234).id == component.id + 1

    def test_recycling_an_id(self):
        component1 = Component1.create(0, 0)
        c1_id = component1.id
        Component1.destroy(component1.id)
        component2 = Component1.create(1, 1)

        assert component2.id == c1_id

    def test_masked_after_being_destroyed(self):
        component1 = Component1.create(1, 1)
        component2 = Component1.create(2, 2)
        component3 = Component1.create(3, 3)

        Component1.destroy(component2.id)

        assert Component1.get(component1.id) == component1
        assert Component1.get(component3.id) == component3
        assert len(Component1) == 2

    def test_mutating_data_by_an_instance(self):
        component = Component1.create(1, 2)
        assert Component1.get(component.id) == (1, 2)

        component.x = 132
        component.y = 1234

        assert (132, 1234) == Component1.get(component.id)

    def test_mutating_data_by_class_array(self):
        component1 = Component1.create(0, 0)
        component2 = Component1.create(1, 1)

        Component1.x += 100

        assert (100, 0) == component1
        assert (101, 1) == component2

    def test_internal_length(self):
        assert Component1.internal_length == len(Component1.view_raw_arrays())

    def test_automatic_growth(self):
        starting_length = Component1.internal_length
        for _ in range(starting_length + 1):
            Component1.create(1, 2)

        assert Component1.internal_length > starting_length

    def test_automatic_shrink(self):
        starting_length = Component1.internal_length
        ids = []
        for _ in range(starting_length + 1):
            ids.append(Component1.create(1, 2).id)

        grown_length = Component1.internal_length
        for i in ids:
            Component1.destroy(i)

        assert Component1.internal_length < grown_length

    def test_data_integrity_through_growing_and_shrinking(self):
        starting_length = Component1.internal_length
        grow_to = starting_length * 3

        first = start = last = None
        fi, si, li = 0, starting_length - 1, grow_to - 1
        ids = []
        for i in range(grow_to):
            comp = Component1.create(i, 111_111 * i)
            if i == fi:
                first = comp
            elif i == si:
                start = comp
            elif i == li:
                last = comp
            else:
                ids.append(comp.id)
        for i in ids:
            Component1.destroy(i)

        for c, i in zip((first, start, last), (fi, si, li)):
            expected = (i, 111_111 * i)
            assert c.values == expected
            assert Component1.get(c.id).values == expected

    def test_component_inheritance(self):
        class C(Component1):
            pass

        comp = C(x=1, y=2)
        assert comp.x == 1
        assert comp.y == 2

    def test_base_component_subclasses(self):
        class C(Component1):
            pass

        all_subclasses = base.Component.get_subclasses()

        assert C in all_subclasses
        assert Component1 in all_subclasses
        assert Component2 in all_subclasses

    def test_getting_subclasses_of_a_subclass(self):
        class C1(Component1):
            pass

        class C2(Component2):
            pass

        class C3(C1):
            pass

        subclasses1 = Component1.get_subclasses()
        subclasses2 = Component2.get_subclasses()

        assert C1 in subclasses1
        assert C3 in subclasses1
        assert C1 not in subclasses2
        assert C3 not in subclasses2
        assert C2 in subclasses2
        assert C2 not in subclasses1

    def test_ids_back_to_0_after_clear(self):
        for _ in range(2):
            Component1.create(0, 0)
        Component1.clear()

        assert Component1.create(0, 0).id == 0

    def test_clearing_a_component(self):
        for _ in range(2):
            Component1.create(0, 0)
        assert len(Component1) == 2

        Component1.clear()

        assert len(Component1) == 0

    def test_clearing_a_parent_class(self):
        for _ in range(2):
            Component1.create(0, 0)
            Component2.create(0, 0)
        assert len(Component1) == 2
        assert len(Component2) == 2

        base.Component.clear()

        assert len(Component1) == 0
        assert len(Component2) == 0


class Entity1(base.Entity):
    comp1: Component1
    comp2: Component2


class Entity2(base.Entity):
    comp1: Component1
    comp2: Component2


class TestEntity:
    def test_entities_share_an_id_pool(self):
        entity0 = Entity1.create(
            comp1=Component1.create(1, 2), comp2=Component2.create(3, 4)
        )
        entity1 = Entity2.create(
            comp1=Component1.create(3, 4), comp2=Component2.create(5, 6)
        )
        entity2 = Entity1.create(
            comp1=Component1.create(5, 6), comp2=Component2.create(7, 8)
        )

        assert entity0.id == 0
        assert entity1.id == 1
        assert entity2.id == 2

    def test_entity_found_with_only_id(self):
        entity0 = Entity1.create(
            comp1=Component1.create(1, 2), comp2=Component2.create(3, 4)
        )
        entity1 = Entity2.create(
            comp1=Component1.create(3, 4), comp2=Component2.create(5, 6)
        )
        entity2 = Entity1.create(
            comp1=Component1.create(5, 6), comp2=Component2.create(7, 8)
        )

        get0 = base.Entity.get(0)
        assert isinstance(get0, Entity1)
        assert get0 == entity0

        get1 = base.Entity.get(1)
        assert isinstance(get1, Entity2)
        assert get1 == entity1

    def test_binding_components_with_entity(self):
        entity = Entity1.create(
            comp1=Component1.create(1, 2), comp2=Component2.create(3, 4)
        )

        assert entity.comp1.x == 1.0
        assert entity.comp1.y == 2.0
        assert entity.comp2.z == 3.0
        assert entity.comp2.w == 4.0

        c1_id = entity.comp1.id
        c2_id = entity.comp2.id
        assert entity.comp1 == Component1.get(c1_id)
        assert entity.comp2 == Component2.get(c2_id)

    def test_equality_comparison(self):
        entity1 = Entity1.create(
            comp1=Component1.create(1, 2), comp2=Component2.create(3, 4)
        )
        entity2 = Entity1.create(
            comp1=Component1.create(1, 2), comp2=Component2.create(3, 4)
        )
        entity3 = Entity1.create(
            comp1=Component1.create(3, 4), comp2=Component2.create(1, 2)
        )

        assert entity1 == Entity1.get(entity1.id)
        assert entity2 != entity1
        assert entity3 != entity1

    def test_getting_a_previously_created_entity(self):
        entity = Entity1.create(
            comp1=Component1.create(1, 2), comp2=Component2.create(3, 4)
        )

        assert Entity1.get(entity.id) == entity

    def test_destroying_an_entity(self):
        entity = Entity1.create(
            comp1=Component1.create(1, 2), comp2=Component2.create(3, 4)
        )
        id = entity.id
        id1 = entity.comp1.id
        id2 = entity.comp2.id

        Entity1.destroy(entity.id)

        assert Entity1.get(id) is None
        assert Component1.get(id1) is None
        assert Component2.get(id2) is None
        assert entity.comp1 is None
        assert entity.comp2 is None

    def test_destroy_entity_with_only_id(self):
        entity = Entity1.create(
            comp1=Component1.create(1, 2), comp2=Component2.create(3, 4)
        )

        base.Entity.destroy(entity.id)

        assert Entity1.get(entity.id) is None

    def test_safe_to_destroy_entity_that_does_not_exist(self):
        base.Entity.destroy(1_000_000)
        Entity1.destroy(1_000_000)
        assert True  # should not exit early from exception

    def test_access_to_masked_component_arrays(self):
        entity1 = Entity1.create(
            comp1=Component1.create(1, 2), comp2=Component2.create(101, 102)
        )
        entity2 = Entity1.create(
            comp1=Component1.create(3, 4), comp2=Component2.create(103, 104)
        )
        entity3 = Entity2.create(
            comp1=Component1.create(5, 6), comp2=Component2.create(105, 106)
        )

        assert np.all(Entity1.comp1.x == np.array([1, 3], float))
        assert np.all(Entity1.comp1.y == np.array([2, 4], float))
        assert np.all(Entity1.comp2.z == np.array([101, 103], float))
        assert np.all(Entity1.comp2.w == np.array([102, 104], float))

    def test_modifying_components_through_entity_mask(self):
        entity1 = Entity1.create(
            comp1=Component1.create(1, 2), comp2=Component2.create(101, 102)
        )
        entity2 = Entity1.create(
            comp1=Component1.create(3, 4), comp2=Component2.create(103, 104)
        )
        entity3 = Entity2.create(
            comp1=Component1.create(5, 6), comp2=Component2.create(105, 106)
        )

        Entity1.comp1.x += 1_000
        assert entity1.comp1.x == 1_001
        assert entity2.comp1.x == 1_003
        assert entity3.comp1.x == 5

    def test_mask_on_mask_operation(self):
        entity1 = Entity1.create(
            comp1=Component1.create(1, 2), comp2=Component2.create(101, 102)
        )
        entity2 = Entity1.create(
            comp1=Component1.create(3, 4), comp2=Component2.create(103, 104)
        )
        entity3 = Entity2.create(
            comp1=Component1.create(5, 6), comp2=Component2.create(105, 106)
        )

        Entity1.comp1.x = Entity1.comp1.x + Entity1.comp2.z
        assert entity1.comp1.x == 102
        assert entity2.comp1.x == 106
        assert entity3.comp1.x == 5
        Entity1.comp1.y += Entity1.comp2.w
        assert entity1.comp1.y == 104
        assert entity2.comp1.y == 108
        assert entity3.comp1.y == 6

    def test_internal_length(self):
        Entity1.clear()

        assert len(Entity1) == 0
        assert Entity1.internal_length > 0

    def test_auto_allocation(self):
        c1 = Component1.create(1, 2)
        c2 = Component2.create(1, 2)
        length = len(Entity1)

        for _ in range(length + 1):
            Entity1.create(c1, c2)

        assert len(Entity1) > length

    def test_auto_deallocation(self):
        c1 = Component1.create(1, 2)
        c2 = Component2.create(1, 2)
        length = len(Entity1)
        instances = []

        for _ in range(length + 1):
            instances.append(Entity1.create(c1, c2))

        grown_length = len(Entity1)
        assert grown_length > length

        for inst in instances:
            base.Entity.destroy(inst)

        assert len(Entity1) < grown_length

    def test_clearing_a_subclass(self):
        class NestedClass(Entity1):
            pass

        entities1, entities2, entities3 = [], [], []
        components1, components2, components3 = [], [], []
        for i in range(3):
            comp1 = Component1.create(i, i)
            comp2 = Component2.create(i, i)
            components1.extend((comp1, comp2))
            entities1.append(Entity1.create(comp1, comp2))

            comp1 = Component1.create(i, i)
            comp2 = Component2.create(i, i)
            components2.extend((comp1, comp2))
            entities2.append(Entity2.create(comp1, comp2))

            comp1 = Component1.create(i, i)
            comp2 = Component2.create(i, i)
            components3.extend((comp1, comp2))
            entities3.append(NestedClass.create(comp1, comp2))

        Entity1.clear()

        for e in entities2:
            assert base.Entity.get(e.id) == e
        for c in components2:
            assert type(c).get(c.id) == c

        for e in entities1 + entities3:
            assert base.Entity.get(e.id) is None
            assert e.comp1 is None
            assert e.comp2 is None
        for c in components1 + components3:
            assert type(c).get(c.id) is None
            assert c.values == (None, None)

    def test_clearing_the_base_class(self):
        class NestedClass(Entity1):
            pass

        entities = []
        components = []
        for i in range(3):
            comp1 = Component1.create(i, i)
            comp2 = Component2.create(100 * i, 100 * i)
            components.extend((comp1, comp2))
            entities.append(Entity1.create(comp1, comp2))

            comp1 = Component1.create(i, i)
            comp2 = Component2.create(100 * i, 100 * i)
            components.extend((comp1, comp2))
            entities.append(Entity2.create(comp1, comp2))

            comp1 = Component1.create(i, i)
            comp2 = Component2.create(100 * i, 100 * i)
            components.extend((comp1, comp2))
            entities.append(NestedClass.create(comp1, comp2))

        base.Entity.clear()

        for e in entities:
            assert base.Entity.get(e.id) is None
            assert len(Entity1) == 0
            assert len(Entity2) == 0
            assert len(NestedClass) == 0
            assert len(base.Entity) == 0
        for c in components:
            assert type(c).get(c.id) is None
            assert len(Component1) == 0
            assert len(Component2) == 0

    def test_entity_inheritance(self):
        class E(Entity1):
            pass

        c1 = Component1.create(1, 2)
        c2 = Component2.create(3, 4)
        entity = E(comp1=c1, comp2=c2)
        assert entity.comp1 == c1
        assert entity.comp2 == c2

    def test_iterating_entity_subclasses(self):
        class E(Entity1):
            pass

        all_subclasses = base.Entity.get_subclasses()

        assert Entity1 in all_subclasses
        assert Entity2 in all_subclasses
        assert E in all_subclasses

        e1_subclasses = Entity1.get_subclasses()
        assert E in e1_subclasses
        assert Entity2 not in e1_subclasses

    def test_iterating_over_only_entities_with_given_components(self):
        class E1(base.Entity):
            c1: Component1

        class E2(base.Entity):
            c2: Component2

        result = base.Entity.get_subclasses(
            components=(Component1, Component2)
        )
        assert E1 not in result
        assert E2 not in result
        assert Entity1 in result
        assert Entity2 in result

        result = base.Entity.get_subclasses(components=(Component1))
        assert E1 in result
        assert E2 not in result
        assert Entity1 in result
        assert Entity2 in result

    def test_getting_a_component_by_type(self):
        class C(Component2):
            pass

        c1 = Component1.create(1, 2)
        c2 = Component2.create(3, 4)
        entity = Entity1.create(c1, c2)

        assert entity.get_component(Component1) == c1
        assert entity.get_component(C) is None

    def testing_if_an_entity_type_has_a_component_type_field(self):
        class C(Component2):
            pass

        assert Entity1.has_field(Component1)
        assert Entity1.has_field(Component2)
        assert not Entity1.has_field(C)

    def test_getting_access_to_component_entity_mask(self):
        Entity1.create(Component1.create(1, 2), Component2.create(3, 4))
        Entity1.create(Component1.create(5, 6), Component2.create(7, 8))

        assert np.all(Entity1.get_mask(Component1).x == (1, 5))
        assert np.all(Entity1.get_mask(Component1).y == (2, 6))

    def test_proxy_to_masked_component_field(self):
        callable_proxy = Entity1.comp1.proxy("x")
        assert len(callable_proxy()) == 0

        Entity1.create(Component1.create(1, 2), Component2.create(3, 4))
        Entity1.create(Component1.create(5, 6), Component2.create(7, 8))
        Entity2.create(Component1.create(9, 9), Component2.create(9, 9))

        assert np.all(callable_proxy() == Entity1.comp1.x)


class GlTypeComponent(base.Component):
    v3: gl.vec3
    m4: gl.mat4


def test_component_arrays_with_multi_dimensional_dtype():
    m4_1 = np.array(
        ((1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6), (4, 5, 6, 7)), gl.mat4
    )
    m4_2 = m4_1 + 1

    instance1 = GlTypeComponent.create((1, 2, 3), m4_1)
    instance2 = GlTypeComponent.create((2, 3, 4), m4_2)

    assert_approx(instance1.v3, (1, 2, 3))
    assert_approx(instance2.v3, (2, 3, 4))

    assert_approx(instance1.m4, m4_1)
    assert_approx(instance2.m4, m4_2)
