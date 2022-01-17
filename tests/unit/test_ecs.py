import numpy as np

import pytest
import threading
import time

from gamelib import ecs


@pytest.fixture(autouse=True)
def cleanup():
    for cls in (ExampleEntity, ExampleComponent, ExampleComponent2):
        cls.clear()


class TestDynamicArrayManager:
    def test_init(self):
        arrays = ecs.DynamicArrayManager(field1=float, field2=int)

        assert arrays.field1.dtype == float
        assert arrays.field2.dtype == int
        assert all(len(a) == 0 for a in arrays)

    def test_adding_entries(self):
        arrays = ecs.DynamicArrayManager(field1=float, field2=int)

        entry = arrays.new_entry(1.0, 2)
        entry2 = arrays.new_entry(2.0, 3)

        assert arrays[entry.id] == (1.0, 2)
        assert len(arrays.field1) == 2
        assert len(arrays.field2) == 2

    def test_new_entry_with_kwargs(self):
        arrays = ecs.DynamicArrayManager(field1=float, field2=int)

        entry = arrays.new_entry(field1=123, field2=321)

        assert arrays[entry.id] == (123.0, 321)

    def test_new_entry_without_values(self):
        arrays = ecs.DynamicArrayManager(field1=float, field2=int)

        entry = arrays.new_entry()
        entry.field1 = 1
        entry.field2 = 2

        assert arrays[entry.id] == entry

    def test_length_of_object_is_internal_length(self):
        arrays = ecs.DynamicArrayManager(field1=float, field2=int)

        assert len(arrays) > 0

    def test_length_of_fields_are_masked(self):
        arrays = ecs.DynamicArrayManager(field1=float, field2=int)

        assert len(arrays) > 0
        assert len(arrays.field1) == 0
        assert len(arrays.field2) == 0

    def test_adding_entries_forcing_growth(self):
        arrays = ecs.DynamicArrayManager(field1=float, field2=int)
        starting_length = len(arrays)

        for i in range(starting_length + 10):
            arrays.new_entry(i, i)

        assert len(arrays) > starting_length + 10

    def test_out_of_bounds_returns_none(self):
        arrays = ecs.DynamicArrayManager(field1=float, field2=int)
        assert arrays[1_000_000] is None

    def test_removing_entries(self):
        arrays = ecs.DynamicArrayManager(field1=float, field2=int)

        id = arrays.new_entry(1.0, 1).id
        assert arrays[id] is not None
        del arrays[id]

        assert arrays[id] is None

    def test_removing_entries_forcing_shrinking(self):
        arrays = ecs.DynamicArrayManager(field1=float, field2=int)

        starting_length = len(arrays)
        i = 0
        entries = []
        while len(arrays) <= starting_length:
            entries.append(arrays.new_entry(i, i))
            i += 1
            if i >= 50:
                assert False

        grown_length = len(arrays)
        for entry in entries:
            del arrays[entry.id]

        assert len(arrays) < grown_length

    def test_ids_are_recycled(self):
        arrays = ecs.DynamicArrayManager(field1=float, field2=int)

        entry1 = arrays.new_entry()
        entry2 = arrays.new_entry()
        entry3 = arrays.new_entry()

        id1 = entry1.id
        del arrays[id1]
        entry4 = arrays.new_entry()
        entry5 = arrays.new_entry()

        assert entry4.id == id1
        assert entry5.id == entry3.id + 1

    def test_clear(self):
        arrays = ecs.DynamicArrayManager(field1=float, field2=int)

        entry1 = arrays.new_entry()
        entry2 = arrays.new_entry()
        entry3 = arrays.new_entry()
        arrays.clear()

        assert len(arrays.field1) == 0
        assert arrays.new_entry().id == 0

    def test_initial_index_preserves_identity_through_reallocation(self):
        arrays = ecs.DynamicArrayManager(field1=float, field2=int)

        entries = [arrays.new_entry(i, i) for i in range(50)]
        for entry in entries[:-1]:
            # print(f"{len(arrays)=}, {entry.id=}, {len(arrays.field1)=}")
            print(arrays.field1)
            print(arrays.get_index(49))
            del arrays[entry.id]

        # despite the array no longer being large enough for 50 entries,
        # the 50th index (49) hasn't been deleted so should still work.
        assert len(arrays) < 50
        assert arrays[entries[-1].id] == (49, 49)

    def test_field_array(self):
        arrays = ecs.DynamicArrayManager(field1=float, field2=int)

        entries = [arrays.new_entry(i, i) for i in range(3)]
        del arrays[entries[1].id]

        # note that the indices for the fields wont necessarily be
        # the same as the initial indices used. Only access through
        # the arrays object can preserve identity
        assert np.all(arrays.field1 == np.array([0, 2], float))
        assert np.all(arrays.field2 == np.array([0, 2], int))

    def test_mutating_the_internal_data(self):
        arrays = ecs.DynamicArrayManager(field1=float, field2=int)
        entry = arrays.new_entry(123.0, 124)
        assert entry.field1 == 123.0
        assert entry.field2 == 124

        entry.field1 = 321
        entry.field2 = 456

        assert arrays[entry.id] == (321, 456)


class ExampleComponent(ecs.Component):
    x: float
    y: float


class ExampleComponent2(ecs.Component):
    z: float
    w: float


class TestComponent:
    def test_creating_a_component(self):
        component = ExampleComponent(123, 124)

        assert component.x == 123
        assert component.y == 124
        assert 123 in ExampleComponent.x
        assert 124 in ExampleComponent.y
        actual = ExampleComponent.get_raw_arrays()[component.id]
        assert actual["x"], actual["y"] == (123, 124)

    def test_getting_a_component(self):
        component = ExampleComponent(1.23, 4.56)

        assert ExampleComponent.get(component.id) == component

    def test_destroying_a_component(self):
        component1 = ExampleComponent(0.1234, 0.1234)
        component2 = ExampleComponent(321, 321)

        assert component1 == ExampleComponent.get(component1.id)
        assert component2 == ExampleComponent.get(component2.id)

        component1.destroy(component1.id)
        ExampleComponent.destroy(component2.id)

        assert ExampleComponent.get(component1.id) is None
        assert ExampleComponent.get(component2.id) is None

    def test_length_of_component(self):
        length = ExampleComponent.length
        ExampleComponent(0, 0)

        assert ExampleComponent.length == length + 1

    def test_equality_comparison(self):
        assert ExampleComponent(123.0, 123.0) == (123.0, 123.0)
        assert ExampleComponent(123.0, 123.0) != (123.1, 123.1)

    def test_getting_an_existing_component_by_id(self):
        id = ExampleComponent(519, 542).id

        looked_up = ExampleComponent.get(id)
        assert looked_up == (519, 542)

        looked_up.x = 1024
        looked_up.y = 1025

        assert ExampleComponent.get(id) == (1024, 1025)

    def test_id_when_created(self):
        component = ExampleComponent(1234, 1234)

        assert component.id is not None
        assert ExampleComponent.get(component.id) == (1234, 1234)
        assert ExampleComponent(1234, 1234).id == component.id + 1

    def test_recycling_an_id(self):
        component1 = ExampleComponent(0, 0)
        c1_id = component1.id
        ExampleComponent.destroy(component1.id)
        component2 = ExampleComponent(1, 1)

        assert component2.id == c1_id

    def test_ids_back_to_0_after_clear(self):
        for _ in range(10):
            ExampleComponent(0, 0)
        ExampleComponent.clear()

        assert ExampleComponent(0, 0).id == 0

    def test_clearing_a_component(self):
        for _ in range(10):
            component = ExampleComponent(0, 0)

        assert ExampleComponent.length >= 10

        ExampleComponent.clear()

        assert ExampleComponent.length == 0
        assert component.x is None
        assert component.y is None

    def test_masked_after_being_destroyed(self):
        component1 = ExampleComponent(1, 1)
        component2 = ExampleComponent(2, 2)
        component3 = ExampleComponent(3, 3)

        ExampleComponent.destroy(component2.id)

        assert ExampleComponent.get(component1.id) == component1
        assert ExampleComponent.get(component3.id) == component3
        assert ExampleComponent.length == 2

    def test_mutating_data_by_an_instance(self):
        component = ExampleComponent(1, 2)
        assert ExampleComponent.get(component.id) == (1, 2)

        component.x = 132
        component.y = 1234

        assert (132, 1234) == ExampleComponent.get(component.id)

    def test_mutating_data_by_class_array(self):
        component1 = ExampleComponent(0, 0)
        component2 = ExampleComponent(1, 1)

        ExampleComponent.x += 100

        assert (100, 0) == component1
        assert (101, 1) == component2

    def test_locking_access_using_context_manager(self):
        instance = ExampleComponent(0, 0)
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
            with ExampleComponent:
                second_peek = (instance.x, instance.y)
                assert second_peek != first_peek
                for _ in range(100):
                    assert second_peek == (instance.x, instance.y)
        finally:
            running = False


class ExampleEntity(ecs.Entity):
    comp1: ExampleComponent
    comp2: ExampleComponent2


class TestEntity:
    def test_binding_components_with_entity(self):
        entity = ExampleEntity(
            comp1=ExampleComponent(1, 2), comp2=ExampleComponent2(3, 4)
        )

        assert entity.comp1.x == 1.0
        assert entity.comp1.y == 2.0
        assert entity.comp2.z == 3.0
        assert entity.comp2.w == 4.0

        c1_id = entity.comp1.id
        c2_id = entity.comp2.id
        assert entity.comp1 == ExampleComponent.get(c1_id)
        assert entity.comp2 == ExampleComponent2.get(c2_id)

    def test_equality_comparison(self):
        entity1 = ExampleEntity(
            comp1=ExampleComponent(1, 2), comp2=ExampleComponent2(3, 4)
        )
        entity2 = ExampleEntity(
            comp1=ExampleComponent(1, 2), comp2=ExampleComponent2(3, 4)
        )
        entity3 = ExampleEntity(
            comp1=ExampleComponent(3, 4), comp2=ExampleComponent2(1, 2)
        )

        assert entity1 == entity2
        assert entity3 != entity1

    def test_getting_a_previously_created_entity(self):
        entity = ExampleEntity(
            comp1=ExampleComponent(1, 2), comp2=ExampleComponent2(3, 4)
        )

        assert ExampleEntity.get(entity.id) == entity

    def test_destroying_an_entity(self):
        entity = ExampleEntity(
            comp1=ExampleComponent(1, 2), comp2=ExampleComponent2(3, 4)
        )
        id = entity.id
        id1 = entity.comp1.id
        id2 = entity.comp2.id

        ExampleEntity.destroy(entity.id)

        assert ExampleEntity.get(id) is None
        assert ExampleComponent.get(id1) is None
        assert ExampleComponent2.get(id2) is None
        assert entity.comp1 is None
        assert entity.comp2 is None

    def test_access_to_masked_component_arrays(self):
        ExampleComponent(123, 123)
        ExampleComponent(1234, 1234)

        entity1 = ExampleEntity(
            comp1=ExampleComponent(1, 2), comp2=ExampleComponent2(101, 102)
        )
        entity2 = ExampleEntity(
            comp1=ExampleComponent(3, 4), comp2=ExampleComponent2(103, 104)
        )

        assert np.all(ExampleEntity.comp1.x == np.array([1, 3], float))
        assert np.all(ExampleEntity.comp1.y == np.array([2, 4], float))
        assert np.all(ExampleEntity.comp2.z == np.array([101, 103], float))
        assert np.all(ExampleEntity.comp2.w == np.array([102, 104], float))
