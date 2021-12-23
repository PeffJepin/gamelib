import pytest
import threading
import time

from gamelib.component import Component


class ExampleComponent(Component):
    x: float
    y: float


class TestComponent:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        ExampleComponent.clear()

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

        assert isinstance(component.id, int)
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

    def test_allocating_space_manually(self):
        max_length = len(ExampleComponent.get_raw_arrays())
        ExampleComponent.reallocate(max_length + 5)
        assert len(ExampleComponent.get_raw_arrays()) == max_length + 5

    def test_allocating_space_automatically(self):
        max_length = len(ExampleComponent.get_raw_arrays())
        current_length = ExampleComponent.length

        for i in range(max_length - current_length + 1):
            ExampleComponent(i, i)

        assert len(ExampleComponent.get_raw_arrays()) > max_length

    def test_freeing_space_automatically(self):
        starting_length = len(ExampleComponent.get_raw_arrays()) * 2 

        for i in range(starting_length):
            ExampleComponent(i, i)

        second_length = len(ExampleComponent.get_raw_arrays()) 
        for i in reversed(range(starting_length - 1)):
            ExampleComponent.destroy(i)
        
        assert len(ExampleComponent.get_raw_arrays()) < second_length

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

