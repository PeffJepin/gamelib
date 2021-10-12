import numpy as np
import pytest

from src.gamelib import Config
from src.gamelib.component import BaseComponent, ArrayAttribute, PublicAttribute


class TestArrayAttribute:
    def test_types_receive_instance_of_the_array(self):
        assert self.ExampleComponent.attr.dtype == np.uint8
        assert isinstance(self.ExampleComponent.attr[:], np.ndarray)

    def test_get_on_instance_indexes_by_entity_id(self):
        self.ExampleComponent.attr[10] = 150
        obj = self.ExampleComponent(10)

        assert 150 == obj.attr

    def test_set_on_instance_indexes_by_entity_id(self):
        obj = self.ExampleComponent(14)
        obj.attr = 14

        assert self.ExampleComponent.attr[14] == 14

    def test_must_be_used_on_a_component(self):
        with pytest.raises(RuntimeError):

            class NotAComponent:
                attr = ArrayAttribute(int)

    def test_index_error_on_out_of_bounds_entity_id(self):
        obj = self.ExampleComponent(1_000_000)

        with pytest.raises(IndexError):
            obj.attr = 100

    def test_reallocation_to_a_new_size(self):
        assert 25 != len(self.ExampleComponent.attr)
        Config.MAX_ENTITIES = 25
        vars(self.ExampleComponent)["attr"].reallocate()

        assert 25 == len(self.ExampleComponent.attr)

    def test_array_masks_components_that_dont_exist(self):
        for i in range(10):
            comp = self.ExampleComponent(i)
            comp.attr = 1
        assert all(1 == self.ExampleComponent.attr[:10])
        assert not any(self.ExampleComponent.attr[10:])

    class ExampleComponent(BaseComponent):
        attr = ArrayAttribute(np.uint8)

    @pytest.fixture(autouse=True)
    def reallocate_attribute(self):
        attr = vars(self.ExampleComponent)["attr"]
        attr.reallocate()


class TestComponent:
    def test_type_can_get_an_iterable_of_associated_array_attributes(self):
        expected = [
            vars(self.ExampleComponent)["arr1"],
            vars(self.ExampleComponent)["arr2"],
        ]
        assert all(
            [
                value in expected
                for value in self.ExampleComponent.get_array_attributes()
            ]
        )

    def test_type_can_get_an_iterable_of_associated_public_attributes(self):
        expected = [
            vars(self.ExampleComponent)["pub1"],
            vars(self.ExampleComponent)["pub2"],
        ]
        assert all(
            [
                value in expected
                for value in self.ExampleComponent.get_public_attributes()
            ]
        )

    def test_created_instances_can_be_found_by_the_type(self):
        instance = self.ExampleComponent(0)

        assert self.ExampleComponent[0] is instance

    def test_destroyed_instances_cannot_be_found_by_the_type(self):
        instance = self.ExampleComponent(0)
        instance.destroy()

        assert self.ExampleComponent[0] is None

    def test_can_destroy_all_instances_from_type(self):
        for i in range(10):
            self.ExampleComponent(i)

        self.ExampleComponent.destroy_all()
        retrieved_instances = [self.ExampleComponent[i] for i in range(10)]

        assert not any(retrieved_instances)

    class ExampleComponent(BaseComponent):
        arr1 = ArrayAttribute(int)
        arr2 = ArrayAttribute(float)
        pub1 = PublicAttribute(float)
        pub2 = PublicAttribute(float)

    @pytest.fixture(autouse=True)
    def clear_instances(self):
        self.ExampleComponent.destroy_all()
