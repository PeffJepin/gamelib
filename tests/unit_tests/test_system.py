import numpy as np
import pytest

from src.gamelib.system import System, ArrayAttribute, PublicAttribute


class TestSystem:
    def test_subclasses_each_get_their_own_Component_attribute(self):
        assert self.System1.Component is not self.System2.Component

    def test_subclass_attributes_are_unique_from_base_system_attributes(self):
        system_component_subclasses = System.Component.__subclasses__()
        assert self.System1.Component not in system_component_subclasses
        assert self.System2.Component not in system_component_subclasses

    def test_can_find_related_public_attributes(self):
        all_public_attrs = [
            self.Comp1.__dict__["attr1"],
            self.Comp1.__dict__["attr2"],
            self.Comp2.__dict__["attr1"],
        ]
        for attr in self.System1.public_attributes:
            assert attr in all_public_attrs

    def test_can_find_related_array_attributes(self):
        all_array_attrs = [
            self.Comp1.__dict__["attr3"],
            self.Comp2.__dict__["attr2"],
            self.Comp2.__dict__["attr3"],
        ]
        for attr in self.System1.array_attributes:
            assert attr in all_array_attrs

    class System1(System):
        pass

    class System2(System):
        pass

    class Comp1(System1.Component):
        attr1 = PublicAttribute(int)
        attr2 = PublicAttribute(float)
        attr3 = ArrayAttribute(int)

    class Comp2(System1.Component):
        attr1 = PublicAttribute(int)
        attr2 = ArrayAttribute(str)
        attr3 = ArrayAttribute(float)


class TestArrayAttribute:
    def test_types_receive_instance_of_the_array(self):
        assert self.ExampleComponent.attr.dtype == np.uint8
        assert (100,) == self.ExampleComponent.attr.shape

    def test_get_on_instance_indexes_by_entity_id(self):
        self.ExampleComponent.attr[10] = 150
        obj = self.ExampleComponent(10)

        assert 150 == obj.attr

    def test_set_on_instance_indexes_by_entity_id(self):
        obj = self.ExampleComponent(14)
        obj.attr = 14

        assert 14 == self.ExampleComponent.attr[14]

    def test_must_be_used_on_a_component(self):
        with pytest.raises(RuntimeError):

            class NotAComponent:
                attr = ArrayAttribute(int, 10)

    def test_index_error_on_out_of_bounds_entity_id(self):
        obj = self.ExampleComponent(1999)

        with pytest.raises(IndexError):
            obj.attr = 100

    def test_reallocation_to_a_new_size(self):
        assert 25 != len(self.ExampleComponent.attr)
        self.ExampleSystem.MAX_ENTITIES = 25
        self.ExampleComponent.__dict__["attr"].reallocate()

        assert 25 == len(self.ExampleComponent.attr)

    class ExampleSystem(System):
        MAX_ENTITIES = 100

    class ExampleComponent(ExampleSystem.Component):
        attr = ArrayAttribute(np.uint8)


class TestPublicAttribute:
    def test_public_attribute_does_not_work_before_allocation(
        self, class_with_public_attr
    ):
        with pytest.raises(FileNotFoundError):
            attr = class_with_public_attr.attr

    def test_access_can_be_made_after_allocation(self, class_with_public_attr):
        actual_public_attr = class_with_public_attr.__dict__["attr"]
        actual_public_attr.allocate_shm()

        try:
            assert all(class_with_public_attr.attr[:] == 0)
        finally:
            actual_public_attr.close_shm()

    def test_cannot_be_accessed_after_closed(self, class_with_public_attr):
        actual_public_attr = class_with_public_attr.__dict__["attr"]
        actual_public_attr.allocate_shm()
        actual_public_attr.close_shm()

        with pytest.raises(FileNotFoundError):
            attr = class_with_public_attr.attr

    def test_changes_not_reflected_until_update(self, class_with_public_attr):
        actual_public_attr = class_with_public_attr.__dict__["attr"]
        actual_public_attr.allocate_shm()

        class_with_public_attr.attr[:] = 10

        try:
            assert all(class_with_public_attr.attr[:] == 0)
            actual_public_attr.update()
            assert all(class_with_public_attr.attr[:] == 10)
        finally:
            actual_public_attr.close_shm()

    def test_array_size_dictated_by_System_MAX_ENTITIES(self, class_with_public_attr):
        System.MAX_ENTITIES = 16
        actual_public_attr = class_with_public_attr.__dict__["attr"]
        actual_public_attr.allocate_shm()

        try:
            assert len(class_with_public_attr.attr) == 16
        finally:
            actual_public_attr.close_shm()

    def test_indexed_by_object_entity_id(self, class_with_public_attr):
        actual_public_attr = class_with_public_attr.__dict__["attr"]
        actual_public_attr.allocate_shm()
        inst = class_with_public_attr(5)

        inst.attr = 10

        try:
            assert class_with_public_attr.attr[5] == 0
            actual_public_attr.update()
            assert class_with_public_attr.attr[5] == 10
        finally:
            actual_public_attr.close_shm()

    @pytest.fixture
    def class_with_public_attr(self):
        class ExampleSystem(System):
            pass

        class MyObject(ExampleSystem.Component):
            attr = PublicAttribute(np.uint8)

        return MyObject
