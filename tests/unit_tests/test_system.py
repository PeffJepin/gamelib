import numpy as np
import pytest

from src.gamelib.system import System, ArrayAttribute, PublicAttribute


class TestSystem:
    def test_subclasses_each_get_their_own_Event_attribute(self):
        assert System1.Event is not System2.Event

    def test_subclasses_each_get_their_own_Component_attribute(self):
        assert System1.Component is not System2.Component

    def test_subclass_attributes_are_unique_from_base_system_attributes(self):
        system_event_subclasses = System.Event.__subclasses__()
        system_component_subclasses = System.Component.__subclasses__()
        assert System1.Component not in system_component_subclasses
        assert System2.Component not in system_component_subclasses
        assert System1.Event not in system_event_subclasses
        assert System2.Event not in system_event_subclasses


class TestArrayAttribute:
    class Example:
        entity_id: int
        attr = ArrayAttribute(np.uint8, length=100)

        def __init__(self, id_):
            self.entity_id = id_

    def test_types_receive_instance_of_the_array(self):
        assert self.Example.attr.dtype == np.uint8
        assert (100,) == self.Example.attr.shape

    def test_get_instances_index_into_array(self):
        self.Example.attr[10] = 150
        obj = self.Example(10)

        assert 150 == obj.attr

    def test_set_instance_index_into_array(self):
        obj = self.Example(14)
        obj.attr = 14

        assert 14 == self.Example.attr[14]

    def test_incompatible_if_object_has_no_entity_id(self):
        class BadExample:
            attr = ArrayAttribute(np.uint8, 10)

        obj = BadExample()

        with pytest.raises(AttributeError):
            obj.attr = 100

    def test_index_error_on_out_of_bounds_entity_id(self):
        obj = self.Example(1999)

        with pytest.raises(IndexError):
            obj.attr = 100


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
        class MyObject:
            attr = PublicAttribute(np.uint8)

            def __init__(self, entity_id):
                self.entity_id = entity_id

        return MyObject


class System1(System):
    pass


class System2(System):
    pass
