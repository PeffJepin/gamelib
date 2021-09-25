import numpy as np
import pytest

from src.gamelib.system import System, ArrayAttribute


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

    def test_array_can_be_resized(self):
        array_attr = vars(self.Example)["attr"]
        array_attr.reallocate(200)
        assert (200,) == self.Example.attr.shape

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


class System1(System):
    pass


class System2(System):
    pass
