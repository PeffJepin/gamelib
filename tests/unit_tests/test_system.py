import numpy as np
import pytest

from src.gamelib import sharedmem
from src.gamelib.sharedmem import SharedBlock
from src.gamelib.system import System, ArrayAttribute, PublicAttribute, ProcessSystem


class TestSystem:
    def test_subclasses_each_get_their_own_Component_attribute(self):
        assert self.System1.Component is not self.System2.Component

    def test_subclass_attributes_are_unique_from_base_system_attributes(self):
        system_component_subclasses = System.Component.__subclasses__()
        assert self.System1.Component not in system_component_subclasses
        assert self.System2.Component not in system_component_subclasses

    def test_finds_public_attributes(self):
        attrs = [
            vars(self.Comp1)["attr1"],
            vars(self.Comp1)["attr2"],
            vars(self.Comp2)["attr1"],
        ]
        discovered = self.System1.public_attributes
        assert all([attr in discovered for attr in attrs])

    def test_finds_shared_arrays(self):
        attrs = [
            vars(self.Comp1)["attr1"],
            vars(self.Comp1)["attr2"],
            vars(self.Comp2)["attr1"],
        ]
        specs = [spec for attr in attrs for spec in attr.shared_specs]
        for spec in self.System1.shared_specs:
            assert spec in specs

    def test_can_find_related_array_attributes(self):
        all_array_attrs = [
            self.Comp1.__dict__["attr3"],
            self.Comp2.__dict__["attr2"],
            self.Comp2.__dict__["attr3"],
        ]
        for attr in self.System1.array_attributes:
            assert attr in all_array_attrs

    class System1(ProcessSystem):
        pass

    class System2(ProcessSystem):
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
        System.MAX_ENTITIES = 25
        vars(self.ExampleComponent)["attr"].reallocate()

        assert 25 == len(self.ExampleComponent.attr)

    def test_array_masks_components_that_dont_exist(self):
        for i in range(10):
            comp = self.ExampleComponent(i)
            comp.attr = 1
        assert all(1 == self.ExampleComponent.attr[:10])
        assert not any(self.ExampleComponent.attr[10:])

    class ExampleSystem(ProcessSystem):
        pass

    class ExampleComponent(ExampleSystem.Component):
        attr = ArrayAttribute(np.uint8)

    @pytest.fixture(autouse=True)
    def reallocate_attribute(self):
        attr = vars(self.ExampleComponent)['attr']
        attr.reallocate()


class TestPublicAttribute:
    def test_public_attribute_does_not_work_before_allocation(self, attr):
        with pytest.raises(Exception):
            attr = ExampleComponent.attr

    def test_access_can_be_made_after_allocation(self, allocated_attr):
        assert all(ExampleComponent.attr[:] == 0)

    def test_cannot_be_accessed_after_closed(self):
        attr = vars(ExampleComponent)["attr"]
        sharedmem.allocate(attr.shared_specs)
        ExampleComponent.attr[:] = 1

        sharedmem.unlink()
        attr.open = False
        with pytest.raises(Exception):
            ExampleComponent.attr

    def test_changes_not_reflected_until_update(self, allocated_attr):
        ExampleComponent.attr[:] = 10

        assert all(ExampleComponent.attr[:] == 0)
        allocated_attr.update()
        assert all(ExampleComponent.attr[:] == 10)

    def test_array_size_dictated_by_System_MAX_ENTITIES(self, attr):
        System.MAX_ENTITIES = 16
        attr = vars(ExampleComponent)["attr"]
        sharedmem.allocate(attr.shared_specs)

        assert len(ExampleComponent.attr) == 16

    def test_indexed_by_object_entity_id(self, allocated_attr):
        inst = ExampleComponent(5)
        inst.attr = 10

        assert ExampleComponent.attr[5] == 0
        allocated_attr.update()
        assert ExampleComponent.attr[5] == 10

    @pytest.fixture
    def attr(self):
        attr = vars(ExampleComponent)["attr"]
        try:
            yield attr
        finally:
            attr.open = False

    @pytest.fixture
    def allocated_attr(self):
        attr = vars(ExampleComponent)["attr"]
        sharedmem.allocate(attr.shared_specs)

        try:
            yield attr
        finally:
            attr.open = False


class ExampleSystem(ProcessSystem):
    MAX_ENTITIES = 100


class ExampleComponent(ExampleSystem.Component):
    attr = PublicAttribute(int)


@pytest.fixture(autouse=True)
def close_public_attrs():
    for attr in ExampleSystem.public_attributes:
        attr.open = False
