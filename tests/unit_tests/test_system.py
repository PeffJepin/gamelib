from src.gamelib.ecs.system import System
from src.gamelib.ecs.component import ArrayAttribute, PublicAttribute


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
            vars(self.Comp1)["attr3"],
            vars(self.Comp2)["attr2"],
            vars(self.Comp2)["attr3"],
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
