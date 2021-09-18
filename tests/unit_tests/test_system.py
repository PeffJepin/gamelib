from src.gamelib.system import System



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


class System1(System):
    pass


class System2(System):
    pass
