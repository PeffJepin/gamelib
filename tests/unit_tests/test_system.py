from src.gamelib.system import System


class TestSystem:

    def test_subclasses_each_get_their_own_Event_attribute(self):
        assert System1.Event is not System2.Event

    def test_subclasses_each_get_their_own_Component_attribute(self):
        assert System1.Component is not System2.Component


class System1(System):
    pass


class System2(System):
    pass
