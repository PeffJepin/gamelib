import pytest

from src.gamelib.environment import EntityFactory, EntityCreated, ComponentCreated
from src.gamelib.system import BaseComponent


class TestEntityFactory:
    def test_posts_entity_created_event(self, fake_message_bus):
        factory = EntityFactory(fake_message_bus)

        factory.create()

        assert fake_message_bus.posted

    def test_entity_id_is_incremented_for_each_entity_created(self, fake_message_bus):
        factory = EntityFactory(fake_message_bus)

        factory.create()
        assert EntityCreated(id=0) == fake_message_bus.pop_event()

        factory.create()
        assert EntityCreated(id=1) == fake_message_bus.pop_event()

    def test_raises_index_error_if_max_entities_are_exceeded(self, fake_message_bus):
        factory = EntityFactory(fake_message_bus, max_entities=10)

        for _ in range(10):
            factory.create()

        with pytest.raises(IndexError):
            factory.create()

    def test_posts_component_created_events(self, fake_message_bus):
        factory = EntityFactory(fake_message_bus)

        factory.create((SomeComponent, "9"), (SomeComponent, "1"))

        assert (
            ComponentCreated(entity_id=0, type=SomeComponent, args=("9",))
            in fake_message_bus.posted
        )
        assert (
            ComponentCreated(entity_id=0, type=SomeComponent, args=("1",))
            in fake_message_bus.posted
        )


class SomeComponent(BaseComponent):
    pass
