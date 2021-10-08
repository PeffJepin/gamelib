from collections import defaultdict

import pytest

from src.gamelib import KeyDown, ModifierKeys, Keys
from src.gamelib.events import (
    eventhandler,
    Event,
    post_event, register_marked,
)


class TestInternalIntegration:
    @pytest.fixture
    def handler_container(self):
        container = HandlerContainer()
        register_marked(container)
        return container

    def test_normal_event_should_be_called(self, handler_container):
        post_event(Event())

        assert 1 == handler_container.calls[Event]

    def test_normal_event_should_not_be_called(self, handler_container):
        class OtherEvent(Event):
            pass
        post_event(OtherEvent())

        assert 0 == handler_container.calls[Event]

    def test_keyed_event_should_be_called(self, handler_container):
        post_event(KeyedEvent(), key="ABC")

        assert 1 == handler_container.calls[KeyedEvent]

    def test_keyed_event_should_not_be_called(self, handler_container):
        post_event(KeyedEvent(), key="CBA")

        assert 0 == handler_container.calls[KeyedEvent]

    def test_key_handler_maps_with_keys(self, handler_container):
        post_event(KeyDown(ModifierKeys(False, False, False)), key=Keys.J)

        assert 1 == handler_container.calls[KeyDown]


class KeyedEvent(Event):
    pass


class HandlerContainer:
    def __init__(self):
        self.calls = defaultdict(int)

    @eventhandler(Event)
    def some_event_handler(self, event: Event):
        self.calls[Event] += 1

    @eventhandler(KeyedEvent.ABC)
    def keyed_event_handler(self, event: Event):
        self.calls[KeyedEvent] += 1

    @eventhandler(KeyDown.J)
    def j_down_handler(self, event):
        self.calls[KeyDown] += 1
