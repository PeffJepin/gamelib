from dataclasses import dataclass

from src.gamelib.events import MessageBus, Event


class TestMessageBus:
    @dataclass
    class ExampleEvent(Event):
        string_field: str
        int_field: int

    def test_calls_registered_callbacks_when_an_event_is_received(self, recorded_callback):
        event = self.ExampleEvent('1', 1)
        mb = MessageBus()
        mb.register(self.ExampleEvent, recorded_callback)

        mb.handle(event)

        assert recorded_callback.called

    def test_event_callback_receives_event_as_an_argument(self, recorded_callback):
        event = self.ExampleEvent('1', 1)
        mb = MessageBus()
        mb.register(self.ExampleEvent, recorded_callback)

        mb.handle(event)

        assert recorded_callback.args[0] is event

