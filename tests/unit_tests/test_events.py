import time
from multiprocessing.connection import Pipe

from src.gamelib.events import MessageBus, Event


class TestMessageBus:
    def test_calls_registered_callbacks_when_an_event_is_received(self, recorded_callback, example_event):
        mb = MessageBus()
        mb.register(type(example_event), recorded_callback)

        mb.handle(example_event)

        assert recorded_callback.called

    def test_event_callback_receives_event_as_an_argument(self, recorded_callback, example_event):
        mb = MessageBus()
        mb.register(type(example_event), recorded_callback)

        mb.handle(example_event)

        assert recorded_callback.args[0] is example_event

    def test_event_callback_stops_being_called_if_unregistered(self, recorded_callback, example_event):
        mb = MessageBus()
        mb.register(type(example_event), recorded_callback)

        mb.unregister(type(example_event), recorded_callback)
        mb.handle(example_event)

        assert not recorded_callback.called

    def test_callback_is_not_called_upon_handling_a_different_type_of_event(self, example_event, recorded_callback):
        class NotExampleEvent(Event):
            pass

        other_event = NotExampleEvent()
        mb = MessageBus()
        mb.register(type(example_event), recorded_callback)

        mb.handle(other_event)

        assert not recorded_callback.called

    def test_can_feed_events_into_a_serviced_pipe(self, example_event):
        a, b = Pipe()
        mb = MessageBus()
        mb.service_connection(a, [type(example_event)])

        mb.handle(example_event)

        assert b.recv() == example_event

    def test_will_handle_events_that_get_sent_through_a_serviced_pipe(self, example_event, recorded_callback):
        a, b = Pipe()
        mb = MessageBus()
        mb.service_connection(a, [type(example_event)])
        mb.register(type(example_event), recorded_callback)

        b.send(example_event)
        i = 0
        while True:
            # wait no longer than 10ms for pipe to clear
            if not a.poll(0):
                break
            time.sleep(0.001)
            i += 1
            if i > 10:
                raise AssertionError("The Event remains in the pipe and is not being handled.")

        assert recorded_callback.called