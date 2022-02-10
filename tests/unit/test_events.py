from __future__ import annotations

import pytest
import time
import dataclasses
from typing import NamedTuple
from multiprocessing.connection import Pipe

from gamelib.core import events

from tests.conftest import RecordedCallback


@pytest.fixture(autouse=True, scope="function")
def reset_module():
    events.clear_handlers()


class Event:
    def __init__(self, message="test"):
        self.message = message

    def __eq__(self, other):
        return self.message == other.message


@dataclasses.dataclass
class DataEvent:
    message: str = "test"


class TupleEvent(NamedTuple):
    message: str = "test"


class SomeOtherEvent:
    pass


@pytest.fixture(params=(Event(), DataEvent(), TupleEvent()))
def event(request):
    yield request.param


class TestEventHandling:
    def test_should_callback(self, recorded_callback, event):
        events.subscribe(type(event), recorded_callback)

        events.publish(event)

        assert recorded_callback.called

    def test_should_not_callback(self, recorded_callback, event):
        events.subscribe(type(event), recorded_callback)

        events.publish(SomeOtherEvent())

        assert not recorded_callback.called

    def test_callback_receives_event_as_arg(self, recorded_callback, event):
        events.subscribe(type(event), recorded_callback)

        events.publish(event)

        assert recorded_callback.called
        assert recorded_callback.event is event

    def test_not_called_after_being_unsubscribed(
        self, recorded_callback, event
    ):
        events.subscribe(type(event), recorded_callback)

        events.unsubscribe(type(event), recorded_callback)
        events.publish(event)

        assert not recorded_callback.called

    def test_clearing_a_type_of_event(self):
        cb1, cb2, cb3 = [RecordedCallback() for _ in range(3)]
        events.subscribe(Event, cb1)
        events.subscribe(DataEvent, cb2)
        events.subscribe(TupleEvent, cb3)

        events.clear_handlers(Event, DataEvent)
        events.publish(Event())
        events.publish(DataEvent())
        events.publish(TupleEvent())

        assert not cb1.called and not cb2.called
        assert cb3.called

    def test_feeds_serviced_event_into_pipe(self, event):
        a, b = Pipe()

        events.service_connection(a, type(event))
        events.publish(event)

        if not b.poll(0.01):
            raise AssertionError("Nothing in pipe.")
        assert event == b.recv()

    def test_does_not_feed_unserviced_event(self):
        a, b = Pipe()
        event = SomeOtherEvent()

        events.service_connection(a, Event)
        events.publish(event)

        assert not b.poll(0.01)

    def test_posts_event_received_at_serviced_connection(
        self, recorded_callback, event
    ):
        a, b = Pipe()

        events.service_connection(a)
        events.subscribe(type(event), recorded_callback)

        b.send(event)
        for _ in range(100):
            if recorded_callback.called:
                return  # success
            time.sleep(0.01)
        assert False  # no callback

    def test_pipe_does_not_get_event_after_service_stops(self, event):
        a, b = Pipe()
        events.service_connection(a, type(event))

        events.stop_connection_service(a)
        events.publish(event)

        assert not b.poll(0.001)


class HandlerContainer:
    def __init__(self):
        self.record = {
            Event: 0,
            DataEvent: 0,
            TupleEvent: 0,
            SomeOtherEvent: 0,
        }

    def _record_event(self, event):
        self.record[type(event)] += 1

    @events.handler(Event)
    def event_handler(self, event):
        self._record_event(event)

    @events.handler(DataEvent)
    def data_event_handler(self, event):
        self._record_event(event)

    @events.handler(TupleEvent)
    def tuple_event_handler(self, event):
        self._record_event(event)

    @events.handler(SomeOtherEvent)
    def some_other_handler(self, event):
        self._record_event(event)


class TestHandlerDecorator:
    def test_should_not_delegate_to_handlers(self, event):
        container = HandlerContainer()

        events.publish(event)

        assert container.record[type(event)] == 0

    def test_should_delegate_to_handlers(self, event):
        container = HandlerContainer()
        events.subscribe_marked(container)

        events.publish(event)

        assert container.record[type(event)] == 1

    def test_should_stop_after_unsubscribing(self, event):
        container = HandlerContainer()
        events.subscribe_marked(container)

        events.unsubscribe_marked(container)
        events.publish(event)

        assert container.record[type(event)] == 0

    def test_multiple_instances(self, event):
        c1 = HandlerContainer()
        c2 = HandlerContainer()
        type_ = type(event)

        events.subscribe_marked(c1)
        events.publish(event)
        assert c1.record[type_] == 1 and c2.record[type_] == 0

        events.subscribe_marked(c2)
        events.publish(event)
        assert c1.record[type_] == 2 and c2.record[type_] == 1

        events.unsubscribe_marked(c1)
        events.publish(event)
        assert c1.record[type_] == 2 and c2.record[type_] == 2

    def test_methods_marked_as_handlers_can_be_called_normally(self):
        container = HandlerContainer()
        container.event_handler(Event())

        assert container.record[Event] == 1
