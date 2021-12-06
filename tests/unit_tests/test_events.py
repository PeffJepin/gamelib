from __future__ import annotations

import pytest
import time
from collections import defaultdict
from multiprocessing.connection import Pipe

from gamelib import Keys, events

from gamelib import KeyDown, ModifierKeys
from gamelib.events import (
    Event,
    eventhandler,
    _HANDLER_INJECTION_ATTRIBUTE,
)
from tests.conftest import RecordedCallback


class KeyedEvent(Event):
    pass


class SomeEvent(Event):
    pass


class SomeOtherEvent(Event):
    pass


class TestEventHandling:
    def test_does_call_registered_callback(self, recorded_callback):
        events.register(Event, recorded_callback)

        events.post(Event())

        assert recorded_callback.called

    def test_does_not_call_registered_callback(self, recorded_callback):
        events.register(Event, recorded_callback)

        events.post(SomeEvent())

        assert not recorded_callback.called

    def test_does_call_registered_with_key(self, recorded_callback):
        events.register(Event.B, recorded_callback)

        events.post(Event(), key="B")

        assert recorded_callback.called

    def test_does_not_call_registered_with_key(self, recorded_callback):
        events.register(Event.B, recorded_callback)

        events.post(Event(), key=1)

        assert not recorded_callback.called

    def test_callback_receives_event_as_arg(self, recorded_callback):
        events.register(Event, recorded_callback)

        event = Event()
        events.post(event)

        assert recorded_callback.called
        assert recorded_callback.event is event

    def test_not_called_after_being_unregistered(self, recorded_callback):
        events.register(Event, recorded_callback)

        events.unregister(Event, recorded_callback)
        events.post(Event())

        assert not recorded_callback.called

    def test_clearing_a_type_of_event(self):
        cb1, cb2, cb3 = [RecordedCallback() for _ in range(3)]
        events.register(SomeEvent, cb1)
        events.register(SomeEvent, cb2)
        events.register(SomeOtherEvent, cb3)

        events.clear_handlers(SomeEvent)

        events.post(SomeEvent())
        events.post(SomeOtherEvent())
        assert not cb1.called and not cb2.called
        assert cb3.called

    def test_feeds_event_and_key_into_serviced_pipe(self):
        a, b = Pipe()
        event = Event()

        events.service_connection(a, Event)
        events.post(event)

        if not b.poll(10 / 1_000):
            raise AssertionError("Nothing in pipe.")
        assert (event, None) == b.recv()

    def test_does_not_feed_event_into_serviced_pipe(self):
        a, b = Pipe()
        event = SomeEvent()

        events.service_connection(a, Event)
        events.post(event)

        assert not b.poll(0)

    def test_reads_event_sent_through_pipe_and_posts_them(
        self, recorded_callback
    ):
        a, b = Pipe()

        events.service_connection(a)
        events.register(Event, recorded_callback)

        b.send((Event(), None))
        for _ in range(100):
            if recorded_callback.called:
                return  # success
            time.sleep(1 / 100)
        assert False  # no callback

    def test_posts_events_sent_through_pipe_with_key(self, recorded_callback):
        a, b = Pipe()

        events.service_connection(a)
        events.register(Event.ABC, recorded_callback)

        b.send((Event(), "ABC"))
        for _ in range(100):
            if recorded_callback.called:
                return  # success
            time.sleep(1 / 100)
        assert False  # no callback

    def test_pipe_does_not_get_event_after_service_stops(self):
        a, b = Pipe()
        events.service_connection(a, Event)

        events.stop_connection_service(a)
        events.post(Event())

        assert not b.poll(0)


class TestHandlerDecorator:
    class ExampleUsage:
        field: int = 0

        @eventhandler(SomeEvent)
        def field_incrementer(self, event):
            self.field += 1

        @eventhandler(SomeEvent)
        def some_dummy_method(self, event):
            pass

        @eventhandler(SomeOtherEvent)
        def another_dummy_method(self, event):
            pass

        @eventhandler(SomeOtherEvent.A)
        def keyed_handler(self, event):
            pass

    @pytest.mark.parametrize(
        "fn, expected_key",
        (
            (ExampleUsage.field_incrementer, (SomeEvent, None)),
            (ExampleUsage.some_dummy_method, (SomeEvent, None)),
            (ExampleUsage.another_dummy_method, (SomeOtherEvent, None)),
            (ExampleUsage.keyed_handler, (SomeOtherEvent, "A")),
        ),
    )
    def test_handler_marks_methods_on_type_object(self, fn, expected_key):
        assert expected_key == getattr(fn, _HANDLER_INJECTION_ATTRIBUTE)

    def test_all_marked_handlers_can_be_found_on_an_instance(self):
        instance = self.ExampleUsage()
        fns = [
            instance.field_incrementer,
            instance.some_dummy_method,
            instance.another_dummy_method,
            instance.keyed_handler,
        ]
        discovered = []
        for handlers in events.find_eventhandlers(instance).values():
            discovered.extend(handlers)
        for fn in fns:
            assert fn in discovered

    def test_methods_marked_as_handlers_can_be_called_normally(self):
        inst = self.ExampleUsage()
        inst.field_incrementer(SomeEvent())

        assert inst.field == 1

    def test_methods_discovered_by_events_module_are_bound_to_the_given_instance(
        self,
    ):
        inst = self.ExampleUsage()
        handlers = events.find_eventhandlers(inst)
        for handler_ in handlers[(SomeEvent, None)]:
            handler_(SomeEvent())

        assert inst.field == 1


class TestEvent:
    def test_attr_lookup_on_type_returns_a_key_value(self):
        assert Event.A == (Event, "A")

    def test_keys_can_be_limited_by_a_set_of_strings(self):
        class LimitedEvent(Event):
            key_options = {"A", "B", "C"}

        assert LimitedEvent.A == (LimitedEvent, "A")
        with pytest.raises(ValueError):
            error = LimitedEvent.D

    def test_keys_can_be_mapped_to_other_values_with_a_class(self):
        class KeyMap:
            A = 1
            B = 2
            C = 3

        class MappedEvent(Event):
            key_options = KeyMap

        assert MappedEvent.A == (MappedEvent, 1)
        with pytest.raises(AttributeError):
            error = MappedEvent.D

    def test_keys_can_be_mapped_to_other_values_with_a_dict(self):
        map_ = {
            "A": 1,
            "B": 2,
            "C": 3,
        }

        class MappedEvent(Event):
            key_options = map_

        assert MappedEvent.B == (MappedEvent, 2)
        with pytest.raises(KeyError):
            error = MappedEvent.D

    def test_default_init_with_args(self):
        class MyEvent(Event):
            __slots__ = ["field1", "field2"]

        event = MyEvent(1, 2)

        assert 1 == event.field1 and 2 == event.field2

    def test_default_init_with_kwargs(self):
        class MyEvent(Event):
            __slots__ = ["field1", "field2"]

        event = MyEvent(field2=1, field1=2)

        assert 2 == event.field1 and 1 == event.field2

    def test_default_init_with_both_args_and_kwargs_raises_value_error(self):
        class MyEvent(Event):
            __slots__ = ["field1", "field2"]

        with pytest.raises(ValueError):
            MyEvent(1, field2=2)


class TestModule:
    @pytest.fixture
    def handler_container(self):
        container = HandlerContainer()
        events.register_marked(container)
        return container

    def test_normal_event_should_be_called(self, handler_container):
        events.post(Event())

        assert 1 == handler_container.calls[Event]

    def test_normal_event_should_not_be_called(self, handler_container):
        class OtherEvent(Event):
            pass

        events.post(OtherEvent())

        assert 0 == handler_container.calls[Event]

    def test_keyed_event_should_be_called(self, handler_container):
        events.post(KeyedEvent(), key="ABC")

        assert 1 == handler_container.calls[KeyedEvent]

    def test_keyed_event_should_not_be_called(self, handler_container):
        events.post(KeyedEvent(), key="CBA")

        assert 0 == handler_container.calls[KeyedEvent]

    def test_key_handler_maps_with_keys(self, handler_container):
        events.post(KeyDown(ModifierKeys(False, False, False)), key=Keys.J)

        assert 1 == handler_container.calls[KeyDown]


class HandlerContainer:
    def __init__(self):
        self.calls = defaultdict(int)

    @eventhandler(Event)
    def some_event_handler(self, _):
        self.calls[Event] += 1

    @eventhandler(KeyedEvent.ABC)
    def keyed_event_handler(self, _):
        self.calls[KeyedEvent] += 1

    @eventhandler(KeyDown.J)
    def j_down_handler(self, _):
        self.calls[KeyDown] += 1
