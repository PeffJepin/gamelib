import pytest

from gamelib.core import input, events
from gamelib.core.input import (
    InputSchema,
    Keyboard,
    Buttons,
    MouseDrag,
    MouseScroll,
    MouseMotion,
    Modifiers,
    MouseButton,
    KeyDown,
    MouseDown,
    KeyUp,
    KeyIsPressed,
    MouseUp,
    MouseIsPressed,
)

from tests.conftest import RecordedCallback


@pytest.fixture(autouse=True)
def cleanup():
    events.clear_handlers()


@pytest.fixture(
    params=(
        ("c", KeyDown(Keyboard.C, Modifiers())),
        ("mouse1", MouseDown(0, 0, MouseButton.LEFT)),
        ("scroll", MouseScroll(0, 0)),
        ("motion", MouseMotion(0, 0, 0, 0)),
        ("drag", MouseDrag(0, 0, 0, 0, Buttons(left=True))),
    )
)
def schema_str_and_event(request):
    input_str, provoking_event = request.param
    yield input_str, provoking_event


def test_input_schema_event_basic_integration(
    schema_str_and_event, recorded_callback
):
    bad_callback = RecordedCallback()
    schema_str, event = schema_str_and_event
    schema = InputSchema(
        ("a", "down", bad_callback),
        ("mouse1", "up", bad_callback),
        (schema_str, recorded_callback),
    )

    events.post(event)
    assert recorded_callback.called and not bad_callback.called


def test_enable_disable_input_schema(schema_str_and_event, recorded_callback):
    schema_str, event = schema_str_and_event
    schema = InputSchema(
        (schema_str, recorded_callback),
    )

    schema.disable()
    events.post(event)
    assert not recorded_callback.called

    schema.enable()
    events.post(event)
    assert recorded_callback.called


def test_multiple_schemas(schema_str_and_event):
    cb1, cb2 = [RecordedCallback() for _ in range(2)]
    schema_str, event = schema_str_and_event
    schema1 = InputSchema((schema_str, cb1))
    schema2 = InputSchema((schema_str, cb2))

    events.post(event)
    assert cb1.called and cb2.called

    schema2.disable()
    events.post(event)
    assert cb1.called == 2 and cb2.called == 1

    schema2.enable(master=True)
    events.post(event)
    assert cb1.called == 2 and cb2.called == 2


class TestEventHandlerDecorators:
    @pytest.mark.parametrize("event_type", (KeyUp, KeyDown, KeyIsPressed))
    def test_key_event_handlers_base_case(self, event_type):
        class MyClass:
            i = 0

            @event_type.handler
            def my_handler(self, event):
                self.i += 1

        obj = MyClass()
        input.enable_handlers(obj)

        for key in Keyboard:
            prev = obj.i
            event = event_type(key, Modifiers())
            events.post(event)
            assert obj.i == prev + 1

        input.disable_handlers(obj)
        disabled_at = obj.i

        for key in Keyboard:
            event = event_type(key, Modifiers())
            events.post(event)
            assert obj.i == disabled_at

    @pytest.mark.parametrize("event_type", (KeyUp, KeyDown, KeyIsPressed))
    def test_key_event_with_several_marked_enums(self, event_type):
        class MyClass:
            i = 0

            @event_type.handler(iter("abcdefg"))
            def my_handler(self, event):
                self.i += 1

        obj = MyClass()
        input.enable_handlers(obj)
        expected_calls_from = {
            Keyboard.A,
            Keyboard.B,
            Keyboard.C,
            Keyboard.D,
            Keyboard.E,
            Keyboard.F,
            Keyboard.G,
        }

        for key in Keyboard:
            prev = obj.i
            event = event_type(key, Modifiers())
            events.post(event)

            if key in expected_calls_from:
                assert obj.i == prev + 1
            else:
                assert obj.i == prev

        input.disable_handlers(obj)

        for key in Keyboard:
            event = event_type(key, Modifiers())
            events.post(event)
        assert obj.i == 7

    @pytest.mark.parametrize("event_type", (KeyUp, KeyDown, KeyIsPressed))
    def test_key_event_with_one_marked_enum(self, event_type):
        class MyClass:
            i = 0

            @event_type.handler("a")
            def my_handler(self, event):
                self.i += 1

        obj = MyClass()
        input.enable_handlers(obj)

        for key in Keyboard:
            prev = obj.i
            event = event_type(key, Modifiers())
            events.post(event)
            if key == Keyboard.A:
                assert obj.i == 1
            else:
                assert obj.i == prev

        input.disable_handlers(obj)
        event = event_type(Keyboard.A, Modifiers())
        events.post(event)
        assert obj.i == 1

    @pytest.mark.parametrize("type", (MouseDown, MouseUp, MouseIsPressed))
    def test_mouse_button_events(self, type):
        class MyClass:
            i = 0

            @type.handler
            def plain(self, event):
                self.i += 1

        obj = MyClass()
        input.enable_handlers(obj)

        for button in MouseButton:
            prev = obj.i
            events.post(type(0, 0, button))
            assert obj.i == prev + 1

        last_call = obj.i
        input.disable_handlers(obj)

        for button in MouseButton:
            events.post(type(0, 0, button))
        assert obj.i == last_call

    def test_mouse_motion(self):
        class MyClass:
            i = 0

            @MouseMotion.handler
            def my_handler(self, event):
                self.i += 1

        obj = MyClass()
        input.enable_handlers(obj)

        events.post(MouseMotion(0, 0, 0, 0))
        assert obj.i == 1
        events.post(MouseDrag(0, 0, 0, 0, Buttons(True)))
        assert obj.i == 1

        input.disable_handlers(obj)
        events.post(MouseMotion(0, 0, 0, 0))
        assert obj.i == 1

    def test_mouse_drag(self):
        class MyClass:
            i = 0

            @MouseDrag.handler
            def my_handler(self, event):
                self.i += 1

        obj = MyClass()
        input.enable_handlers(obj)

        events.post(MouseDrag(0, 0, 0, 0, Buttons(True)))
        assert obj.i == 1
        events.post(MouseMotion(0, 0, 0, 0))
        assert obj.i == 1

        input.disable_handlers(obj)
        events.post(MouseDrag(0, 0, 0, 0, Buttons(True)))
        assert obj.i == 1

    def test_mouse_scroll(self):
        class MyClass:
            i = 0

            @MouseScroll.handler
            def my_handler(self, event):
                self.i += 1

        obj = MyClass()
        input.enable_handlers(obj)

        events.post(MouseScroll(0, 0))
        assert obj.i == 1
        events.post(MouseMotion(0, 0, 0, 0))
        assert obj.i == 1

        input.disable_handlers(obj)
        events.post(MouseScroll(0, 0))
        assert obj.i == 1
