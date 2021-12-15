import pytest

from gamelib import events
from gamelib.input import (
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