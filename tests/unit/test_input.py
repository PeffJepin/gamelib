import pytest

from gamelib.core import input
from gamelib.core.input import (
    Keyboard,
    Modifier,
    Action,
    KeyDown,
    KeyUp,
    InputSchema,
    MouseDown,
    MouseUp,
    MouseMotion,
    MouseDrag,
    Buttons,
    MouseScroll,
    Modifiers,
    MouseButton,
    KeyIsPressed,
    MouseIsPressed,
)

from tests.conftest import RecordedCallback


@pytest.fixture(autouse=True, scope="function")
def cleanup():
    input._key_states_to_monitor_lookup.clear()
    input.monitored_key_states.clear()


@pytest.mark.parametrize(
    "string, expected",
    (
        ("esc", Keyboard.ESCAPE),
        ("ESC", Keyboard.ESCAPE),
        ("Esc", Keyboard.ESCAPE),
        ("Escape", Keyboard.ESCAPE),
        ("\n", Keyboard.ENTER),
        (".", Keyboard.PERIOD),
        ("f1", Keyboard.F1),
        ("0", Keyboard.NUMBER_0),
        ("n_0", Keyboard.NUMPAD_0),
        ("numpad_0", Keyboard.NUMPAD_0),
        ("A", Keyboard.A),
        ("a", Keyboard.A),
        ("key_a", Keyboard.A),
        ("mouse1", MouseButton.LEFT),
        ("mouse_left", MouseButton.LEFT),
        ("shift", Modifier.SHIFT),
        ("ctrl", Modifier.CTRL),
        ("CTRL", Modifier.CTRL),
        ("control", Modifier.CTRL),
        ("press", Action.PRESS),
        ("on_press", Action.PRESS),
        ("down", Action.PRESS),
    ),
)
def test_mapping_strings_to_enum(string, expected):
    enum = expected.__class__
    assert enum.map_string(string) == expected
    assert expected == string


def test_tracking_keys_to_monitor(recorded_callback):
    dummy_func = recorded_callback
    schema1 = InputSchema(
        ("a", "is_pressed", dummy_func),
        ("b", "is_pressed", dummy_func),
        ("c", "is_pressed", dummy_func),
        enable=False,
    )
    schema2 = InputSchema(
        ("c", "is_pressed", dummy_func),
        ("d", "is_pressed", dummy_func),
        ("e", "is_pressed", dummy_func),
        enable=False,
    )

    assert len(input.monitored_key_states) == 0

    schema1.enable()
    expected = {Keyboard.A, Keyboard.B, Keyboard.C}
    assert expected == input.monitored_key_states

    schema2.enable()
    expected = {Keyboard.A, Keyboard.B, Keyboard.C, Keyboard.D, Keyboard.E}
    assert expected == input.monitored_key_states

    schema1.disable()
    expected = {Keyboard.C, Keyboard.D, Keyboard.E}
    assert expected == input.monitored_key_states


class TestKeyEvent:
    def test_no_options(self):
        callbacks = [RecordedCallback() for _ in range(3)]
        schema = InputSchema(
            ("a", callbacks[0]),
            ("b", callbacks[1]),
            ("c", callbacks[2]),
        )

        event = KeyDown(Keyboard.A, Modifiers())
        schema(event)
        assert [cb.called for cb in callbacks] == [1, 0, 0]

    @pytest.mark.parametrize(
        "event, expected_index",
        (
            (KeyDown(Keyboard.A, Modifiers()), 0),
            (KeyUp(Keyboard.A, Modifiers()), 1),
        ),
    )
    def test_with_input_action(self, event, expected_index):
        callbacks = [RecordedCallback() for _ in range(2)]
        schema = InputSchema(
            ("a", "down", callbacks[0]),
            ("a", "up", callbacks[1]),
        )

        schema(event)
        expected = [0, 0]
        expected[expected_index] = 1
        assert [cb.called for cb in callbacks] == expected

    @pytest.mark.parametrize(
        "mods, expected_index",
        (
            (Modifiers(), 0),
            (Modifiers(shift=True), 1),
            (Modifiers(shift=True, ctrl=True), 2),
            (Modifiers(shift=True, alt=True), 3),
            (Modifiers(shift=True, alt=True, ctrl=True), 4),
        ),
    )
    def test_with_modifiers(self, mods, expected_index):
        callbacks = [RecordedCallback() for _ in range(5)]
        schema = InputSchema(
            ("a", callbacks[0]),
            ("a", "shift", callbacks[1]),
            ("a", "shift", "ctrl", callbacks[2]),
            ("a", "shift", "alt", callbacks[3]),
            ("a", "shift", "alt", "ctrl", callbacks[4]),
        )
        event = KeyDown(Keyboard.A, mods)

        schema(event)
        expected = [0] * 5
        expected[expected_index] = 1
        assert [cb.called for cb in callbacks] == expected

    @pytest.mark.parametrize(
        "event, callback_index",
        (
            (KeyDown(Keyboard.A, Modifiers()), 0),
            (KeyDown(Keyboard.A, Modifiers(shift=True, alt=True)), 1),
            (KeyDown(Keyboard.A, Modifiers(alt=True)), 2),
            (KeyIsPressed(Keyboard.A, Modifiers()), 3),
            (KeyUp(Keyboard.A, Modifiers()), 4),
            (KeyUp(Keyboard.A, Modifiers(shift=True, alt=True, ctrl=True)), 5),
            (KeyUp(Keyboard.A, Modifiers(ctrl=True)), 6),
            (KeyDown(Keyboard.B, Modifiers()), 7),
            (KeyDown(Keyboard.B, Modifiers(shift=True, alt=True)), 8),
            (KeyDown(Keyboard.B, Modifiers(alt=True)), 9),
        ),
    )
    def test_all_options_integrated(self, event, callback_index):
        callbacks = [RecordedCallback() for _ in range(10)]
        schema = InputSchema(
            ("a", "down", callbacks[0]),
            ("a", "down", ("shift", "alt"), callbacks[1]),
            ("a", "down", "alt", callbacks[2]),
            ("a", "is_pressed", callbacks[3]),
            ("a", "up", callbacks[4]),
            ("a", "up", ("shift", "ctrl", "alt"), callbacks[5]),
            ("a", "up", "ctrl", callbacks[6]),
            ("b", "down", callbacks[7]),
            ("b", "down", ("shift", "alt"), callbacks[8]),
            ("b", "down", "alt", callbacks[9]),
        )

        schema(event)
        expected = [0] * 10
        expected[callback_index] = 1
        assert [cb.called for cb in callbacks] == expected


@pytest.mark.parametrize(
    "event, expected_index",
    (
        (MouseUp(0, 0, MouseButton.LEFT), 0),
        (MouseDown(0, 0, MouseButton.LEFT), 1),
        (MouseDown(0, 0, MouseButton.RIGHT), 2),
        (MouseUp(0, 0, MouseButton.RIGHT), 3),
        (MouseIsPressed(0, 0, MouseButton.MIDDLE), 4),
    ),
)
def test_mouse_button_event(event, expected_index):
    callbacks = [RecordedCallback() for _ in range(5)]
    schema = InputSchema(
        ("mouse1", "up", callbacks[0]),
        ("mouse1", "down", callbacks[1]),
        ("mouse2", "down", callbacks[2]),
        ("mouse2", "up", callbacks[3]),
        ("mouse3", "is_pressed", callbacks[4]),
    )

    expected = [0] * 5
    expected[expected_index] = 1
    schema(event)
    assert [cb.called for cb in callbacks] == expected


def test_mouse_motion_event(recorded_callback):
    schema = InputSchema(
        ("motion", recorded_callback),
    )

    event = MouseMotion(0, 0, 1, 1)
    schema(event)
    assert recorded_callback.called


def test_mouse_drag_event(recorded_callback):
    schema = InputSchema(("drag", recorded_callback))
    event = MouseDrag(0, 0, 0, 0, Buttons(True, False, False))

    schema(event)
    assert recorded_callback.called


def test_mouse_wheel_event(recorded_callback):
    schema = InputSchema(("scroll", recorded_callback))
    event = MouseScroll(0, 0)

    schema(event)
    assert recorded_callback.called
