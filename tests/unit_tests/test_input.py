import pytest

from gamelib.input import (
    InputType as Type,
    InputMod as Mod,
    InputAction as Action,
    KeyEvent,
    InputSchema,
    MouseButtonEvent,
    MouseMotionEvent,
    MouseDragEvent,
    Buttons,
    MouseScrollEvent,
)

from tests.conftest import RecordedCallback


@pytest.mark.parametrize(
    "string, enum, expected",
    (
        ("esc", Type, Type.ESCAPE),
        ("ESC", Type, Type.ESCAPE),
        ("Esc", Type, Type.ESCAPE),
        ("Escape", Type, Type.ESCAPE),
        ("\n", Type, Type.ENTER),
        (".", Type, Type.PERIOD),
        ("f1", Type, Type.F1),
        ("0", Type, Type.NUMBER_0),
        ("n_0", Type, Type.NUMPAD_0),
        ("numpad_0", Type, Type.NUMPAD_0),
        ("A", Type, Type.A),
        ("a", Type, Type.A),
        ("key_a", Type, Type.A),
        ("mouse1", Type, Type.MOUSE1),
        ("mouse_left", Type, Type.MOUSE1),
        ("shift", Mod, Mod.SHIFT),
        ("ctrl", Mod, Mod.CTRL),
        ("CTRL", Mod, Mod.CTRL),
        ("control", Mod, Mod.CTRL),
        ("press", Action, Action.PRESS),
        ("on_press", Action, Action.PRESS),
        ("down", Action, Action.PRESS),
    ),
)
def test_mapping_strings_to_enum(string, enum, expected):
    assert enum.map_string(string) == expected
    assert expected == string


class TestKeyEvent:
    def test_no_options(self):
        callbacks = [RecordedCallback() for _ in range(3)]
        schema = InputSchema(
            ("a", callbacks[0]),
            ("b", callbacks[1]),
            ("c", callbacks[2]),
        )

        event = KeyEvent(Type.A)
        schema(event)
        assert [cb.called for cb in callbacks] == [1, 0, 0]

    @pytest.mark.parametrize(
        "action, expected_index",
        ((Action.PRESS, 1), (Action.RELEASE, 2), (None, 0)),
    )
    def test_with_input_action(self, action, expected_index):
        callbacks = [RecordedCallback() for _ in range(3)]
        event = KeyEvent(Type.A, action)
        expected = [0, 0, 0]
        schema = InputSchema(
            ("a", callbacks[0]),
            ("a", "down", callbacks[1]),
            ("a", "up", callbacks[2]),
        )

        schema(event)
        expected[expected_index] = 1
        assert [cb.called for cb in callbacks] == expected

    @pytest.mark.parametrize(
        "mods, expected_index",
        (
            ((Mod.SHIFT,), 0),
            ((Mod.SHIFT, Mod.CTRL), 1),
            ((Mod.SHIFT, Mod.ALT), 0),
            ((Mod.SHIFT, Mod.CTRL, Mod.ALT), 1),
        ),
    )
    def test_with_modifiers(self, mods, expected_index):
        callbacks = [RecordedCallback() for _ in range(2)]
        event = KeyEvent(Type.A, modifiers=mods)
        expected = [0, 0]
        schema = InputSchema(
            ("a", "shift", callbacks[0]),
            ("a", ("shift", "ctrl"), callbacks[1]),
        )

        schema(event)
        expected[expected_index] = 1
        assert [cb.called for cb in callbacks] == expected

    @pytest.mark.parametrize(
        "event, callback_index",
        (
            (KeyEvent(Type.A, Action.PRESS, (Mod.SHIFT, Mod.CTRL)), 0),
            (
                KeyEvent(Type.A, Action.PRESS, (Mod.SHIFT, Mod.CTRL, Mod.ALT)),
                1,
            ),
            (KeyEvent(Type.A, Action.PRESS, (Mod.ALT, Mod.CTRL)), 2),
            (KeyEvent(Type.A), 3),
            (KeyEvent(Type.A, Action.RELEASE, (Mod.ALT,)), 4),
            (
                KeyEvent(
                    Type.A, Action.RELEASE, (Mod.SHIFT, Mod.CTRL, Mod.ALT)
                ),
                5,
            ),
            (KeyEvent(Type.A, Action.RELEASE, (Mod.SHIFT, Mod.CTRL)), 6),
            (KeyEvent(Type.B, Action.PRESS, (Mod.CTRL,)), 7),
            (
                KeyEvent(Type.B, Action.PRESS, (Mod.CTRL, Mod.SHIFT, Mod.ALT)),
                8,
            ),
            (KeyEvent(Type.B, Action.PRESS, (Mod.ALT, Mod.CTRL)), 9),
        ),
    )
    def test_all_options_integrated(self, event, callback_index):
        callbacks = [RecordedCallback() for _ in range(10)]
        schema = InputSchema(
            ("a", "down", callbacks[0]),
            ("a", "down", ("shift", "alt"), callbacks[1]),
            ("a", "down", "alt", callbacks[2]),
            ("a", callbacks[3]),
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
        (MouseButtonEvent(0, 0, button=Type.MOUSE1, action=Action.RELEASE), 0),
        (MouseButtonEvent(0, 0, button=Type.MOUSE1, action=Action.PRESS), 1),
        (MouseButtonEvent(0, 0, button=Type.MOUSE2, action=Action.PRESS), 2),
        (MouseButtonEvent(0, 0, button=Type.MOUSE2, action=Action.RELEASE), 3),
        (MouseButtonEvent(0, 0, button=Type.MOUSE3, action=Action.RELEASE), 4),
    ),
)
def test_mouse_button_event(event, expected_index):
    callbacks = [RecordedCallback() for _ in range(5)]
    schema = InputSchema(
        ("mouse1", "up", callbacks[0]),
        ("mouse1", "down", callbacks[1]),
        ("mouse2", "down", callbacks[2]),
        ("mouse2", "up", callbacks[3]),
        ("mouse3", callbacks[4]),
    )

    expected = [0] * 5
    expected[expected_index] = 1
    schema(event)
    assert [cb.called for cb in callbacks] == expected


def test_mouse_motion_event(recorded_callback):
    schema = InputSchema(
        ("motion", recorded_callback),
    )

    event = MouseMotionEvent(0, 0, 1, 1)
    schema(event)
    assert recorded_callback.called


def test_mouse_drag_event(recorded_callback):
    schema = InputSchema(("drag", recorded_callback))
    event = MouseDragEvent(0, 0, 0, 0, Buttons(True, False, False))

    schema(event)
    assert recorded_callback.called


def test_mouse_wheel_event(recorded_callback):
    schema = InputSchema(("scroll", recorded_callback))
    event = MouseScrollEvent(0, 0)

    schema(event)
    assert recorded_callback.called
