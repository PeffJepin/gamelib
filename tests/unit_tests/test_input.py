import pytest

from gamelib.input import (
    InputType as Type,
    InputMod as Mod,
    InputAction as Action,
    InputEvent as Event,
    InputSchema,
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


class TestInputSchema:
    def test_base_case(self):
        callbacks = [RecordedCallback() for _ in range(3)]
        schema = InputSchema(
            ("a", callbacks[0]),
            ("b", callbacks[1]),
            ("c", callbacks[2]),
        )

        event = Event(Type.A)
        schema(event)
        assert [cb.called for cb in callbacks] == [1, 0, 0]

    @pytest.mark.parametrize(
        "action, expected_index",
        ((Action.PRESS, 1), (Action.RELEASE, 2), (None, 0)),
    )
    def test_with_input_action(self, action, expected_index):
        callbacks = [RecordedCallback() for _ in range(3)]
        event = Event(Type.A, action)
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
        event = Event(Type.A, modifiers=mods)
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
            (Event(Type.A, Action.PRESS, (Mod.SHIFT, Mod.CTRL)), 0),
            (Event(Type.A, Action.PRESS, (Mod.SHIFT, Mod.CTRL, Mod.ALT)), 1),
            (Event(Type.A, Action.PRESS, (Mod.ALT, Mod.CTRL)), 2),
            (Event(Type.A), 3),
            (Event(Type.A, Action.RELEASE, (Mod.ALT,)), 4),
            (Event(Type.A, Action.RELEASE, (Mod.SHIFT, Mod.CTRL, Mod.ALT)), 5),
            (Event(Type.A, Action.RELEASE, (Mod.SHIFT, Mod.CTRL)), 6),
            (Event(Type.B, Action.PRESS, (Mod.CTRL,)), 7),
            (Event(Type.B, Action.PRESS, (Mod.CTRL, Mod.SHIFT, Mod.ALT)), 8),
            (Event(Type.B, Action.PRESS, (Mod.ALT, Mod.CTRL)), 9),
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
