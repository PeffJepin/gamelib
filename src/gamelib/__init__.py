from collections import namedtuple

from moderngl_window.context.pygame2.keys import Keys

from .events import Event

ModifierKeys = namedtuple("KeyModifiers", "SHIFT, CTRL, ALT")  # Boolean values
MouseButtons = namedtuple("MouseButtons", "LEFT, RIGHT, MIDDLE")  # Boolean values
_MOUSE_MAP = {"LEFT": 1, "RIGHT": 2, "MIDDLE": 3}


class Update(Event):
    pass


class SystemStop(Event):
    pass


class Quit(Event):
    pass


class _BaseKeyEvent(Event):
    key_options = Keys

    __slots__ = ["modifiers"]

    modifiers: ModifierKeys


class KeyDown(_BaseKeyEvent):
    pass


class KeyUp(_BaseKeyEvent):
    pass


class KeyIsPressed(_BaseKeyEvent):
    pass


class MouseDrag(Event):
    __slots__ = ["buttons", "x", "y", "dx", "dy"]

    buttons: MouseButtons
    x: int
    y: int
    dx: int
    dy: int


class MouseMotion(Event):
    __slots__ = ["x", "y", "dx", "dy"]

    x: int
    y: int
    dx: int
    dy: int


class MouseScroll(Event):
    __slots__ = ["dx", "dy"]

    dx: int
    dy: int


class _BaseMouseEvent(Event):
    key_options = _MOUSE_MAP

    __slots__ = ["x", "y"]

    x: int
    y: int


class MouseDown(_BaseMouseEvent):
    pass


class MouseUp(_BaseMouseEvent):
    pass


class MouseIsPressed(_BaseMouseEvent):
    pass
