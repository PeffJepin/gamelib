from collections import namedtuple
import pathlib

import moderngl
import moderngl_window
from moderngl_window.context.pygame2.keys import Keys
import pygame

from .events import Event
from ._window import make_window
from . import resources
from . import gl

window: moderngl_window.BaseWindow
context: moderngl.Context

ModifierKeys = namedtuple("KeyModifiers", "SHIFT, CTRL, ALT")  # Boolean values
MouseButtons = namedtuple(
    "MouseButtons", "LEFT, RIGHT, MIDDLE"
)  # Boolean values
_MOUSE_MAP = {"LEFT": 1, "RIGHT": 2, "MIDDLE": 3}


def init(headless=False, **config):
    global window
    global context

    pygame.init()
    resources.discover_directories(pathlib.Path.cwd())
    window = make_window(headless=headless, **config)
    context = window.ctx
    return window


def exit():
    if window:
        window.close()
    if context:
        context.release()


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
