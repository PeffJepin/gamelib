import pathlib

import moderngl
import moderngl_window
import pygame

from ._window import make_window
from . import resources
from . import gl

window: moderngl_window.BaseWindow
context: moderngl.Context


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


class Update:
    pass


class SystemStop:
    pass


class Quit:
    pass
