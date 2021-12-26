"""This module initializes some internal variables for scheduling tasks
and running the main loop of an application."""

import pathlib
from dataclasses import dataclass

from . import _window
from . import resources
from . import events
from . import time


@dataclass
class _Config:
    """Global config variables.

    ticks per second >= frames per second
    or the buffers may swap back and forth in update
    """

    size: tuple = (1280, 720)
    _fps: int = 60
    _tps: int = 60

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, new_fps):
        assert new_fps <= self.tps
        self._fps = new_fps

    @property
    def tps(self):
        return self._tps

    @tps.setter
    def tps(self, new_tps):
        assert new_tps >= self.fps
        self._tps = new_tps


def _dummy_func():
    pass


# global config and scheduling variables exposed in gamelib/__init__.py
config = _Config()
schedule = time.Schedule()
threaded_schedule = time.Schedule(threaded=True)

_commands = _dummy_func
_update_timer = time.Timer()
_render_timer = time.Timer()
_initialized = False


def init(headless=False, **kwargs):
    """Hook for the package entry point to initialize the window,
    discover resources, etc.

    Parameters
    ----------
    headless : bool, optional
        Make a context with no window?
    **kwargs : Any
        Window config kwargs. see _window.create().
    """

    global _initialized
    if _initialized:
        return

    resources.discover_directories(pathlib.Path.cwd())
    _window.create(headless=headless, **kwargs)
    _initialized = True


def update():
    """Updates all the internal gamelib components.

    This includes:
        Clearing and issuing render commands if a render function has been
        provided.

        Posting the update event.

        Polling for user input.

        Calling scheduled functions.

    Calling this on a loop will tick along the application up to the speed
    specified in the config.
    """

    # not sure yet if I'll use an async main loop or just split this into
    # two threads. This should suffice for now.

    now = time.Timer.now()
    next_frame = _render_timer.remaining(now=now)
    next_update = _update_timer.remaining(now=now)

    if next_frame < next_update:
        _render_timer.tick(config.fps)
        _window.swap_buffers()
        update()
    else:
        if _commands != _dummy_func:
            _window.clear()
            _commands()
        dt = _update_timer.tick(config.tps)
        _window.poll_for_user_input(dt)
        events.post(events.Update(dt))
        schedule.update()


def exit():
    """Called to end an application."""

    _window.close()


def run():
    """A default mainloop for a simple application.

    This won't do rendering if `set_draw_commands` hasn't been called.
    """

    if not _initialized:
        init()
    while _window.is_running():
        update()
    exit()


def set_draw_commands(func):
    """This function will be called to issue draw commands to OpenGL."""

    global _commands
    _commands = func
