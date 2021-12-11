import pathlib
from dataclasses import dataclass

from . import _window
from . import resources
from . import events
from . import time


@dataclass
class _Config:
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


config = _Config()
schedule = time.Schedule()
threaded_schedule = time.Schedule(threaded=True)

_commands = _dummy_func
_update_timer = time.Timer()
_render_timer = time.Timer()
_initialized = False


def init(headless=False, **kwargs):
    global _initialized
    if _initialized:
        return

    resources.discover_directories(pathlib.Path.cwd())
    _window.init(headless=headless, **kwargs)
    _initialized = True


def update():
    # not sure yet if I'll use an async main loop or offload rendering onto
    # it's own thread. For now this solution will suffice.

    now = time.Timer.now()
    next_frame = _render_timer.remaining(now=now)
    next_update = _update_timer.remaining(now=now)

    if next_frame < next_update:
        _render_timer.tick(config.fps)
        _window.swap_buffers()
        update()
    else:
        _window.clear()
        _commands()
        _window.post_input()
        dt = _update_timer.tick(config.tps)
        events.post(events.Update(dt))
        schedule.update()


def exit():
    _window.close()


def run():
    if not _initialized:
        init()
    while _window.is_running():
        update()
    exit()


def is_running():
    return _window.is_running()


def set_draw_commands(func):
    global _commands
    _commands = func
