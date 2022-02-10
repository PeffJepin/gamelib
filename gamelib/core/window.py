"""The window module is the interface with moderngl_window dependency.
This is responsible for polling for user input and translating the window
events into gamelib events and provides the OpenGL context.

In the future I plan on implementing my own window, the public interface of
this module shouldn't really have to change when that happens.

Notes
-----
Link to the docs for the moderngl_window package.

https://moderngl-window.readthedocs.io/en/latest/
"""

import moderngl
import logging
import moderngl_window as mglw
from moderngl_window.conf import settings

from gamelib.core import input
from gamelib.core import events
from gamelib.core.vectors import Vec2

Window = mglw.BaseWindow
Context = moderngl.Context
context: Context = None

_window: Window = None
_frames_offset = 0
_button_type_lookup = dict()
_input_type_lookup = dict()
_poll_for_input = ""
_mouse_position = [0, 0]

# pooling so swap_buffers doesn't post events directly
_queued_input = []
# moderngl_window BaseWindow doesn't have an event polling function,
# instead the events are polled when the buffers are swapped. I'd rather
# not have the two coupled together, but since different windows have
# different polling functions I need this lookup. In the future I'll submit
# a pull request for this and hopefully get a standard method implemented on
# the base window class and optional flag poll=True on the swap_buffer method.
_polling_function_lookup = {
    "moderngl_window.context.headless.Window": "None",
    "moderngl_window.context.glfw.Window": "glfw.poll_events()",
    "moderngl_window.context.pygame2.Window": "self.process_events()",
    "moderngl_window.context.pyglet.Window": "self._window.dispatch_events()",
    "moderngl_window.context.pyqt5.Window": "self._app.processEvents()",
    "moderngl_window.context.pyside2.Window": "self._app.processEvents()",
    "moderngl_window.context.sdl2.Window": "self.process_events()",
    "moderngl_window.context.tk.Window": "self._tk.update_idletasks(); self._tk.update()",
}


def create(headless=False, **config):
    """Initialize the window and construct mappings between the window
    providers constants and the gamelib constants.

    Parameters
    ----------
    headless : bool, optional
        Create a context with no window.
    **config : Any
        "gl_version": (3, 3),
        "class": "moderngl_window.context.pygame2.Window",
        "size": (1280, 720),
        "aspect_ratio": 16 / 9,
        "fullscreen": False,
        "resizable": True,
        "title": "ModernGL Window",
        "vsync": True,
        "cursor": True,
        "samples": 0

    Notes
    -----
    A link to more detailed documentation.

    https://moderngl-window.readthedocs.io/en/latest/reference/settings.conf.settings.html
    """

    global _window
    global _frames_offset
    if _window is not None:
        _frames_offset = _window.frames
        return

    # decide which window class to use
    if "class" not in config:
        config["class"] = "moderngl_window.context.pygame2.Window"
    if headless:
        config["class"] = "moderngl_window.context.headless.Window"

    if config["class"] == "moderngl_window.context.glfw.Window":
        # polling for events needs glfw in namespace for eval
        # see _polling_function_lookup
        pass

    global _poll_for_input
    _poll_for_input = _polling_function_lookup[config["class"]]

    for k, v in config.items():
        settings.WINDOW[k] = v

    _window = mglw.create_window_from_settings()
    global context
    context = _window.ctx

    # Map moderngl_window constants to gamelib enums. This needs to be
    # deferred until now because we don't know who the window provider will be
    # until the window has been made.
    global _button_type_lookup
    _button_type_lookup = {
        _window.mouse.left: input.MouseButton.LEFT,
        _window.mouse.right: input.MouseButton.RIGHT,
        _window.mouse.middle: input.MouseButton.MIDDLE,
    }
    global _input_type_lookup
    _input_type_lookup = {
        window_provider_value: input_type_enum
        for name, window_provider_value in vars(_window.keys).items()
        if (input_type_enum := getattr(input.Keyboard, name, None))
    }
    _hook_window_events()


def swap_buffers():
    """Swap framebuffer."""

    _window.swap_buffers()


def clear(red=0.0, green=0.0, blue=0.0, alpha=0.0, depth=1.0, viewport=None):
    """Clears the framebuffer. The float values should be on a range
    from 0 to 1.

    Parameters
    ----------
    red : float
    blue : float
    green : float
    alpha : float
    depth : float
    viewport : tuple
    """
    _window.clear(red, green, blue, alpha, depth, viewport)


def frames():
    """The current number of frames this window has rendered.
    This is reset by subsequent calls to create.

    Returns
    -------
    int
    """

    return _window.frames - _frames_offset


def close():
    """Close the window."""

    _window.close()


def is_running():
    """Hook for a main loop.

    Returns
    -------
    bool
        The close function will cause this to return False
    """

    return not _window.is_closing


def poll_for_user_input(dt):
    """Gathers input from the window provider and posts it into the gamelib
    event system."""

    eval(_poll_for_input, {}, {"self": _window})
    while _queued_input:
        events.publish(_queued_input.pop(0))
    dispatch_is_pressed_events(dt)


def dispatch_is_pressed_events(dt):
    """Checks each mouse button for state and posts events accordingly.
    Instead of checking all the keys this will only check keys which have
    been subscribed to with the input module currently.
    """

    for key_enum in input.monitored_key_states:
        mglw_key = getattr(_window.keys, key_enum.name, None)

        if not mglw_key:
            logging.debug(f"Key mapping not found for {key_enum!r}.")
            continue

        if _window.is_key_pressed(mglw_key):
            events.publish(input.KeyIsPressed(key_enum, _get_modifiers(), dt))

    if _window.mouse_states.left:
        events.publish(
            input.MouseIsPressed(
                *_mouse_position, button=input.MouseButton.LEFT, dt=dt
            )
        )
    elif _window.mouse_states.right:
        events.publish(
            input.MouseIsPressed(
                *_mouse_position, button=input.MouseButton.RIGHT, dt=dt
            )
        )
    elif _window.mouse_states.middle:
        events.publish(
            input.MouseIsPressed(
                *_mouse_position, button=input.MouseButton.MIDDLE, dt=dt
            )
        )


def _get_buttons():
    """Gets the state of the mouse buttons."""

    m = _window.mouse_states
    return input.Buttons(m.left, m.right, m.middle)


def _get_modifiers():
    """Gets the state of the modifier keys."""

    mods = _window.modifiers
    return input.Modifiers(mods.shift, mods.ctrl, mods.alt)


def _hook_window_events():
    """Defines functions to be integrated with the window that will
    adapt the window providers user input events into gamelib events."""

    def _broadcast_key_event(key, action, modifiers):
        key = _input_type_lookup.get(key)
        modifiers = input.Modifiers(
            bool(modifiers.shift), bool(modifiers.ctrl), bool(modifiers.alt)
        )
        if not key:
            return
        if action == _window.keys.ACTION_PRESS:
            event = input.KeyDown(key, modifiers)
        else:
            event = input.KeyUp(key, modifiers)
        _queued_input.append(event)

    def _broadcast_mouse_press_event(x, y, button):
        x, y = _transform_to_viewport_space(x, y)
        event = input.MouseDown(x, y, _button_type_lookup[button])
        _queued_input.append(event)

    def _broadcast_mouse_release_event(x, y, button):
        x, y = _transform_to_viewport_space(x, y)
        event = input.MouseUp(x, y, _button_type_lookup[button])
        _queued_input.append(event)

    def _broadcast_mouse_motion_event(x, y, dx, dy):
        x, y = _transform_to_viewport_space(x, y)
        dy *= -1
        _mouse_position[0], _mouse_position[1] = x, y
        event = input.MouseMotion(x, y, dx, dy)
        _queued_input.append(event)

    def _broadcast_mouse_drag_event(x, y, dx, dy):
        x, y = _transform_to_viewport_space(x, y)
        dy *= -1
        _mouse_position[0], _mouse_position[1] = x, y
        event = input.MouseDrag(x, y, dx, dy, buttons=_get_buttons())
        _queued_input.append(event)

    def _broadcast_mouse_wheel_event(dx, dy):
        dy *= -1
        event = input.MouseScroll(dx, dy)
        _queued_input.append(event)

    def _close_window(*args):
        exit()

    _window.close_func = _close_window
    _window.key_event_func = _broadcast_key_event
    _window.mouse_press_event_func = _broadcast_mouse_press_event
    _window.mouse_release_event_func = _broadcast_mouse_release_event
    _window.mouse_position_event_func = _broadcast_mouse_motion_event
    _window.mouse_drag_event_func = _broadcast_mouse_drag_event
    _window.mouse_scroll_event_func = _broadcast_mouse_wheel_event


def _transform_to_viewport_space(x, y):
    x = x - _window.viewport[0]
    y = y - _window.viewport[1]
    y = _window.viewport_height - y
    return x, y


def get_context():
    return context


def get_aspect_ratio():
    return _window.aspect_ratio


def get_width():
    return _window.viewport_width


def get_height():
    return _window.viewport_height


def get_cursor():
    # 0, 0 is bottom left in this coordinate space
    return Vec2(*_mouse_position)
