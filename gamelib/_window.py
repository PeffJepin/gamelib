import moderngl
import logging
import moderngl_window as mglw
from moderngl_window.conf import settings

from gamelib import input
from gamelib import events

Window = mglw.BaseWindow
Context = moderngl.Context
context: Context = None

_window: Window = None
_frames_offset = 0
_button_type_lookup = dict()
_input_type_lookup = dict()
_poll_for_input = ""

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


def init(headless=False, **config):
    global _window
    global _frames_offset
    if _window is not None:
        _frames_offset = _window.frames
        return

    if "class" not in config:
        config["class"] = "moderngl_window.context.pygame2.Window"
    if headless:
        config["class"] = "moderngl_window.context.headless.Window"
    if config["class"] == "moderngl_window.context.glfw.Window":
        # polling for events needs glfw in namespace for eval
        # see _polling_function_lookup
        import glfw

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
    _window.swap_buffers()


def clear(*args, **kwargs):
    _window.clear(*args, **kwargs)


def frame():
    return _window.frames - _frames_offset


def close():
    _window.close()


def is_running():
    return not _window.is_closing


def post_input():
    eval(_poll_for_input, {}, {"self": _window})
    while _queued_input:
        events.post(_queued_input.pop(0))
    dispatch_is_pressed_events()


def dispatch_is_pressed_events():
    for key_enum in input.monitored_key_states:
        mglw_key = getattr(_window.keys, key_enum.name, None)

        if not mglw_key:
            logging.debug(f"Key mapping not found for {key_enum!r}.")
            continue

        if _window.is_key_pressed(mglw_key):
            events.post(input.KeyIsPressed(key_enum, _get_modifiers()))


def _get_buttons():
    m = _window.mouse_states
    return input.Buttons(m.left, m.right, m.middle)


def _get_modifiers():
    mods = _window.modifiers
    return input.Modifiers(mods.shift, mods.ctrl, mods.alt)


def _hook_window_events():
    def _broadcast_key_event(key, action, modifiers):
        key = _input_type_lookup.get(key)
        modifiers = input.Modifiers(
            modifiers.shift, modifiers.ctrl, modifiers.alt
        )
        if not key:
            return
        if action == _window.keys.ACTION_PRESS:
            event = input.KeyDown(key, modifiers)
        else:
            event = input.KeyUp(key, modifiers)
        _queued_input.append(event)

    def _broadcast_mouse_press_event(x, y, button):
        event = input.MouseDown(x, y, _button_type_lookup[button])
        _queued_input.append(event)

    def _broadcast_mouse_release_event(x, y, button):
        event = input.MouseUp(x, y, _button_type_lookup[button])
        _queued_input.append(event)

    def _broadcast_mouse_motion_event(x, y, dx, dy):
        event = input.MouseMotion(x, y, dx, dy)
        _queued_input.append(event)

    def _broadcast_mouse_drag_event(x, y, dx, dy):
        event = input.MouseDrag(x, y, dx, dy, buttons=_get_buttons())
        _queued_input.append(event)

    def _broadcast_mouse_wheel_event(dx, dy):
        event = input.MouseScroll(dx, dy)
        _queued_input.append(event)

    _window.key_event_func = _broadcast_key_event
    _window.mouse_press_event_func = _broadcast_mouse_press_event
    _window.mouse_release_event_func = _broadcast_mouse_release_event
    _window.mouse_position_event_func = _broadcast_mouse_motion_event
    _window.mouse_drag_event_func = _broadcast_mouse_drag_event
    _window.mouse_scroll_event_func = _broadcast_mouse_wheel_event
