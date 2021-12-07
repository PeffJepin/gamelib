import moderngl_window as mglw
from moderngl_window.conf import settings

from gamelib import input
from gamelib import events


def make_window(headless=False, **config):
    if "class" not in config:
        config["class"] = "moderngl_window.context.pygame2.Window"
    if headless:
        config["class"] = "moderngl_window.context.headless.Window"
    for k, v in config.items():
        settings.WINDOW[k] = v

    window = mglw.create_window_from_settings()

    # map moderngl_window to input enums
    button_type_lookup = {
        window.mouse.left: input.MouseButton.LEFT,
        window.mouse.right: input.MouseButton.RIGHT,
        window.mouse.middle: input.MouseButton.MIDDLE,
    }
    input_type_lookup = {
        window_provider_value: input_type_enum
        for name, window_provider_value in vars(window.keys).items()
        if (input_type_enum := getattr(input.Keyboard, name, None))
    }

    def _get_buttons():
        m = window.mouse_states
        return input.Buttons(m.left, m.right, m.middle)

    def _broadcast_key_event(key, action, modifiers):
        key = input_type_lookup.get(key)
        modifiers = input.Modifiers(
            modifiers.shift, modifiers.ctrl, modifiers.alt
        )
        if not key:
            return
        if action == window.keys.ACTION_PRESS:
            event = input.KeyDown(key, modifiers)
        else:
            event = input.KeyUp(key, modifiers)
        events.post(event)

    def _broadcast_mouse_press_event(x, y, button):
        event = input.MouseDown(x, y, button_type_lookup[button])
        events.post(event)

    def _broadcast_mouse_release_event(x, y, button):
        event = input.MouseUp(x, y, button_type_lookup[button])
        events.post(event)

    def _broadcast_mouse_motion_event(x, y, dx, dy):
        event = input.MouseMotion(x, y, dx, dy)
        events.post(event)

    def _broadcast_mouse_drag_event(x, y, dx, dy):
        event = input.MouseDrag(x, y, dx, dy, buttons=_get_buttons())
        events.post(event)

    def _broadcast_mouse_wheel_event(dx, dy):
        event = input.MouseScroll(dx, dy)
        events.post(event)

    window.key_event_func = _broadcast_key_event
    window.mouse_press_event_func = _broadcast_mouse_press_event
    window.mouse_release_event_func = _broadcast_mouse_release_event
    window.mouse_position_event_func = _broadcast_mouse_motion_event
    window.mouse_drag_event_func = _broadcast_mouse_drag_event
    window.mouse_scroll_event_func = _broadcast_mouse_wheel_event

    return window
