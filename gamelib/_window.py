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
        window.mouse.left: input.InputType.MOUSE1,
        window.mouse.right: input.InputType.MOUSE2,
        window.mouse.middle: input.InputType.MOUSE3,
    }
    input_type_lookup = {
        window_provider_value: input_type_enum
        for name, window_provider_value in vars(window.keys).items()
        if (input_type_enum := getattr(input.InputType, name, None))
    }

    def _get_modifiers():
        m = window.modifiers
        return input.Modifiers(m.shift, m.ctrl, m.alt)

    def _get_buttons():
        m = window.mouse_states
        return input.Buttons(m.left, m.right, m.middle)

    def _broadcast_key_event(key, action, modifiers):
        type = input_type_lookup.get(key)
        if not type:
            return
        if action == window.keys.ACTION_PRESS:
            action = input.InputAction.PRESS
        else:
            action = input.InputAction.RELEASE
        modifiers = input.Modifiers(
            modifiers.shift, modifiers.ctrl, modifiers.alt
        )
        event = input.KeyEvent(type, action, modifiers)
        events.post(event)

    def _broadcast_mouse_press_event(x, y, button):
        event = input.MouseButtonEvent(
            x,
            y,
            button=button_type_lookup[button],
            action=input.InputAction.PRESS,
            modifiers=_get_modifiers(),
        )
        events.post(event)

    def _broadcast_mouse_release_event(x, y, button):
        event = input.MouseButtonEvent(
            x,
            y,
            button=button_type_lookup[button],
            action=input.InputAction.RELEASE,
            modifiers=_get_modifiers(),
        )
        events.post(event)

    def _broadcast_mouse_motion_event(x, y, dx, dy):
        event = input.MouseMotionEvent(
            x, y, dx, dy, modifiers=_get_modifiers()
        )
        events.post(event)

    def _broadcast_mouse_drag_event(x, y, dx, dy):
        event = input.MouseDragEvent(
            x, y, dx, dy, buttons=_get_buttons(), modifiers=_get_modifiers()
        )
        events.post(event)

    def _broadcast_mouse_wheel_event(dx, dy):
        event = input.MouseScrollEvent(dx, dy)
        events.post(event)

    window.key_event_func = _broadcast_key_event
    window.mouse_press_event_func = _broadcast_mouse_press_event
    window.mouse_release_event_func = _broadcast_mouse_release_event
    window.mouse_position_event_func = _broadcast_mouse_motion_event
    window.mouse_drag_event_func = _broadcast_mouse_drag_event
    window.mouse_scroll_event_func = _broadcast_mouse_wheel_event

    return window
