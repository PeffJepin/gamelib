from enum import Enum
from typing import Any, NamedTuple

from gamelib import events


class StringMappingEnum(Enum):
    @classmethod
    def map_string(cls, string):
        for member in cls:
            if string.lower() in member.value:
                return member

    def __eq__(self, other):
        if isinstance(other, str):
            return other.lower() in self.value
        elif isinstance(other, type(self)):
            return other.name == self.name
        else:
            return False

    def __hash__(self):
        return hash(self.name)


class InputType(StringMappingEnum):
    """Defines which string values map to which keyboard inputs."""

    ESCAPE = ("escape", "esc")
    SPACE = ("space",)
    ENTER = ("enter", "return", "\n")
    PAGE_UP = ("page_up",)
    PAGE_DOWN = ("page_down",)
    LEFT = ("left",)
    RIGHT = ("right",)
    UP = ("up",)
    DOWN = ("down",)

    TAB = ("\t", "tab")
    COMMA = (",", "comma")
    MINUS = ("-", "minus")
    PERIOD = (".", "period")
    SLASH = ("/", "slash")
    SEMICOLON = (";", "semicolon")
    EQUAL = ("=", "equal")
    LEFT_BRACKET = ("[", "left_bracket")
    RIGHT_BRACKET = ("]", "right_bracket")
    BACKSLASH = ("\\", "backslash")
    BACKSPACE = ("back", "backspace")
    INSERT = ("ins", "insert")
    DELETE = ("del", "delete")
    HOME = ("home",)
    END = ("end",)
    CAPS_LOCK = ("caps", "caps_lock")

    F1 = ("f1",)
    F2 = ("f2",)
    F3 = ("f3",)
    F4 = ("f4",)
    F5 = ("f5",)
    F6 = ("f6",)
    F7 = ("f7",)
    F8 = ("f8",)
    F9 = ("f9",)
    F10 = ("f10",)
    F11 = ("f11",)
    F12 = ("f12",)

    NUMBER_0 = ("0",)
    NUMBER_1 = ("1",)
    NUMBER_2 = ("2",)
    NUMBER_3 = ("3",)
    NUMBER_4 = ("4",)
    NUMBER_5 = ("5",)
    NUMBER_6 = ("6",)
    NUMBER_7 = ("7",)
    NUMBER_8 = ("8",)
    NUMBER_9 = ("9",)

    NUMPAD_0 = ("n_0", "numpad_0")
    NUMPAD_1 = ("n_1", "numpad_1")
    NUMPAD_2 = ("n_2", "numpad_2")
    NUMPAD_3 = ("n_3", "numpad_3")
    NUMPAD_4 = ("n_4", "numpad_4")
    NUMPAD_5 = ("n_5", "numpad_5")
    NUMPAD_6 = ("n_6", "numpad_6")
    NUMPAD_7 = ("n_7", "numpad_7")
    NUMPAD_8 = ("n_8", "numpad_8")
    NUMPAD_9 = ("n_9", "numpad_9")

    A = ("a", "key_a")
    B = ("b", "key_b")
    C = ("c", "key_c")
    D = ("d", "key_d")
    E = ("e", "key_e")
    F = ("f", "key_f")
    G = ("g", "key_g")
    H = ("h", "key_h")
    I = ("i", "key_i")
    J = ("j", "key_j")
    K = ("k", "key_k")
    L = ("l", "key_l")
    M = ("m", "key_m")
    N = ("n", "key_n")
    O = ("o", "key_o")
    P = ("p", "key_p")
    Q = ("q", "key_q")
    R = ("r", "key_r")
    S = ("s", "key_s")
    T = ("t", "key_t")
    U = ("u", "key_u")
    V = ("v", "key_v")
    W = ("w", "key_w")
    X = ("x", "key_x")
    Y = ("y", "key_y")
    Z = ("z", "key_z")

    MOUSE1 = ("mouse1", "mouse_1", "mouse_left")
    MOUSE2 = ("mouse2", "mouse_2", "mouse_right")
    MOUSE3 = ("mouse3", "mouse_3", "mouse_middle")
    SCROLL = ("scroll", "mouse_scroll")
    MOTION = ("mouse_motion", "mouse_movement", "motion")
    DRAG = ("mouse_drag", "drag")


class InputMod(StringMappingEnum):
    SHIFT = ("shift", "mod1")
    CTRL = ("control", "ctrl", "mod2")
    ALT = ("alt", "mod3")


class InputAction(StringMappingEnum):
    PRESS = ("press", "down", "on_press", "on_down")
    RELEASE = ("release", "up", "on_release", "on_up")
    IS_PRESSED = ("pressed", "ispressed", "isdown", "is_pressed", "is_down")


class _InputHandlerTree:
    def __init__(self):
        mod_lookup = {
            # SHIFT, CTRL, ALT
            (True, True, True): None,
            (True, True, False): None,
            (True, False, True): None,
            (False, True, True): None,
            (False, False, True): None,
            (False, True, False): None,
            (True, False, False): None,
            (False, False, False): None,
        }
        self._registry = {
            InputAction.PRESS: mod_lookup.copy(),
            InputAction.RELEASE: mod_lookup.copy(),
            InputAction.IS_PRESSED: mod_lookup.copy(),
            None: mod_lookup.copy(),
        }

    def register(self, callback, modifiers=(), action=None):
        self._registry[action][self._mods_to_key(modifiers)] = callback

    def get_callback(self, mods=(), action=None) -> Any:
        mod_lookup = self._registry[action]
        for perm in self._mod_key_permutations(mods):
            callback = mod_lookup[perm]
            if callback is not None:
                return callback
        if action is not None:
            # fallback to no action specified
            return self.get_callback(mods, action=None)
        return None

    @staticmethod
    def _mods_to_key(mods):
        return (
            InputMod.SHIFT in mods,
            InputMod.CTRL in mods,
            InputMod.ALT in mods,
        )

    @classmethod
    def _mod_key_permutations(cls, mods):
        key = cls._mods_to_key(mods)

        # first look for exact match
        yield key

        # try flipping one mod off
        for i in range(3):
            if key[i]:
                yield *key[:i], False, *key[i + 1 :]

        # try flipping two mods off
        if key[0] and key[1]:
            yield False, False, key[2]
        if key[0] and key[2]:
            yield False, key[1], False
        if key[1] and key[2]:
            yield key[0], False, False

        # fallback to no mods
        yield False, False, False


class InputSchema:
    def __init__(self, *designations):
        self._callback_trees = dict()
        self._process_schema(*designations)
        self._events = [
            MouseScrollEvent,
            MouseButtonEvent,
            KeyEvent,
            MouseMotionEvent,
            MouseDragEvent,
        ]
        self.enable()

    def enable(self, master=False):
        if master:
            events.clear_handlers(*self._events)
        for event in self._events:
            events.register(event, self)

    def disable(self):
        for event in self._events:
            events.unregister(event, self)

    def _process_schema(self, *schema):
        for desc in schema:
            input_type, *optional, callback = desc

            if isinstance(input_type, str):
                input_type = InputType.map_string(input_type)

            mods = []
            action = None
            for arg in optional:
                # str might be mod or action
                if isinstance(arg, str):
                    if mod := InputMod.map_string(arg):
                        mods.append(mod)
                    else:
                        action = InputAction.map_string(arg)

                # list/tuple are mods
                elif isinstance(arg, list) or isinstance(arg, tuple):
                    for mod in arg:
                        if isinstance(mod, str):
                            mods.append(InputMod.map_string(mod))
                        else:
                            mods.append(mod)

            if input_type not in self._callback_trees:
                self._callback_trees[input_type] = _InputHandlerTree()
            callback_tree = self._callback_trees[input_type]
            callback_tree.register(callback, modifiers=mods, action=action)

    def __call__(self, event):
        callback_tree = self._callback_trees.get(event.type)
        if not callback_tree:
            return
        callback = callback_tree.get_callback(
            mods=event.modifiers,
            action=event.action,
        )
        if callback:
            callback(event)


class Modifiers(NamedTuple):
    SHIFT: bool
    CTRL: bool
    ALT: bool


class Buttons(NamedTuple):
    LEFT: bool
    RIGHT: bool
    MIDDLE: bool


class _InputEvent(events.Event):
    __slots__ = ["type", "action", "modifiers"]

    type: InputType
    action: InputAction
    modifiers: Modifiers

    def __init__(
        self,
        type,
        action=None,
        modifiers=Modifiers(False, False, False),
        **kwargs,
    ):
        super().__init__(
            type=type, action=action, modifiers=modifiers, **kwargs
        )


class KeyEvent(_InputEvent):
    # just for clarity
    pass


class MouseButtonEvent(_InputEvent):
    __slots__ = ["x", "y"]

    x: int
    y: int

    def __init__(self, x, y, *, button, **kwargs):
        super().__init__(type=button, x=x, y=y, **kwargs)


class MouseMotionEvent(_InputEvent):
    __slots__ = ["x", "y", "dx", "dy"]

    x: int
    y: int
    dx: int
    dy: int

    def __init__(self, x, y, dx, dy, **kwargs):
        super().__init__(
            x=x, y=y, dx=dx, dy=dy, type=InputType.MOTION, **kwargs
        )


class MouseDragEvent(_InputEvent):
    __slots__ = ["x", "y", "dx", "dy", "buttons"]

    x: int
    y: int
    dx: int
    dy: int
    buttons: Buttons

    def __init__(self, x, y, dx, dy, buttons, **kwargs):
        super().__init__(
            x=x,
            y=y,
            dx=dx,
            dy=dy,
            buttons=buttons,
            type=InputType.DRAG,
            **kwargs,
        )


class MouseScrollEvent(_InputEvent):
    __slots__ = ["dx", "dy"]

    dx: int
    dy: int

    def __init__(self, dx, dy, **kwargs):
        super().__init__(dx=dx, dy=dy, type=InputType.SCROLL, **kwargs)
