from collections import defaultdict
from enum import Enum
from dataclasses import dataclass
from typing import NamedTuple

from gamelib import events


class _StringMappingEnum(Enum):
    @classmethod
    def map_string(cls, string):
        for member in cls:
            if string.lower() in member.value:
                return member

    def __repr__(self):
        return f"<{self.__class__.__name__}.{self.name}>"

    def __eq__(self, other):
        if isinstance(other, str):
            return other.lower() in self.value
        elif isinstance(other, type(self)):
            return other.name == self.name
        else:
            return False

    def __hash__(self):
        return hash(self.name)


class Keyboard(_StringMappingEnum):
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


class MouseButton(_StringMappingEnum):
    LEFT = ("mouse1", "mouse_1", "mouse_left")
    RIGHT = ("mouse2", "mouse_2", "mouse_right")
    MIDDLE = ("mouse3", "mouse_3", "mouse_middle")


class Mouse(_StringMappingEnum):
    MOTION = ("mouse_motion", "mouse_movement", "motion")
    DRAG = ("mouse_drag", "drag")
    SCROLL = ("scroll", "mouse_scroll", "wheel")


class Modifier(_StringMappingEnum):
    SHIFT = ("shift", "mod1")
    CTRL = ("control", "ctrl", "mod2")
    ALT = ("alt", "mod3")


class Action(_StringMappingEnum):
    PRESS = ("press", "down", "on_press", "on_down")
    RELEASE = ("release", "up", "on_release", "on_up")
    IS_PRESSED = ("pressed", "ispressed", "isdown", "is_pressed", "is_down")


class Modifiers(NamedTuple):
    SHIFT: bool = False
    CTRL: bool = False
    ALT: bool = False


class Buttons(NamedTuple):
    LEFT: bool = False
    RIGHT: bool = False
    MIDDLE: bool = False


@dataclass
class _InputEvent:
    pass


@dataclass
class KeyDown(_InputEvent):
    key: Keyboard
    modifiers: Modifiers


@dataclass
class KeyUp(_InputEvent):
    key: Keyboard
    modifiers: Modifiers


@dataclass
class KeyIsPressed(_InputEvent):
    key: Keyboard
    modifiers: Modifiers


@dataclass
class MouseDown(_InputEvent):
    x: int
    y: int
    button: MouseButton


@dataclass
class MouseUp(_InputEvent):
    x: int
    y: int
    button: MouseButton


@dataclass
class MouseIsPressed(_InputEvent):
    x: int
    y: int
    button: MouseButton


@dataclass
class MouseMotion(_InputEvent):
    x: int
    y: int
    dx: int
    dy: int


@dataclass
class MouseDrag(_InputEvent):
    x: int
    y: int
    dx: int
    dy: int
    buttons: Buttons


@dataclass
class MouseScroll(_InputEvent):
    dx: int
    dy: int


class InputSchema:
    def __init__(self, *schema):
        self._callback_tree = _InputHandlerLookup()
        self._process_schema(*schema)
        self.enable()

    def enable(self, *, master=False):
        if master:
            events.clear_handlers(*self._events)
        for event in self._events:
            events.register(event, self)

    def disable(self):
        for event in self._events:
            events.unregister(event, self)

    @property
    def _events(self):
        return _InputEvent.__subclasses__()

    def _process_schema(self, *schema):
        for desc in schema:
            input_type, *optional, callback = desc

            if isinstance(input_type, str):
                enum = Keyboard.map_string(input_type)
                enum = enum or MouseButton.map_string(input_type)
                enum = enum or Mouse.map_string(input_type)
                input_type = enum

            mods = []
            action = None
            for arg in optional:
                # str might be mod or action
                if isinstance(arg, str):
                    parse_action = Action.map_string(arg)
                    if parse_action:
                        action = parse_action
                    else:
                        mods.append(arg)

                # list/tuple are mods
                elif isinstance(arg, list) or isinstance(arg, tuple):
                    for mod in arg:
                        mods.append(mod)

                elif isinstance(arg, Modifier):
                    mods.append(arg)

                elif isinstance(arg, Action):
                    action = arg

            mods = Modifiers(
                Modifier.SHIFT in mods,
                Modifier.CTRL in mods,
                Modifier.ALT in mods,
            )
            self._callback_tree.register(
                callback, input_type, modifiers=mods, action=action
            )

    def __call__(self, event):
        callback = self._callback_tree.get_callback(event)
        if not callback:
            return
        callback(event)


class _InputHandlerLookup:
    def __init__(self):
        self._lookup = {
            KeyDown: defaultdict(dict),
            KeyUp: defaultdict(dict),
            KeyIsPressed: defaultdict(dict),
            MouseDown: dict(),
            MouseUp: dict(),
            MouseIsPressed: dict(),
            MouseDrag: None,
            MouseScroll: None,
            MouseMotion: None,
        }

    def register(
        self, callback, input_enum, modifiers=Modifiers(), action=None
    ):
        if input_enum in Keyboard:
            if action is None:
                action = Action.PRESS
            if action == Action.PRESS:
                self._lookup[KeyDown][input_enum][modifiers] = callback
            elif action == Action.RELEASE:
                self._lookup[KeyUp][input_enum][modifiers] = callback
            elif action == Action.IS_PRESSED:
                self._lookup[KeyIsPressed][input_enum][modifiers] = callback

        elif input_enum in MouseButton:
            if action is None:
                action = Action.PRESS
            if action == Action.PRESS:
                self._lookup[MouseDown][input_enum] = callback
            elif action == Action.RELEASE:
                self._lookup[MouseUp][input_enum] = callback
            elif action == Action.IS_PRESSED:
                self._lookup[MouseIsPressed][input_enum] = callback

        else:
            if input_enum == Mouse.MOTION:
                self._lookup[MouseMotion] = callback
            elif input_enum == Mouse.DRAG:
                self._lookup[MouseDrag] = callback
            elif input_enum == Mouse.SCROLL:
                self._lookup[MouseScroll] = callback

    def get_callback(self, event):
        event_type = type(event)
        if not self._lookup.get(event_type):
            return

        enum = getattr(event, "key", None) or getattr(event, "button", None)
        if enum:
            modifiers = getattr(event, "modifiers", None)
            if modifiers is not None:
                return self._lookup[event_type][enum].get(modifiers)
            return self._lookup[event_type].get(enum)
        return self._lookup[event_type]
