"""This module in it's current form is basically an extension to the events.py
module. The aim is to provide a quick, easy way to get user input integrated
into an application.

The only component needed to get user input integrated is InputSchema. The
input events can be selected with python strings, or if more verbose
declarations are desired the Enum classes Keyboard, Mouse, and MouseButton can
be used.

Example
-------
A barebones example - see documentation below for further detail.

>>> def attack(event):
...     # perform attack
...     ...
...
>>> def jump():
...     # taking event as parameter is optional
...     ...
...
>>> schema = InputSchema(
...     ("mouse1", "press", attack),
...     ("space", "press", jump),
... )
>>> # schema is now active and will handle input events posted by the window.
...
>>> schema.disable()
>>> # schema will no longer handle events
...
>>> schema.enable(master=True)
>>> # multiple schemas can be active at once
>>> # master option means this will be the only schema processing input
"""
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass
from typing import NamedTuple

from gamelib import events

_key_states_to_monitor_lookup = dict()
monitored_key_states = set()


class InputSchema:
    """InputSchema serves as both an adapter to the events module for
    user input, and defines the format in which user input mappings are
    declared.

    The window is responsible for collecting the user input and posting
    events, so this wont work with gamelib.init(headless=True).

    Multiple InputSchema objects can be active at once, as such a single
    input event can be consumed by multiple InputSchema instances, though
    each instance can only hold one callback for a particular event mapping.
    """

    def __init__(self, *schema, enable=True):
        """Map different types of user input to callback functions.
        The schema will immediately begin handling input events that get
        posted by the window.

        Parameters
        ----------
        schema: tuple
            The general format for an input handler looks like:
                (input_type, *optional, callback)

            input_type
                should map to either Keyboard.*, MouseButton.* or Mouse.*
            *optional
                args should define action/modifiers where applicable.
            callback
                the function that will handle the event described.

        enable : bool, optional
            Optional flag - Should this schema be enabled on __init__ ?

        Example
        -------
        >>> def handler(event):
        ...     # taking event as parameter is optional
        ...     ...
        ...
        >>> input_schema = InputSchema(
        ...     # MouseDown event. Modifiers not applicable
        ...     ("mouse1", "press", handler),
        ...
        ...     # MouseUp event. Modifiers not applicable.
        ...     ("mouse2", "release", handler),
        ...
        ...     # MouseMotion event. Actions and modifiers not applicable.
        ...     ("motion", handler),
        ...
        ...     # MouseDrag event. Actions and modifiers not applicable.
        ...     ("drag", handler),
        ...
        ...     # When applicable, the default action is Action.PRESS.
        ...     ("a", handler),  # KeyDown, Keyboard.A
        ...     ("mouse3", handler),  # MouseDown, MouseButton.MIDDLE
        ...
        ...     # Modifiers can be grouped into an iterable for clarity
        ...     ("a", "release", ("shift", "ctrl"), handler),
        ...
        ...     # Or modifiers can just be given as *args. Order irrelevant.
        ...     ("a", "release", "shift", "ctrl", handler),
        ...
        ...     # KeyIsPressed event
        ...     ("space", "is_pressed", handler),
        ...
        ...     # Descriptions can be made more verbose with module Enums.
        ...     (Keyboard.B, Action.PRESS, Modifier.ALT, handler),
        ...     (MouseButton.LEFT, Action.RELEASE, handler),
        ...     (Mouse.MOTION, handler),
        ... )
        """

        self._callback_tree = _InputHandlerLookup()
        self._process_schema(*schema)
        if enable:
            self.enable()

    def enable(self, *, master=False):
        """Subscribes this Schema to handle input events posted to the
        events module.

        Parameters
        ----------
        master : bool, optional
            If True, then this will clear all other input handlers
            subscribed to handle input events, such that this instance
            will be the only remaining event handler.

            This may be inadvisable if you're handling any _InputEvent
            types directly through the events.py machinery, as this will
            remove those handlers as well, not just InputSchemas.
        """

        if master:
            events.clear_handlers(*self._events)
        for event in self._events:
            events.subscribe(event, self)
        _key_states_to_monitor_lookup[self] = set(
            self._callback_tree.key_is_pressed_types
        )
        _update_monitored_key_states()

    def disable(self):
        """Stop this instance from handling input events."""

        for event in self._events:
            events.unsubscribe(event, self)
        del _key_states_to_monitor_lookup[self]
        _update_monitored_key_states()

    @property
    def _events(self):
        return _InputEvent.__subclasses__()

    def _process_schema(self, *schema):
        """Processes the schema passed to __init__."""

        for desc in schema:
            input_type, *optional, callback = desc

            if isinstance(input_type, str):
                input_type = (
                    Keyboard.map_string(input_type)
                    or MouseButton.map_string(input_type)
                    or Mouse.map_string(input_type)
                )

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

            action = action or Action.PRESS
            mods = Modifiers(
                Modifier.SHIFT in mods,
                Modifier.CTRL in mods,
                Modifier.ALT in mods,
            )
            self._callback_tree.register(
                callback, input_type, modifiers=mods, action=action
            )

    def __call__(self, event):
        """The event module will call this object as if it were an event
        handler itself.

        Instead lookup the appropriate handler and forward the event there.
        """

        callback = self._callback_tree.get_callback(event)
        if not callback:
            return

        try:
            callback(event)
        except TypeError:
            callback()


def _update_monitored_key_states():
    global monitored_key_states
    sets = tuple(_key_states_to_monitor_lookup.values())
    if len(sets) == 1:
        monitored_key_states = sets[0]
    else:
        monitored_key_states = set.union(*sets)


class _StringMappingEnum(Enum):
    """Internal class for managing the mapping of many possible strings to
    an Enum field."""

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
    """Defines string mappings for Mouse clicking events."""

    LEFT = ("mouse1", "mouse_1", "mouse_left")
    RIGHT = ("mouse2", "mouse_2", "mouse_right")
    MIDDLE = ("mouse3", "mouse_3", "mouse_middle")


class Mouse(_StringMappingEnum):
    """Defines string mappings for non clicking events."""

    MOTION = ("mouse_motion", "mouse_movement", "motion")
    DRAG = ("mouse_drag", "drag")
    SCROLL = ("scroll", "mouse_scroll", "wheel")


class Modifier(_StringMappingEnum):
    """Defines string mappings for modifier keys."""

    SHIFT = ("shift", "mod1")
    CTRL = ("control", "ctrl", "mod2")
    ALT = ("alt", "mod3")


class Action(_StringMappingEnum):
    """Defines string mappings for different input action types."""

    PRESS = ("press", "down", "on_press", "on_down")
    RELEASE = ("release", "up", "on_release", "on_up")
    IS_PRESSED = ("pressed", "is_pressed", "is_down")


class Modifiers(NamedTuple):
    """Tuple passed around with input events that care about Modifier keys."""

    SHIFT: bool = False
    CTRL: bool = False
    ALT: bool = False


class Buttons(NamedTuple):
    """Tuple passed around with input events that care about MouseButton
    state."""

    LEFT: bool = False
    RIGHT: bool = False
    MIDDLE: bool = False


@dataclass
class _InputEvent:
    """Base class for convenience."""

    pass


@dataclass
class KeyDown(_InputEvent):
    """Posted from window provider."""

    key: Keyboard
    modifiers: Modifiers


@dataclass
class KeyUp(_InputEvent):
    """Posted from window provider."""

    key: Keyboard
    modifiers: Modifiers


@dataclass
class KeyIsPressed(_InputEvent):
    """Key state is extracted once per frame for this event."""

    key: Keyboard
    modifiers: Modifiers


@dataclass
class MouseDown(_InputEvent):
    """Posted from window provider."""

    x: int
    y: int
    button: MouseButton


@dataclass
class MouseUp(_InputEvent):
    """Posted from window provider."""

    x: int
    y: int
    button: MouseButton


@dataclass
class MouseIsPressed(_InputEvent):
    """Key state is extracted once per tick for this event."""

    x: int
    y: int
    button: MouseButton


@dataclass
class MouseMotion(_InputEvent):
    """Posted from window provider."""

    x: int
    y: int
    dx: int
    dy: int


@dataclass
class MouseDrag(_InputEvent):
    """Posted from window provider."""

    x: int
    y: int
    dx: int
    dy: int
    buttons: Buttons


@dataclass
class MouseScroll(_InputEvent):
    """Posted from window provider. dx not always applicable."""

    dx: int
    dy: int


class _InputHandlerLookup:
    """Internal class that organizes the handlers for an InputSchema. Some
    events have nested lookups since their lookup depends on Action,
    Modifiers or both. May want to consider hashing the nested lookups in
    the future."""

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
        self, callback, input_enum, modifiers=Modifiers(), action=Action.PRESS
    ):
        if input_enum in Keyboard:
            if action == Action.PRESS:
                self._lookup[KeyDown][input_enum][modifiers] = callback
            elif action == Action.RELEASE:
                self._lookup[KeyUp][input_enum][modifiers] = callback
            elif action == Action.IS_PRESSED:
                self._lookup[KeyIsPressed][input_enum][modifiers] = callback

        elif input_enum in MouseButton:
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
        """ "Try to get registered callback for this event.

        Returns
        -------
        Callable | None:
            depending on if a handler has been registered for this event.
        """

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

    @property
    def mouse_is_pressed_types(self):
        return self._lookup[MouseIsPressed].keys()

    @property
    def key_is_pressed_types(self):
        return self._lookup[KeyIsPressed].keys()
