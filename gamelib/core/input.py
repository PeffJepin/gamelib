"""This module in it's current form is basically an extension to the events.py
module. The aim is to provide a quick, easy way to get user input integrated
into an application.

The only component needed to get user input integrated is InputSchema. The
input events can be selected with python strings, or if more verbose
declarations are desired the enum.Enum classes Keyboard, Mouse, and MouseButton
can be used.

Example
-------

A barebones InputSchema example - see documentation below for further detail.
InputSchema is best used to map specific keystrokes to specific functions.

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

Similar to the events.py module, there is an API for marking event handling
functions with decorators. This is preferable to using InputSchema if you want
to have one function handle an entire class of event, rather than delegating
based off of keystroke.

>>> class ExampleController:
...     @KeyDown.handler
...     def handle_entire_keydown_event(self, event):
...         # this handles fires for every key type
...         pass
...
...     @KeyIsPressed.handler(iter("ASDW"))
...     def handle_a_range_of_keys(self, event):
...         # you can still narrow down which keys are being watched.
...         pass
...
>>> # The ExampleController class must be enabled to start handling events.
>>> controller = ExampleController()
>>> enable_handlers(controller)
...
>>> # Disable the controller like so:
>>> disable_handlers(controller)
"""
# TODO: Eventually the StringMappingEnum class will need to be implemented
#   in such a way that the values mapped to the enums are simply ints and the
#   string matching lookup can be stored in some dictionary or something.
#   It's not really a problem now, but if these events have to start getting
#   sent across connections I want to be sure they're not taking up a bunch
#   of unnecessary space.

import collections
import dataclasses
import enum
from typing import NamedTuple
from typing import Callable
from typing import Iterable

from gamelib.core import events
from gamelib import utils

_key_states_to_monitor_lookup = dict()
_decorated_schemas = dict()
monitored_key_states = set()


class InputSchema:
    """InputSchema serves as both an adapter to the events module for
    input events, and defines a format in which user input mappings can be
    declared.

    The window is responsible for collecting the user input and posting
    events, so this will not work with gamelib.init(headless=True).

    Multiple InputSchema objects can be active at once, as such, a single
    input event can be consumed by multiple InputSchema instances, though
    each instance can only hold one callback for a particular event mapping.
    """

    def __init__(self, *schema, enable=True):
        """Map different types of user input to callback functions.
        The schema will immediately begin handling input events that get
        posted by the window unless the `enable` flag is set to False.

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
            Should this schema be enabled on __init__ ?

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
        ...     # Descriptions can be made more verbose with module enum.Enums.
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
        """Subscribes this InputSchema to handle input events posted to the
        events module.

        Parameters
        ----------
        master : bool, optional
            If True, then this will clear all other input handlers
            subscribed to handle input events, such that this instance
            will be the only remaining event handler.

            Note that this will also clear event handlers defined using the
            _InputEvent.handler API.
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
        try:
            del _key_states_to_monitor_lookup[self]
        except KeyError:
            # should be safe to disable if not enabled
            pass
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

        Instead, lookup the appropriate handler and forward the event there.
        """

        callback = self._callback_tree.get_callback(event)
        if not callback:
            return

        try:
            callback(event)
        except TypeError as e:
            if "positional argument" in e.args[0]:
                # callback without passing event as an argument
                callback()
            else:
                raise e


class _StringMappingEnum(enum.Enum):
    """Internal class for managing the mapping of many possible strings to
    an enum.Enum field."""

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

    shift: bool = False
    ctrl: bool = False
    alt: bool = False


class Buttons(NamedTuple):
    """Tuple passed around with input events that care about MouseButton
    state."""

    left: bool = False
    right: bool = False
    middle: bool = False


class _InputHandlerTag(NamedTuple):
    """Data involved with marking a method as an InputEvent handler."""

    event_type: type
    enums: tuple


def enable_handlers(obj):
    """Given an object with methods marked by the _InputEvent.handler API,
    enable all the marked handlers.

    Using marked handlers is more suitable for a case where you want one
    function to handle many different keystrokes, where using the InputSchema
    API is more suited towards delegating many individual keystrokes to many
    individual functions.

    Parameters
    ----------
    obj : object
        The marked object.

    Examples
    --------
    A basic example. See tests for many more.

    >>> class MyController:
    ...     @KeyDown.handler
    ...     def handle_key_down(self, event):
    ...         pass
    ...
    >>> obj = MyController()
    >>> enable_handlers(obj)
    >>> # obj will now handle KeyDown events.
    """

    if obj in _decorated_schemas:
        schema = _decorated_schemas[obj]
    else:
        marked_handlers = utils.MethodMarker.lookup(obj, type="input")
        schema = _DecoratedInputSchema(
            *[
                (mark.extra, handler)
                for mark, handler in marked_handlers.items()
            ]
        )
        _decorated_schemas[obj] = schema
    schema.enable()


def disable_handlers(obj):
    """Disables an object previously enabled with the `enable_handlers`
    function. Safe to call even when obj isn't enabled.

    Parameters
    ----------
    obj : object
        An instance of a class marked with _InputEvent.handler decorators.
    """

    if obj not in _decorated_schemas:
        return
    schema = _decorated_schemas.pop(obj)
    schema.disable()


@dataclasses.dataclass
class _InputEvent:
    ENUM = None
    ACTION = None

    @classmethod
    def handler(cls, input_enums=None):
        """See `enable_handlers` or the module docstring for example usage."""

        func = None

        # might get called from _InputEvent.handler like:
        #   @_InputEvent.handler
        #   def handler(event):
        # in which case input_enums is actually the `handler` function.
        if isinstance(input_enums, Callable):
            func, input_enums = input_enums, None

        if not input_enums and cls.ENUM != Mouse:
            enums = tuple(cls.ENUM)
        elif isinstance(input_enums, str):
            enums = (cls.ENUM.map_string(input_enums),)
        elif isinstance(input_enums, Keyboard):
            enums = (input_enums,)
        elif isinstance(input_enums, Iterable):
            enums = tuple(
                e if isinstance(e, cls.ENUM) else cls.ENUM.map_string(e)
                for e in input_enums
            )
        else:
            enums = None

        tag = _InputHandlerTag(cls, enums)
        return utils.MethodMarker(func, type="input", extra=tag)


@dataclasses.dataclass
class KeyDown(_InputEvent):
    """Posted from window provider."""

    ENUM = Keyboard
    ACTION = Action.PRESS

    key: Keyboard
    modifiers: Modifiers


@dataclasses.dataclass
class KeyUp(_InputEvent):
    """Posted from window provider."""

    ENUM = Keyboard
    ACTION = Action.RELEASE

    key: Keyboard
    modifiers: Modifiers


@dataclasses.dataclass
class KeyIsPressed(_InputEvent):
    """Key state is extracted once per frame for this event."""

    ENUM = Keyboard
    ACTION = Action.IS_PRESSED

    key: Keyboard
    modifiers: Modifiers
    dt: float = 0


@dataclasses.dataclass
class MouseDown(_InputEvent):
    """Posted from window provider."""

    ENUM = MouseButton
    ACTION = Action.PRESS

    x: int
    y: int
    button: MouseButton


@dataclasses.dataclass
class MouseUp(_InputEvent):
    """Posted from window provider."""

    ENUM = MouseButton
    ACTION = Action.RELEASE

    x: int
    y: int
    button: MouseButton


@dataclasses.dataclass
class MouseIsPressed(_InputEvent):
    """Key state is extracted once per tick for this event."""

    ENUM = MouseButton
    ACTION = Action.IS_PRESSED

    x: int
    y: int
    button: MouseButton
    dt: float = 0


@dataclasses.dataclass
class MouseMotion(_InputEvent):
    """Posted from window provider."""

    ENUM = Mouse

    x: int
    y: int
    dx: int
    dy: int


@dataclasses.dataclass
class MouseDrag(_InputEvent):
    """Posted from window provider."""

    ENUM = Mouse

    x: int
    y: int
    dx: int
    dy: int
    buttons: Buttons


@dataclasses.dataclass
class MouseScroll(_InputEvent):
    """Posted from window provider. dx not always applicable."""

    ENUM = Mouse

    dx: int
    dy: int


class _InputHandlerLookup:
    """Internal class that organizes the handlers for an InputSchema. Some
    events have nested lookups since their lookup depends on Action,
    Modifiers or both. May want to consider hashing the nested lookups in
    the future."""

    def __init__(self):
        self._lookup = {
            KeyDown: collections.defaultdict(dict),
            KeyUp: collections.defaultdict(dict),
            KeyIsPressed: collections.defaultdict(dict),
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
        """Try to get registered callback for this event.

        Returns
        -------
        Callable | None:
            depending on if a handler has been registered for this event.
        """

        event_type = type(event)
        if not self._lookup.get(event_type):
            return

        enum_ = getattr(event, "key", None) or getattr(event, "button", None)
        if enum_:
            modifiers = getattr(event, "modifiers", None)
            if modifiers is not None:
                return self._lookup[event_type][enum_].get(modifiers)
            return self._lookup[event_type].get(enum_)
        return self._lookup[event_type]

    @property
    def mouse_is_pressed_types(self):
        return self._lookup[MouseIsPressed].keys()

    @property
    def key_is_pressed_types(self):
        return self._lookup[KeyIsPressed].keys()


class _DecoratedInputSchema:
    """Used internally for managing input handlers marked by the
    _InputEvent.handler decorator.

    Like InputSchema, but instead of focusing on delegating events to
    specific callbacks based on event attributes, this implementation is
    focused on delegating many different keystrokes all to the same handler.
    """

    def __init__(self, *schema):
        self._handler_lookup = collections.defaultdict(dict)
        for (tag, handler) in schema:
            if tag.enums is None:
                self._handler_lookup[tag.event_type][None] = handler
            else:
                for e in tag.enums:
                    self._handler_lookup[tag.event_type][e] = handler

    def __call__(self, event):
        if type(event) not in self._handler_lookup:
            return

        key = getattr(event, "key", None)
        button = getattr(event, "button", None)
        enum_ = key or button
        callback = self._handler_lookup[type(event)].get(enum_)

        if not callback:
            return
        try:
            callback(event)
        except TypeError as e:
            if "required positional argument" in e.args[0]:
                callback()
            else:
                raise e

    def enable(self):
        for event_type in self._handler_lookup.keys():
            events.subscribe(event_type, self)

        if handlers := self._handler_lookup.get(KeyIsPressed):
            _key_states_to_monitor_lookup[self] = set(handlers.keys())
            _update_monitored_key_states()

    def disable(self):
        for event_type in self._handler_lookup.keys():
            events.unsubscribe(event_type, self)

        if self._handler_lookup.get(KeyIsPressed):
            del _key_states_to_monitor_lookup[self]
            _update_monitored_key_states()


def _update_monitored_key_states():
    global monitored_key_states
    sets = tuple(_key_states_to_monitor_lookup.values())
    if not sets:
        monitored_key_states = set()
    elif len(sets) == 1:
        monitored_key_states = sets[0]
    else:
        monitored_key_states = set.union(*sets)
