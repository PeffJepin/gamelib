from __future__ import annotations

import logging
import threading
import time
import warnings
from collections import defaultdict, namedtuple
from multiprocessing.connection import Connection
from typing import Type, List, Callable, Union, Tuple, Any

from moderngl_window.context.pygame2.keys import Keys

EventKey: Tuple[Type[Event], Any]
EventHandler: Callable[[Event], None]
ModifierKeys = namedtuple("KeyModifiers", "SHIFT, CTRL, ALT")  # Boolean values
MouseButtons = namedtuple("MouseButtons", "LEFT, RIGHT, MIDDLE")  # Boolean values

_HANDLER_INJECTION_ATTRIBUTE = "_gamelib_handler_"
_MOUSE_MAP = {"LEFT": 1, "RIGHT": 2, "MIDDLE": 3}


class _EventType(type):
    key_options: Union[set, dict, object, None]

    def __getattr__(cls, name: str):
        if name.startswith("_"):
            raise AttributeError(
                f"_EventType won't create keys with leading underscores. {name=}"
            )

        if cls.key_options is None:
            name = name

        elif isinstance(cls.key_options, set):
            if name not in cls.key_options:
                raise ValueError(f"Expected {name=} to be in {cls.key_options!r}")
            name = name

        elif isinstance(cls.key_options, dict):
            name = cls.key_options[name]

        else:
            name = getattr(cls.key_options, name)

        return cls, name


class Event(metaclass=_EventType):
    """
    An Event can have its type object queried for attributes to make keys.
    Note: conflicts with dataclass

    Examples
    --------
    With key_options = None (default)

    >>> Event.anything_that_could_be_a_valid_attr_name
    (KeyedEvent, 'anything_that_could_be_a_valid_attr_name')

    Keys should not start with a leading underscore.

    >>> Event._A  # Raises AttributeError.

    Choices for keys can be limited by a set.

    >>> class MyKeyedEvent(Event):
    ...     key_options = {'A', 'B', 'C'}
    ...
    >>> MyKeyedEvent.C
    (MyKeyedEvent, 'C')
    ...
    >>> MyKeyedEvent.H  # Raises ValueError

    Choices can also be limited with a dict or object via key/attribute lookup
    In this case keys are both limited and mapped based on values.

    >>> class MyKeyedEvent(Event):
    ...     key_options = {'A': 2, 'B': 4, 'C': 6}
    ...
    >>> MyKeyedEvent.A
    (MyKeyedEvent, 2)
    ...
    >>> MyKeyedEvent.Z  # raises KeyError
    """

    __slots__ = []
    key_options: Union[set, dict, object, None] = None

    def __init__(self, *args, **kwargs):
        """
        Default __init__: fills slots with either args or kwargs.

        Parameters
        ----------
        *args : Any
            As many args as there are slots. They will be filled in order. Don't mix with **kwargs
        **kwargs : Any
            Key value pairs map to slot names and values. Not to be used with *args.
        """

        if args and kwargs:
            raise ValueError("Default Event.__init__ shouldn't mix *args and **kwargs.")
        for slot, arg in zip(self.__slots__, args):
            setattr(self, slot, arg)
        for slot, arg in kwargs.items():
            setattr(self, slot, arg)

    def __eq__(self, other):
        """
        Compare as equal if other is of the same type as self
        and all attributes defined by __slots__ are the same.
        """
        if type(self) is type(other):
            self_slots = [(slot, getattr(self, slot)) for slot in self.__slots__]
            other_slots = [(slot, getattr(other, slot)) for slot in other.__slots__]
            return self_slots == other_slots


def handler(event_key):
    """
    Marks the decorated function object which can later be retrieved
    with the find_handlers function in this module.

    Parameters
    ----------
    event_key : tuple[type[Event], Any]
        The key value for an event is a tuple of the event type and an arbitrary key.
        Because of this each type of event can be subscribed to with an additional key.

    Examples
    --------
    How handler would be used with a normal event: event_key == (Update, None)

    >>> @handler(Update):
    >>> def update_handler_function(self, event: Update) -> None:
    ...     # do update
    ...     ...

    How handler would be used with a keyed event: event_key == (Event, 'ABC')

    >>> @handler(Event.ABC)
    >>> def some_keyed_event_handler(self, event: Event) -> None:
    ...     # Only called when a Event instance is posted with 'ABC' as a key.
    ...     # See Event for more detailed documentation on the key values.
    ...     ...
    """

    def inner(fn):
        if isinstance(event_key, type):
            setattr(fn, _HANDLER_INJECTION_ATTRIBUTE, (event_key, None))
        else:
            setattr(fn, _HANDLER_INJECTION_ATTRIBUTE, event_key)
        return fn

    return inner


def find_handlers(obj):
    """
    Helper function that finds all functions in an object's
    directory that have been marked by the handler decorator.

    Parameters
    ----------
    obj : object

    Returns
    -------
    handlers : dict[EventKey, list[EventHandler]]
        A dictionary mapping event keys to a list of event handler functions
    """

    handlers = defaultdict(list)
    for name in dir(obj):
        attr = getattr(obj, name, None)
        if (attr is None) or not isinstance(attr, Callable):
            continue
        if (event_type := getattr(attr, _HANDLER_INJECTION_ATTRIBUTE, None)) is None:
            # might be a bound method
            if (fn := getattr(attr, "__func__", None)) is not None:
                for k, v in find_handlers(fn).items():
                    handlers[k].extend(v)
            continue
        handlers[event_type].append(attr)
    return handlers


class MessageBus:
    def __init__(self, initial_handlers=None):
        """
        Parameters
        ----------
        initial_handlers : dict[EventKey, List[EventHandler]] | None
            Handlers can be registered after MessageBus construction or passed in this initial dict.
        """
        self.handlers = defaultdict(list)
        if initial_handlers:
            for event_type, callbacks in initial_handlers.items():
                if isinstance(event_type, type(Event)):
                    event_type = (event_type, None)
                self.handlers[event_type].extend(callbacks)
        self._adapters = dict()

    def register(self, event_key, callback) -> None:
        """
        Register a function as a callback for given event type.

        Parameters
        ----------
        event_key : type[Event] | tuple[type[Event], Any]
            A regular will be described by it's Type
            A KeyedEvent will be described by a tuple of it's Type and some key value.
        callback : EventHandler
        """
        if isinstance(event_key, type(Event)):
            event_key = (event_key, None)
        self.handlers[event_key].append(callback)

    def unregister(self, event_key, callback) -> None:
        """
        Stop handling this event type with this callback.

        Parameters
        ----------
        event_key : Type[Event]
        callback : EventHandler
        """
        try:
            if isinstance(event_key, type(Event)):
                event_key = (event_key, None)
            self.handlers[event_key].remove(callback)
        except ValueError:
            warnings.warn(
                f"Tried to unregister {callback!r} from {self!r} when it was not registered"
            )

    def post_event(self, event, key=None) -> None:
        """
        Calls all callbacks registered to the type of this event. This includes pushing
        the event out through currently serviced connections.

        Parameters
        ----------
        event : Event
            The event instance to be posted.
        key : Any
            Key to be used when posting a KeyedEvent
        """
        event_key = (type(event), key)
        for handler_ in self.handlers[event_key]:
            handler_(event)

    def service_connection(self, conn, event_keys) -> None:
        """
        MessageBus will send all events it publishes through a serviced connection.

        Parameters
        ----------
        conn : Connection
        event_keys : List[EventKey]
            List of EventKeys that should be fed through this pipe.

            If any of the keys are just a Type[Event] then they will be
            assigned None key by default.
        """
        for i, key in enumerate(event_keys):
            if not isinstance(key, tuple):
                # convert Type[Event] into EventKey
                event_keys[i] = (key, None)

        adapter = _ConnectionAdapter(self, conn, event_keys)
        for key in event_keys:
            self.register(key, adapter)
        self._adapters[conn] = adapter
        adapter.thread.start()

    def stop_connection_service(self, conn) -> None:
        """
        Stops the MessageBus from sending events through this connection.

        Parameters
        ----------
        conn : Connection
        """
        adapter = self._adapters.pop(conn, None)
        if adapter is None:
            return
        for type_ in adapter.event_types:
            self.unregister(type_, adapter)
        adapter.is_active = False


class Update(Event):
    pass


class SystemStop(Event):
    pass


class Quit(Event):
    pass


class _BaseKeyEvent(Event):
    key_options = Keys

    __slots__ = ["modifiers"]

    modifiers: ModifierKeys


class KeyDown(_BaseKeyEvent):
    pass


class KeyUp(_BaseKeyEvent):
    pass


class KeyIsPressed(_BaseKeyEvent):
    pass


class MouseDrag(Event):
    __slots__ = ["buttons", "x", "y", "dx", "dy"]

    buttons: MouseButtons
    x: int
    y: int
    dx: int
    dy: int


class MouseMotion(Event):
    __slots__ = ["x", "y", "dx", "dy"]

    x: int
    y: int
    dx: int
    dy: int


class MouseScroll(Event):
    __slots__ = ["dx", "dy"]

    dx: int
    dy: int


class _BaseMouseEvent(Event):
    key_options = _MOUSE_MAP

    __slots__ = ["x", "y"]

    x: int
    y: int


class MouseDown(_BaseMouseEvent):
    pass


class MouseUp(_BaseMouseEvent):
    pass


class MouseIsPressed(_BaseMouseEvent):
    pass


class _ConnectionAdapter:
    def __init__(
        self,
        mb: MessageBus,
        conn: Connection,
        event_types: List[Type[Event]],
        poll_freq: int = 1,
    ):
        self.mb = mb
        self.freq = poll_freq
        self.conn = conn
        self.event_types = event_types
        self.is_active = True
        self.thread = threading.Thread(target=self._listen, daemon=True)

    def _listen(self):
        try:
            while self.is_active:
                while self.conn.poll(0):
                    self.mb.post_event(self.conn.recv())
                time.sleep(self.freq / 1_000)
        except Exception as e:
            logging.debug(
                f"Exception occurred in {self.__class__.__name__} pipe listener thread.",
                exc_info=e,
            )
            self.is_active = False
            self.mb.stop_connection_service(self.conn)

    def __call__(self, event):
        self.conn.send(event)
