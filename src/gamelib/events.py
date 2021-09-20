import logging
import threading
import time
import warnings
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Type, List, Callable, Union

from moderngl_window.context.pygame2.keys import Keys

_HANDLER_INJECTION_ATTRIBUTE = "_gamelib_handler_"
_MOUSE_MAP = {"LEFT": 1, "RIGHT": 2, "MIDDLE": 3}


@dataclass(frozen=True)
class Event:
    pass


EventHandler = Callable[[Event], None]
EventKey = Union[Type[Event], tuple]
ModifierKeys = namedtuple("KeyModifiers", "SHIFT, CTRL, ALT")  # True/False values
MouseButtons = namedtuple("MouseButtons", "LEFT, RIGHT, MIDDLE")  # True/False values


class _KeyedEventType(type):
    choices: Union[set, dict, object, None]

    def __getattr__(cls, name: str):
        if name.startswith("_"):
            raise AttributeError(
                f"_KeyedEventType won't create keys starting with '_'."
            )

        if cls.choices is None:
            key = name

        elif isinstance(cls.choices, set):
            if name not in cls.choices:
                raise ValueError(f"Expected {name=} to be in {cls.choices!r}")
            key = name

        elif isinstance(cls.choices, dict):
            key = cls.choices[name]

        else:
            key = getattr(cls.choices, name)

        return cls, key


class KeyedEvent(Event, metaclass=_KeyedEventType):
    """
    A KeyedEvent can have its type object queried for attributes to make keys.

    Examples
    --------
    With choices = None (default)

    >>> KeyedEvent.anything_that_could_be_a_valid_attr_name
    (KeyedEvent, 'anything_that_could_be_a_valid_attr_name')

    Keys should not start with a leading underscore.

    >>> KeyedEvent._A  # Raises AttributeError.

    Choices for keys can be limited by a set.

    >>> class MyKeyedEvent(KeyedEvent):
    ...     choices = {'A', 'B', 'C'}
    ...
    >>> MyKeyedEvent.C
    (MyKeyedEvent, 'C')
    ...
    >>> MyKeyedEvent.H  # Raises ValueError

    Choices can also be limited with a dict or object via key/attribute lookup
    In this case keys are both limited and mapped based on values.

    >>> class MyKeyedEvent(KeyedEvent):
    ...     choices = {'A': 2, 'B': 4, 'C': 6}
    ...
    >>> MyKeyedEvent.A
    (MyKeyedEvent, 2)
    ...
    >>> MyKeyedEvent.Z  # raises KeyError
    """

    choices: Union[set, dict, object, None] = None


class Update(Event):
    pass


class SystemStop(Event):
    pass


class Quit(Event):
    pass


@dataclass(frozen=True)
class KeyDown(KeyedEvent):
    choices = Keys
    modifiers: ModifierKeys


@dataclass(frozen=True)
class KeyUp(KeyedEvent):
    choices = Keys
    modifiers: ModifierKeys


@dataclass(frozen=True)
class KeyIsPressed(KeyedEvent):
    choices = Keys
    modifiers: ModifierKeys


@dataclass(frozen=True)
class MouseDrag(Event):
    buttons: MouseButtons
    x: int
    y: int
    dx: int
    dy: int


@dataclass(frozen=True)
class MouseMotion(Event):
    x: int
    y: int
    dx: int
    dy: int


@dataclass(frozen=True)
class MouseScroll(Event):
    dx: int
    dy: int


class MouseDown(KeyedEvent):
    choices = _MOUSE_MAP


class MouseUp(KeyedEvent):
    choices = _MOUSE_MAP


class MouseIsPressed(KeyedEvent):
    choices = _MOUSE_MAP


def handler(event_key):
    """
    Marks the decorated function object which can later be retrieved
    with the find_handlers function in this module.

    Parameters
    ----------
    event_key : type[Event] | tuple[type[Event], Any]
        An Event is described by it's Type
        A KeyedEvent is described by a tuple of it's Type and some key value.

    Examples
    --------
    How handler would be used with a normal event.

    >>> @handler(Update):
    >>> def update_handler_function(self, event: Update) -> None:
    ...     # do update
    ...     ...

    How handler would be used with a keyed event.

    >>> @handler(KeyedEvent.ABC)
    >>> def some_keyed_event_handler(self, event: KeyedEvent) -> None:
    ...     # Only called when a KeyedEvent instance is posted with 'ABC' as a key.
    ...     # See KeyedEvent for more detailed documentation on the key values.
    ...     ...
    """

    def inner(fn):
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
        if isinstance(event, KeyedEvent) and key is None:
            raise ValueError(
                f"{event!r} was expected to be posted with a key, but {key=}"
            )
        event_key = type(event) if key is None else (type(event), key)
        for handler_ in self.handlers[event_key]:
            handler_(event)

    def service_connection(self, conn, event_keys) -> None:
        """
        MessageBus will send all events it publishes through a serviced connection.

        Parameters
        ----------
        conn : Connection
        event_keys : List[Type[Event]]
            List of event types that should be piped through.
        """
        adapter = _ConnectionAdapter(self, conn, event_keys)
        for type_ in event_keys:
            self.register(type_, adapter)
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
