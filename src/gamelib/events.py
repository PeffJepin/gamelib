from __future__ import annotations

import inspect
import threading
import warnings
from collections import defaultdict
from multiprocessing.connection import Connection
from typing import Type, List, Callable, Union, Tuple, Any

EventKey: Tuple[Type[BaseEvent], Any]
EventHandler: Callable[[BaseEvent], None]
_HANDLER_INJECTION_ATTRIBUTE = "_gamelib_handler_"


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


class BaseEvent(metaclass=_EventType):
    """
    An Event can have its type object queried for attributes to make keys.
    Note: conflicts with dataclass

    Examples
    --------
    With key_options = None (default)

    >>> BaseEvent.anything_that_could_be_a_valid_attr_name
    (KeyedEvent, 'anything_that_could_be_a_valid_attr_name')

    Keys should not start with a leading underscore.

    >>> BaseEvent._A  # Raises AttributeError.

    Choices for keys can be limited by a set.

    >>> class MyKeyedEvent(BaseEvent):
    ...     key_options = {'A', 'B', 'C'}
    ...
    >>> MyKeyedEvent.C
    (MyKeyedEvent, 'C')
    ...
    >>> MyKeyedEvent.H  # Raises ValueError

    Choices can also be limited with a dict or object via key/attribute lookup
    In this case keys are both limited and mapped based on values.

    >>> class MyKeyedEvent(BaseEvent):
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


def eventhandler(event_key):
    """
    Marks the decorated function object which can later be retrieved
    with the find_handlers function in this module.

    Parameters
    ----------
    event_key : tuple[type[BaseEvent], Any]
        The key value for an event is a tuple of the event type and an arbitrary key.
        Because of this each type of event can be subscribed to with an additional key.

    Examples
    --------
    How eventhandler would be used with a normal event: event_key == (Event, None)

    >>> @eventhandler(BaseEvent):
    >>> def update_handler_function(self, event: BaseEvent) -> None:
    ...     # do update
    ...     ...

    How eventhandler would be used with a keyed event: event_key == (Event, 'ABC')

    >>> @eventhandler(BaseEvent.ABC)
    >>> def some_keyed_event_handler(self, event: BaseEvent) -> None:
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
    directory that have been marked by the eventhandler decorator.

    Parameters
    ----------
    obj : object

    Returns
    -------
    handlers : dict[EventKey, list[EventHandler]]
        A dictionary mapping event keys to a list of event handler functions
    """

    handlers = defaultdict(list)
    for name, attr in inspect.getmembers(obj):
        if not isinstance(attr, Callable):
            continue
        if (event_type := getattr(attr, _HANDLER_INJECTION_ATTRIBUTE, None)) is None:
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
                if isinstance(event_type, type(BaseEvent)):
                    event_type = (event_type, None)
                self.handlers[event_type].extend(callbacks)
        self._adapters = dict()

    def register(self, event_key, *callbacks) -> None:
        """
        Register a function as a callback for given event type.

        Parameters
        ----------
        event_key : type[BaseEvent] | tuple[type[BaseEvent], Any]
            A regular will be described by it's Type
            A KeyedEvent will be described by a tuple of it's Type and some key value.
        *callbacks : EventHandler
        """
        if isinstance(event_key, type(BaseEvent)):
            event_key = (event_key, None)
        self.handlers[event_key].extend(callbacks)

    def unregister(self, event_key, *callbacks) -> None:
        """
        Stop handling this event type with this callback.

        Parameters
        ----------
        event_key : Type[BaseEvent]
        *callbacks : EventHandler
        """
        if isinstance(event_key, type(BaseEvent)):
            event_key = (event_key, None)
        for callback in callbacks:
            try:
                self.handlers[event_key].remove(callback)
            except ValueError:
                warnings.warn(
                    f"Tried to unregister {callback!r} from {self!r} when it was not registered"
                )

    def register_marked_handlers(self, obj):
        """
        Shorthand for finding the @handler docorated functions on an object
        and registering them as callbacks.

        Parameters
        ----------
        obj : object
            Object containing @handler marked functions.
        """
        for event_key, handlers in find_handlers(obj).items():
            self.register(event_key, *handlers)

    def unregister_marked_handlers(self, obj):
        """
        Shorthand for finding @handler decorated functions and
        unregistering them as callbacks.

        Parameters
        ----------
        obj : object
            Object containing @handler marked functions
        """
        for event_key, handlers in find_handlers(obj).items():
            self.unregister(event_key, *handlers)

    def post_event(self, event, key=None) -> None:
        """
        Calls all callbacks registered to the type of this event. This includes pushing
        the event out through currently serviced connections.

        Parameters
        ----------
        event : BaseEvent
            The event instance to be posted.
        key : Any
            Key to be used when posting a KeyedEvent
        """
        event_key = (type(event), key)
        for handler in self.handlers[event_key]:
            handler(event)

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
        adapter.stop()


class _ConnectionAdapter:
    def __init__(
        self,
        mb: MessageBus,
        conn: Connection,
        event_keys: List[EventKey],
    ):
        self.mb = mb
        self.conn = conn
        self.event_types = event_keys
        self._running = True
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        try:
            while self._running:
                if self.conn.poll(0.001):
                    self.mb.post_event(self.conn.recv())
        except (BrokenPipeError, EOFError):
            self._running = False

    def stop(self):
        self._running = False

    def __call__(self, event):
        self.conn.send(event)
