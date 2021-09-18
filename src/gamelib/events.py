import logging
import threading
import time
import warnings

from collections import defaultdict
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Dict, Type, List, Callable

_HANDLER_INJECTION_NAME = "_gamelib_handler_"


@dataclass
class Event:
    pass


@dataclass
class Update(Event):
    pass


EventHandler = Callable[[Event], None]


class MessageBus:
    def __init__(self, initial_handlers=None):
        """
        Parameters
        ----------
        initial_handlers : Dict[Type[Event], List[EventHandler]] | None
            Handlers can be registered after MessageBus construction or passed in this initial dict.
        """
        self.handlers = defaultdict(list)
        if initial_handlers:
            for event_type, callbacks in initial_handlers.items():
                self.handlers[event_type].extend(callbacks)
        self._adapters = dict()

    def register(self, event_type, callback) -> None:
        """
        Register a function as a callback for given event type.

        Parameters
        ----------
        event_type : Type[Event]
        callback : EventHandler
        """
        self.handlers[event_type].append(callback)

    def unregister(self, event_type, callback) -> None:
        """
        Stop handling this event type with this callback.

        Parameters
        ----------
        event_type : Type[Event]
        callback : EventHandler
        """
        try:
            self.handlers[event_type].remove(callback)
        except ValueError:
            warnings.warn(
                f"Tried to unregister {callback!r} from {self!r} when it was not registered"
            )

    def post_event(self, event) -> None:
        """
        Calls all callbacks registered to the type of this event. This includes pushing
        the event out through currently serviced connections.

        Parameters
        ----------
        event : Event
        """
        for handler_ in self.handlers[type(event)]:
            handler_(event)

    def service_connection(self, conn, event_types) -> None:
        """
        MessageBus will send all events it publishes through a serviced connection.

        Parameters
        ----------
        conn : Connection
        event_types : List[Type[Event]]
            List of event types that should be piped through.
        """
        adapter = _ConnectionAdapter(self, conn, event_types)
        for type_ in event_types:
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


def handler(event_type):
    """
    Marks the decorated function object which can later be retrieved
    with the find_handlers function in this module.

    Parameters
    ----------
    event_type : Type[Event]
        The type of event this callback should get registered to.
    """

    def inner(fn):
        setattr(fn, _HANDLER_INJECTION_NAME, event_type)
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
    handlers : Dict[Type[Event], List[EventHandler]]
        A dictionary mapping event types to a list of event handler functions
    """
    handlers = defaultdict(list)
    for name in dir(obj):
        attr = getattr(obj, name, None)
        if (attr is None) or not isinstance(attr, Callable):
            continue
        if (event_type := getattr(attr, _HANDLER_INJECTION_NAME, None)) is None:
            # might be a bound method
            if (fn := getattr(attr, "__func__", None)) is not None:
                for k, v in find_handlers(fn).items():
                    handlers[k].extend(v)
            continue
        handlers[event_type].append(attr)
    return handlers


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
