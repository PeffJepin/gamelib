import threading
import time
import warnings

from collections import defaultdict
from multiprocessing.connection import Connection
from types import MethodType
from typing import Dict, Type, List, Callable

_HANDLER_INJECTION_NAME = "_gamelib_handlers_"


class Event:
    pass


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

    def publish_event(self, event) -> None:
        """
        Calls all callbacks registered to the type of this event. This includes pushing
        the event out through currently serviced connections.

        Parameters
        ----------
        event : Event
        """
        for handler in self.handlers[type(event)]:
            handler(event)

    def service_connection(self, conn, event_types) -> None:
        """
        MessageBus will send all events it publishes through a serviced connection.

        Parameters
        ----------
        conn : PipeConnection
        event_types : List[Type[Event]]
            List of event types that should be piped through.
        """
        adapter = _ConnectionAdapter(self, conn, event_types)
        for type_ in event_types:
            self.register(type_, adapter)
        self._adapters[conn] = adapter

    def stop_connection_service(self, conn) -> None:
        """
        Stops the MessageBus from sending events through this connection.

        Parameters
        ----------
        conn : PipeConnection
        """
        adapter = self._adapters.pop(conn)
        for type_ in adapter.event_types:
            self.unregister(type_, adapter)
        adapter.is_active = False


def handlermethod(event_type: Type[Event]):
    """
    FOR USE ON METHODS

    Wraps a method declaration until its class object is created.
    Once created, the marker injects the class object to track handlers in __set_name__.

    https://docs.python.org/3/reference/datamodel.html#creating-the-class-object
    """
    class Marker:
        def __init__(self, fn):
            self.fn = fn

        def __set_name__(self, owner, name):
            fn, self.fn = self.fn, None
            setattr(owner, name, fn)
            if not (handlers := getattr(owner, _HANDLER_INJECTION_NAME, None)):
                handlers = defaultdict(list)
                setattr(owner, _HANDLER_INJECTION_NAME, handlers)
            handlers[event_type].append(getattr(owner, name))
    return Marker


def find_handlers(obj):
    """
    Helper function so other modules don't need to worry about how handler implements method marking.
    Since handlers are defined at class creation time, the functions need to be bound to the given instance.

    Parameters
    ----------
    obj : object

    Returns
    -------
    handlers : Dict[Type[Event], List[EventHandler]]
        Returns the injected handler dictionary if it is there, otherwise an empty version.
    """
    handlers = getattr(obj, _HANDLER_INJECTION_NAME, None)
    bound_handlers = defaultdict(list)
    if handlers is None:
        return bound_handlers

    for k, list_ in handlers.items():
        for handler in list_:
            bound_handlers[k].append(MethodType(handler, obj))
    return bound_handlers


class _ConnectionAdapter:
    def __init__(
        self,
        mb: MessageBus,
        conn: Connection,
        event_types: List[Type[Event]],
        recv_freq: int = 1,
    ):
        self.mb = mb
        self.freq = recv_freq
        self.conn = conn
        self.event_types = event_types
        self.is_active = True
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        # TODO:
        #   This will need some smarter Exception handling at some point. What happens if the connection is lost?
        #   Should the program just error out? Or is that okay and this thread can just exit peacefully?
        #   Once logging is set up it should at least be logged.
        try:
            while self.is_active:
                while self.conn.poll(0):
                    self.mb.publish_event(self.conn.recv())
                time.sleep(self.freq / 1_000)
        except (BrokenPipeError, EOFError):
            self.is_active = False

    def __call__(self, event):
        self.conn.send(event)
