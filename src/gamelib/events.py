import threading
import time

from collections import defaultdict
from multiprocessing.connection import PipeConnection
from types import MethodType
from typing import Dict, Type, List, Callable

_HANDLER_INJECTION_NAME = '__gamelib_handlers__'


class Event:
    pass


class Update(Event):
    pass


EventHandler = Callable[[Event], None]


class MessageBus:
    def __init__(self, initial_handlers: Dict[Type[Event], List[Callable]] = None):
        self.handlers = defaultdict(list)
        if initial_handlers:
            for event_type, callbacks in initial_handlers.items():
                self.handlers[event_type].extend(callbacks)
        self._adapters = dict()

    def register(self, event_type: Type[Event], callback: Callable):
        self.handlers[event_type].append(callback)

    def unregister(self, event_type: Type[Event], callback: Callable):
        self.handlers[event_type].remove(callback)

    def handle(self, event: Event):
        for handler in self.handlers[type(event)]:
            handler(event)

    def service_connection(self, conn: PipeConnection, event_types: List[Type[Event]]):
        adapter = _ConnectionAdapter(self, conn, event_types)
        for type_ in event_types:
            self.register(type_, adapter)
        self._adapters[conn] = adapter

    def stop_connection_service(self, conn: PipeConnection):
        adapter = self._adapters.pop(conn)
        for type_ in adapter.event_types:
            self.unregister(type_, adapter)
        adapter.is_active = False


class _ConnectionAdapter:
    def __init__(self, mb: MessageBus, conn: PipeConnection, event_types: List[Type[Event]], recv_freq: int = 1):
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
                    self.mb.handle(self.conn.recv())
                time.sleep(self.freq / 1_000)
        except (BrokenPipeError, EOFError):
            self.is_active = False

    def __call__(self, event):
        self.conn.send(event)


def handler(event_type: Type[Event]):
    """
    FOR USE ON CLASS METHODS

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
    handlers = getattr(obj, _HANDLER_INJECTION_NAME, None) or defaultdict(list)
    for k, list_ in handlers.items():
        for i, handler_ in enumerate(list_):
            bound_method = MethodType(handler_, obj)
            list_[i] = bound_method
    return handlers
