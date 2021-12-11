"""The events module is meant to allow separate components of an application to
communicate without explicit knowledge of each other through a callback system.

An event can be basically any object, and should be serializable. It is just
a container for some data, making namedtuple/dataclass good choices.

Examples
--------
Events are just data containers, the following are essentially equivilant:

>>> @dataclass
>>> class Update:
...     dt: float

>>> class Update(NamedTuple):
...     dt: float

>>> class Update:
...     def __init__(self, dt):
...         self.dt = dt


Subscribe and unsubscribe functions as event handlers:

>>> def do_update(event):
...     print(f"dt={event.dt}")
...
>>> subscribe(Update, do_update)
>>> post(Update(0.01))
dt=0.01
>>> unsubscribe(Update, do_update)
>>> post(Update(0.01))
>>> # no callback


Using an object as a container for handlers:

>>> class System:
...     @handler(Update)
...     def do_update(self, event):
...         print(f"Doing update, dt={event.dt}")
...
>>> system = System()
>>> post(Update(0.01))
>>> # nothing happens
>>> subscribe_obj(system)
>>> post(Update(0.01))
Doing update, dt=0.01
"""

from __future__ import annotations

import inspect
import threading
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from multiprocessing.connection import Connection
from typing import Callable, Sequence, NamedTuple

_HANDLER_INJECTION_ATTRIBUTE = "_gamelib_handler_"
_event_handlers = defaultdict(list)
_adapters = dict()


class Update(NamedTuple):
    dt: float


class Signal(Enum):
    """
    Todo: Signal class for events with no args. Should be interchangable
        with event and offer fast ipc support.
    """

    pass


class SystemStop:
    pass


class Quit:
    pass


def post(event):
    """Calls callbacks registered to the type of this event.

    Parameters
    ----------
    event : Any
        An event is just a data container.
    """

    key = type(event)
    for handler_ in _event_handlers[key]:
        handler_(event)


def subscribe(event_type, *callbacks):
    """Subscribe callbacks to a given event type.

    Parameters
    ----------
    event_type : type
    *callbacks : Callable
    """

    _event_handlers[event_type].extend(callbacks)


def unsubscribe(event_type, *callbacks) -> None:
    """Unsubscribe callbacks from a given event type.

    Parameters
    ----------
    event_type : type
    *callbacks : Callable
    """

    for callback in callbacks:
        try:
            _event_handlers[event_type].remove(callback)
        except ValueError:
            pass


def subscribe_obj(obj):
    """Finds methods bound to an object that have been marked as event
    handlers and subscribe them to appropriate events.

    Parameters
    ----------
    obj : object
    """

    for event_key, handlers in find_marked_handlers(obj).items():
        subscribe(event_key, *handlers)


def unsubscribe_obj(obj):
    """Removes methods bound to an object that have been marked as event
    handlers.

    Parameters
    ----------
    obj : object
    """

    for event_key, handlers in find_marked_handlers(obj).items():
        unsubscribe(event_key, *handlers)


def clear_handlers(*event_types):
    """If event types are given this will clear all handlers from just those
    types, otherwise it will clear all event handlers.

    Parameters
    ----------
    *event_types : type
    """

    if not event_types:
        _event_handlers.clear()
        for adapter in _adapters.values():
            adapter.stop()
        _adapters.clear()
    else:
        for type_ in event_types:
            _event_handlers[type_].clear()


def handler(event_type):
    """Decorator to mark methods of a class as event handlers. See tests or
    the module docstring above for examples.

    It is probably not advisable to create a large number of instances of
    a class using these markers, as the module is not optimized for adding and
    removing handlers frequently.

    This is better served to organize several handlers together on a system
    that itself might operate over many instances of objects.

    Parameters
    ----------
    event_type : type
    """

    def inner(fn):
        setattr(fn, _HANDLER_INJECTION_ATTRIBUTE, event_type)
        return fn

    return inner


def find_marked_handlers(obj):
    """Given an object that has methods marked as event handlers, searches
    through the object and finds all the marked methods.

    Parameters
    ----------
    obj : object

    Returns
    -------
    dict[type, list[Callable]]:
        A dictionary mapping event types to the actual handlers.
    """

    handlers = defaultdict(list)
    for name, attr in inspect.getmembers(obj):
        if not isinstance(attr, Callable):
            continue
        if (
            event_type := getattr(attr, _HANDLER_INJECTION_ATTRIBUTE, None)
        ) is None:
            continue
        handlers[event_type].append(attr)
    return handlers


def service_connection(conn, *event_types, poll=True):
    """Send the specified event_types over the given connection when they
    are posted. If `poll` is True (the default) then this will also poll the
    pipe and read events out of it, posting them after being received.

    Parameters
    ----------
    conn : Connection
    *event_types : type
    poll : bool, optional
    """

    adapter = _ConnectionAdapter(conn, event_types)
    _adapters[conn] = adapter
    for type_ in event_types:
        subscribe(type_, adapter)
    if poll:
        adapter.start()


def stop_connection_service(conn):
    """Stops serving events to and from the given connection.

    Parameters
    ----------
    conn : Connection
    """

    adapter = _adapters.pop(conn, None)
    if adapter is None:
        return
    for type_ in adapter.event_types:
        unsubscribe(type_, adapter)
    adapter.stop()


class _ConnectionAdapter:
    """Internal helper for serving and receiving from a multiprocessing.Pipe"""

    def __init__(
        self,
        conn: Connection,
        event_types: Sequence[type],
    ):
        self.conn = conn
        self.event_types = event_types
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self._running = False

    def _poll(self):
        """Mainloop for a polling thread."""

        self._running = True
        while self._running:
            try:
                if not self.conn.poll(0.01):
                    continue
                message = self.conn.recv()
                event = message
                post(event)
            except (BrokenPipeError, EOFError):
                self._running = False
                break
            except TypeError as e:
                if isinstance(message, Exception):
                    raise message
                else:
                    raise e

    def start(self):
        """Start the thread"""

        self.thread.start()

    def stop(self):
        """Stop the thread"""

        self._running = False

    def __call__(self, event):
        """Handles event by passing through the pipe."""

        self.conn.send(event)
