"""The events module is meant to allow separate components of an application to
communicate without explicit knowledge of each other through a callback system.

An event can be basically any object, and should be serializable. It is just
a container for some data, making namedtuple/dataclass good choices.

Examples
--------
Events are just data containers, the following are essentially equivilant:

>>> @dataclasses.dataclass
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
>>> publish(Update(0.01))
dt=0.01
>>> unsubscribe(Update, do_update)
>>> publish(Update(0.01))
>>> # no callback


Using an object as a container for handlers:

>>> class System:
...     @handler(Update)
...     def do_update(self, event):
...         print(f"Doing update, dt={event.dt}")
...
>>> system = System()
>>> publish(Update(0.01))
>>> # nothing happens
>>> subscribe_marked(system)
>>> publish(Update(0.01))
Doing update, dt=0.01
"""
# TODO: I should consider at some point whether or not to change the handler
#   decorator to behave more like the _InputEvent.handler. It would make event
#   handler marking more uniform, but may conflict with using dataclasses and
#   named tuples for events.

import threading
import collections
import multiprocessing

from typing import Sequence
from typing import NamedTuple

from gamelib import utils


_HANDLER_INJECTION_ATTRIBUTE = "_gamelib_handler_"
_event_handlers = collections.defaultdict(list)
_internal_handlers = list()
_adapters = dict()


class Update(NamedTuple):
    dt: float


class InternalUpdate(NamedTuple):
    dt: float


class Quit:
    pass


def publish(event):
    """Calls callbacks registered to the type of this event.

    Parameters
    ----------
    event : Any
        An event is just a data container.
    """

    if isinstance(event, InternalUpdate):
        for handler_ in _internal_handlers:
            handler_(event)
        return

    for handler_ in _event_handlers[type(event)]:
        handler_(event)


def subscribe(event_type, *callbacks):
    """Subscribe callbacks to a given event type.

    Parameters
    ----------
    event_type : type
    *callbacks : Callable
    """

    if event_type == InternalUpdate:
        _internal_handlers.extend(callbacks)
    else:
        _event_handlers[event_type].extend(callbacks)


def unsubscribe(event_type, *callbacks) -> None:
    """Unsubscribe callbacks from a given event type.

    Parameters
    ----------
    event_type : type
    *callbacks : Callable
    """

    if event_type == InternalUpdate:
        for callback in callbacks:
            try:
                _internal_handlers.remove(callback)
            except ValueError:
                pass
        return

    for callback in callbacks:
        try:
            _event_handlers[event_type].remove(callback)
        except ValueError:
            pass


def subscribe_marked(obj):
    """Finds methods bound to an object that have been marked as event
    handlers and subscribe them to appropriate events.

    Parameters
    ----------
    obj : object
    """

    for event_key, handlers in find_marked_handlers(obj).items():
        subscribe(event_key, *handlers)


def unsubscribe_marked(obj):
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

    return utils.MethodMarker(type="event", extra=event_type)


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

    handlers = collections.defaultdict(list)
    for mark, method in utils.MethodMarker.lookup(obj, type="event").items():
        handlers[mark.extra].append(method)
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
        conn: multiprocessing.Pipe,
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
                publish(event)
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
