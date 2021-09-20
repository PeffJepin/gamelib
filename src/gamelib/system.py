from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Type

from . import events


class SystemMeta(type):
    """
    Used as the metaclass for System.

    This creates Types and binds them to name 'Event' and 'Component' in the class
    namespace. These types are created at class creation time, so are unique
    for each subclass of System.
    """

    @classmethod
    def __prepare__(metacls, name, bases, **kwargs):
        namespace: dict = super().__prepare__(name, bases, **kwargs)
        namespace["Event"] = type("SystemBaseEvent", (events.Event,), {})
        namespace["Component"] = type("SystemBaseComponent", (), {})
        return namespace


class System(mp.Process, metaclass=SystemMeta):
    """
    A System is a process that processes events passed over a Pipe Connection.

    All subclasses of System will have their own unique Event and Component
    Type attributes. All Components of a System and all Events to be raised
    by a System should derive from these base types creating an explicit
    link between a System and its Components/Events.

    For example given this system:
        class Physics(System):
            ...
        All components used by this system should derive from Physics.Component
        All events this system might raise should derive from Physics.Event

    """

    Event: Type
    Component: Type

    _running: bool
    _message_bus: events.MessageBus

    def __init__(self, conn):
        """
        Initialize the system. Note _message_bus and _running are not
        initialized until later - once the System's Process has been started.

        Parameters
        ----------
        conn : Connection
            The Connection object from a multiprocessing.Pipe() used for communication.
        """
        self.HANDLERS = events.find_handlers(self)
        self._conn = conn
        super().__init__()

    def run(self):
        """
        The entry point to the System's Process.

        The message bus thread is currently processing events,
        this main thread can just sleep until ready to exit.
        """
        # Might be worth trying an asynchronous event loop in the future.
        self._running = True
        self._message_bus = events.MessageBus(self.HANDLERS)
        self._message_bus.service_connection(
            self._conn, self.Event.__subclasses__() + System.Event.__subclasses__()
        )
        while self._running:
            time.sleep(1 / 1_000)

    def update(self):
        """Stub for subclass defined behavior."""

    @events.handler(events.Update)
    def _update(self, event):
        self.update()
        self._message_bus.post_event(UpdateComplete(type(self)))

    @events.handler(events.SystemStop)
    def _stop(self, event):
        self._running = False


@dataclass(frozen=True)
class UpdateComplete(System.Event):
    system_type: Type[System]
