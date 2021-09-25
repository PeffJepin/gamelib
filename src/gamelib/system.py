from __future__ import annotations

import multiprocessing as mp
import time
from multiprocessing.connection import Connection
from typing import Type

import numpy as np

from . import events


class _SystemMeta(type):
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


class System(mp.Process, metaclass=_SystemMeta):
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


class UpdateComplete(System.Event):
    __slots__ = ["system_type"]

    system_type: Type[System]


class ArrayAttribute:
    def __init__(self, dtype, length=1):
        """
        A Descriptor that manages a numpy array which can be indexed on
        an instance with an 'entity_id' attribute.

        An ArrayAttribute instance can be reallocated with a new size using the
        reallocate function.

        Parameters
        ----------
        dtype : np.dtype
        length : int
            The underlying array will be 1 dimensional with this length
        """
        self._array = np.zeros((length,), dtype)

    def __get__(self, instance, owner):
        if instance is None:
            return self._array
        index = self._get_entity_id(instance)
        return self._array[index]

    def __set__(self, obj, value):
        index = self._get_entity_id(obj)
        self._array[index] = value

    def reallocate(self, new_length=None):
        """
        Reallocates the underlying memory with an optional new length.

        Parameters
        ----------
        new_length : int | None
            Optional new size to make the array.
        """
        dtype = self._array.dtype
        length = new_length or self._array.shape[0]
        self._array = np.zeros((length,), dtype)

    def _get_entity_id(self, instance):
        entity_id = getattr(instance, "entity_id", None)
        if entity_id is None:
            raise AttributeError(
                f"{self.__class__.__name__} Should be defined on a class with an 'entity_id' attribute."
            )
        if not (0 <= entity_id < self._array.shape[0]):
            raise IndexError(
                f"{entity_id=} is out of range for array with length {self._array.shape[0]}"
            )
        return entity_id
