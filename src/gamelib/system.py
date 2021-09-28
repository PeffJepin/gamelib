from __future__ import annotations

import multiprocessing as mp
import time
from multiprocessing.connection import Connection
from typing import Type

import numpy as np

from . import events
from .sharedmem import DoubleBufferedArray


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

    MAX_ENTITIES: int = 1028
    _running: bool
    _message_bus: events.MessageBus

    def __init__(self, conn, max_entities=1024):
        """
        Initialize the system. Note _message_bus and _running are not
        initialized until later - once the System's Process has been started.

        Parameters
        ----------
        conn : Connection
            The Connection object from a multiprocessing.Pipe() used for communication.
        """
        self.HANDLERS = events.find_handlers(self)
        self.COMPONENTS = self.Component.__subclasses__()

        for attr in self._find_public_attributes():
            attr.allocate_shm()

        self._max_entities = max_entities
        self._conn = conn
        super().__init__()

    def join(self, timeout=0):
        super().join(timeout)
        for attr in self._find_public_attributes():
            attr.close_shm()

    def run(self):
        """
        The entry point to the System's Process.

        The message bus thread is currently processing events,
        this main thread can just sleep until ready to exit.
        """
        # Might be worth trying an asynchronous event loop in the future.
        System.MAX_ENTITIES = self._max_entities
        self._message_bus = events.MessageBus(self.HANDLERS)
        self._message_bus.service_connection(
            self._conn, self.Event.__subclasses__() + System.Event.__subclasses__()
        )

        self._running = True
        while self._running:
            time.sleep(1 / 1_000)

        for attr in self._find_public_attributes():
            attr.close_shm()

    def update(self):
        """Stub for subclass defined behavior."""

    @events.handler(events.Update)
    def _update(self, event):
        self.update()
        self._message_bus.post_event(UpdateComplete(type(self)))

    @events.handler(events.SystemStop)
    def _stop(self, event):
        self._running = False

    def _find_public_attributes(self):
        attrs = []
        for comp_type in self.COMPONENTS:
            discovered = [
                v for k, v in vars(comp_type).items() if isinstance(v, PublicAttribute)
            ]
            attrs.extend(discovered)
        return attrs


class UpdateComplete(System.Event):
    __slots__ = ["system_type"]

    system_type: Type[System]


class ArrayAttribute:
    _owner: object
    _name: str

    def __init__(self, dtype, length=System.MAX_ENTITIES):
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

    def __set_name__(self, owner, name):
        self._owner = owner
        self._name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self._array
        index = self._get_entity_id(instance)
        return self._array[index]

    def __set__(self, obj, value):
        index = self._get_entity_id(obj)
        self._array[index] = value

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


class PublicAttribute(ArrayAttribute):
    def __init__(self, dtype):
        self.dtype = dtype
        self._array = None

    def __get__(self, instance, owner):
        if self._array is None:
            self._array = DoubleBufferedArray(
                self._shm_id, (System.MAX_ENTITIES,), self.dtype
            )
        return super().__get__(instance, owner)

    def allocate_shm(self):
        self._array = DoubleBufferedArray.create(
            self._shm_id, (System.MAX_ENTITIES,), self.dtype
        )

    def close_shm(self):
        if self._array is not None:
            self._array.close()
            self._array = None

    def update(self):
        self._array.swap()

    @property
    def _shm_id(self):
        return f"{self._owner.__class__.__name__}__{self._name}"
