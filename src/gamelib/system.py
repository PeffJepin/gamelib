from __future__ import annotations

from multiprocessing.connection import Connection
from typing import Type

import numpy as np

from . import Update, SystemStop
from .events import MessageBus, eventhandler, BaseEvent
from .sharedmem import DoubleBufferedArray


class _SystemMeta(type):
    """
    Used as the metaclass for System.

    This creates Types and binds them to name 'Event' and 'Component' in the class
    namespace. These types are created at class creation time, so are unique
    for each subclass of System.
    """

    Event: Type[BaseEvent]
    Component: Type[Component]

    @classmethod
    def __prepare__(metacls, name, bases, **kwargs):
        namespace: dict = super().__prepare__(name, bases, **kwargs)
        namespace["Event"] = type("SystemBaseEvent", (BaseEvent,), {})
        namespace["Component"] = type("SystemBaseComponent", (), {})
        return namespace

    @property
    def public_attributes(cls):
        attrs = []
        for comp_type in cls.Component.__subclasses__():
            discovered = [
                v for k, v in vars(comp_type).items() if isinstance(v, PublicAttribute)
            ]
            attrs.extend(discovered)
        return attrs


class System(metaclass=_SystemMeta):
    """
    A System will be run in a multiprocessing.Process and communicates
    with the parent process through a multiprocessing.Pipe.

    All subclasses of System will have their own unique Event and Component
    Type attributes.

    All Components of a System and all Events a System might raise
    should inherit from these base types creating an explicit
    link between a System and its Components/Events.

    For example given this system:
        class Physics(System):
            ...
        All components used by this system should derive from Physics.Component
        All events this system might raise should derive from Physics.Event

    """

    MAX_ENTITIES: int = 1024
    Event: Type
    Component: Type

    def __init__(self, conn, max_entities=1024):
        """
        Parameters
        ----------
        conn : Connection
            The Connection object from a multiprocessing.Pipe() used for communication.
        max_entities : int
            Passed in so the child Process can set the correct value if changed from default.
        """
        System.MAX_ENTITIES = max_entities
        outgoing_handlers = {
            event_type: [lambda e: conn.send(e)]
            for event_type in self.Event.__subclasses__()
            + System.Event.__subclasses__()
        }
        self._message_bus = MessageBus(outgoing_handlers)
        self._message_bus.register_marked_handlers(self)
        self._conn = conn
        self._running = True
        self._event_queue = []
        self._main()

    @classmethod
    def setup(cls):
        """Explicit setup must be called before initializing and using a System"""
        for attr in cls.public_attributes:
            attr.allocate_shm()

    @classmethod
    def teardown(cls):
        """Explicit teardown must be called after joining a System Process."""
        for attr in cls.public_attributes:
            attr.close_shm()

    def update(self):
        """Stub for subclass defined behavior."""

    def _main(self):
        while self._running:
            self._poll()

    def _poll(self):
        while self._conn.poll(0):
            event = self._conn.recv()
            if isinstance(event, Update):
                self._message_bus.post_event(event)
                for e in self._event_queue:
                    self._message_bus.post_event(e)
                self._message_bus.post_event(UpdateComplete(type(self)))
                self._event_queue = []
            elif isinstance(event, SystemStop):
                self._message_bus.post_event(event)
                break
            else:
                self._event_queue.append(event)

    @eventhandler(Update)
    def _update(self, event):
        self.update()

    @eventhandler(SystemStop)
    def _stop(self, event):
        self._running = False


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

    def __set__(self, obj, value):
        index = self._get_entity_id(obj)
        self._array[index] = value

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
        return f"{self._owner.__name__}__{self._name}"
