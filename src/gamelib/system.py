from __future__ import annotations

import multiprocessing as mp
import traceback
from multiprocessing.connection import Connection
from typing import Type, List

import numpy as np
from numpy import ma

from . import Update, SystemStop, events, sharedmem
from .events import eventhandler, Event
from .sharedmem import DoubleBufferedArray, SharedBlock, ArraySpec


class BaseComponent:
    SYSTEM: Type[System]

    def __init__(self, entity_id):
        self.entity_id = entity_id


class _SystemMeta(type):
    """
    Used as the metaclass for System.

    This creates a BaseComponent 'Component' in the class namespace.
    cls.Component is unique for each subclass of System.
    """

    Component: Type[BaseComponent]

    def __new__(mcs, *args, **kwargs):
        cls: _SystemMeta = super().__new__(mcs, *args, **kwargs)
        cls.Component.SYSTEM = cls
        return cls

    @classmethod
    def __prepare__(mcs, name, bases, **kwargs):
        namespace: dict = super().__prepare__(name, bases, **kwargs)
        namespace["Component"] = type("SystemBaseComponent", (BaseComponent,), {})
        return namespace

    @property
    def public_attributes(cls):
        """
        Gets a list of all PublicAttribute instances owned by subclasses of cls.Component

        Returns
        -------
        atts : List[PublicAttribute]
        """
        return [
            attr
            for comp_type in cls.Component.__subclasses__()
            for attr in vars(comp_type).values()
            if isinstance(attr, PublicAttribute)
        ]

    @property
    def shared_specs(cls):
        """
        Get a list of all the shm ArraySpecs this system implements.

        Returns
        -------
        specs : Iterable[ArraySpec]
        """
        specs = []
        for attr in cls.public_attributes:
            specs.extend(attr.shared_specs)
        return specs

    @property
    def array_attributes(cls):
        """
        Gets a list of all ArrayAttribute instances defined on subclasses of cls.Component

        Returns
        -------
        attrs : List[ArrayAttribute]
        """
        attrs = []
        for comp_type in cls.Component.__subclasses__():
            discovered = [
                v for k, v in vars(comp_type).items() if isinstance(v, ArrayAttribute)
            ]
            attrs.extend(discovered)
        return attrs


class System(metaclass=_SystemMeta):
    """
    A System will be run in a multiprocessing.Process and communicates
    with the parent process through a multiprocessing.Pipe.

    All subclasses of System will have their own unique Component Type attributes.

    All Components of a System should inherit from these base types creating an explicit
    link between a System and its Components.

    For example given this system:
        class Physics(System):
            ...
        All components used by this system should derive from Physics.Component
    """

    MAX_ENTITIES: int = 1024
    Component: Type[BaseComponent]
    _running: bool

    def update(self):
        """
        Stub for subclass defined behavior.

        The Update event will first invoke this method, then post all events
        pooled in the event queue, and finally send a SystemUpdateComplete back
        to the main process.
        """

    def stop(self):
        self._running = False
        events.unregister_marked(self)

    @eventhandler(Update)
    def _update_handler(self, event):
        self.update()


class ProcessSystem(System):
    def __init__(self, conn, **kwargs):
        self._conn = conn
        self._running = False
        self._event_queue = []
        events.register_marked(self)

    @classmethod
    def run_in_process(cls, **kwargs):
        """
        Method that should be called from the main process to actually start the system.

        System.MAX_ENTITIES should be set before calling if not default.
        If using PublicAttributes sharedmem should be allocated first.

        Returns
        -------
        local : Connection
            multiprocessing.Pipe
        process: Process
        """
        local, foreign = mp.Pipe()
        process = mp.Process(
            target=cls._run,
            args=(foreign, System.MAX_ENTITIES),
            kwargs=kwargs,
        )
        process.start()
        return local, process

    @classmethod
    def _run(cls, conn, max_entities, **kwargs):
        """Internal process entry point. Sets global System state before starting."""
        System.MAX_ENTITIES = max_entities
        for attr in cls.array_attributes:
            attr.reallocate()
        inst = cls(conn, **kwargs)
        inst._main()
        cls._teardown_shared_state()

    def raise_event(self, event, key=None):
        """Sends an event back to the main process."""
        self._conn.send((event, key))

    def _main(self):
        self._running = True
        while self._running:
            self._poll()

    def _poll(self):
        if not self._conn.poll(0):
            return
        message = self._conn.recv()
        try:
            (event, key) = message
            if isinstance(event, Update) or isinstance(event, SystemStop):
                events.post_event(event, key=key)
            else:
                self._event_queue.append((event, key))
        except Exception as e:
            msg_with_traceback = f"{e}\n\n{traceback.format_exc()}"
            self._conn.send(type(e)(msg_with_traceback))

    @classmethod
    def _teardown_shared_state(cls):
        """Local only teardown."""
        for attr in cls.public_attributes:
            attr.close()
        sharedmem.close()

    @eventhandler(SystemStop)
    def _stop(self, _):
        self.stop()

    @eventhandler(Update)
    def _update_handler(self, _):
        for (event, key) in self._event_queue:
            events.post_event(event, key=key)
        self._event_queue = []
        self.update()
        self.raise_event(SystemUpdateComplete(type(self)))


class SystemUpdateComplete(Event):
    """
    Posted to the main process after a system finishes handling an Update event.
    """

    __slots__ = ["system_type"]

    system_type: Type[System]


class ArrayAttribute:
    _owner: Type[BaseComponent]
    _name: str
    _array: ma.MaskedArray

    def __init__(self, dtype):
        """
        A Descriptor that manages a numpy array. This should be used on a System.Component.

        Parameters
        ----------
        dtype : type[int] | type[float] | np.dtype
        """
        self._dtype = dtype

    def __set_name__(self, owner, name):
        if not issubclass(owner, BaseComponent):
            raise TypeError(
                f"{self.__class__.__name__} is meant to be used on BaseComponent subclasses."
            )
        self._owner = owner
        self._name = name
        self.reallocate()

    def __get__(self, instance, owner):
        """
        Returns
        -------
        value : ma.MaskedArray | int | float | str
            If invoked on an instance returns an entry from the array using entity_id as index.
            Otherwise if invoked from the Type object it returns the entire array.
        """
        if instance is None:
            return self._array
        index = instance.entity_id
        return self._array[index]

    def __set__(self, obj, value):
        index = obj.entity_id
        self._array[index] = value

    def reallocate(self):
        """Reallocates the underlying array. Does not preserve data."""
        self._array = ma.zeros((self.length,), self._dtype)
        self._array[:] = ma.masked

    @property
    def length(self):
        return System.MAX_ENTITIES


class PublicAttribute:

    _dbl_buffer: DoubleBufferedArray | None = None

    def __init__(self, dtype):
        """
        A Descriptor for a shared public attribute.

        Normally an array attribute is localized to its own process,
        data stored in one of these attributes can be accessed from any
        process.

        THERE ARE NO LOCKING MECHANISMS IN PLACE

        The array will be double buffered, with writes going to
        one array and reads to the other. Once all systems have
        signaled that they are done Updating the main process
        will copy the write buffer into the read buffer.

        Parameters
        ----------
        dtype : type[int] | type[float] | np.dtype
        """
        self.open = False
        self._dtype = dtype

    def __set_name__(self, owner, name):
        if not issubclass(owner, BaseComponent):
            raise TypeError(
                f"{self.__class__.__name__} is meant to be used on BaseComponent subclasses."
            )
        self._owner = owner
        self._name = name

    def __get__(self, instance, owner):
        """
        Get reference to the DoubleBufferedArray. Will attempt to connect the array to shm
        if it is not already connected.

        Returns
        -------
        value : Numeric | DoubleBufferedArray
            A value from the array at index = entity_id if accessed from an instance.
            The whole array if accessed from the component type.

        Raises
        ------
        FileNotFoundError:
            If memory has not been allocated by the time of access.
        """
        if not self.open:
            self._open()
        if instance is None:
            return self._dbl_buffer
        index = instance.entity_id
        return self._dbl_buffer[index]

    def __set__(self, obj, value):
        """
        Set an entry in this array to value indexed on obj.entity_id

        Raises
        ------
        FileNotFoundError:
            If shm has not been allocated by time of use.
        """
        if not self.open:
            self._open()
        index = obj.entity_id
        self._dbl_buffer[index] = value

    def __repr__(self):
        return f"<PublicAttribute({self._owner.__name__}.{self._name}, open={self._dbl_buffer.is_open})>"

    def update(self):
        """
        Copies the write buffer into the read buffer.

        Note that there are no synchronization primitives guarding this process.
        """
        self._dbl_buffer.flip()

    def close(self):
        self.open = False
        self._dbl_buffer = None

    @property
    def shared_specs(self):
        """
        Get the underlying shared memory specification.

        Returns
        -------
        specs : tuple[ArraySpec]
        """
        if not self.open:
            return DoubleBufferedArray(self._shm_name, self._dtype, System.MAX_ENTITIES).specs
        return self._dbl_buffer.specs

    @property
    def _shm_name(self):
        return f"{self._owner.__class__.__name__}__{self._name}"

    def _open(self):
        self._dbl_buffer = DoubleBufferedArray(self._shm_name, self._dtype, System.MAX_ENTITIES)
        self._dbl_buffer.connect()
        self.open = True
