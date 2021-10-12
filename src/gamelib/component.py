from __future__ import annotations

from typing import Type, Dict

import numpy as np
from numpy import ma


from . import events, sharedmem, Config


class ArrayAttribute:
    _owner: Type[BaseComponent]
    _name: str
    _array: ma.MaskedArray

    def __init__(self, dtype):
        """
        A Descriptor that manages a numpy array. This should be used on a System.Component.

        Parameters
        ----------
        dtype : Any
            Any numpy compatible dtype
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
        value : ma.MaskedArray | Any
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
        self._array = ma.zeros((Config.MAX_ENTITIES,), self._dtype)
        self._array[:] = ma.masked


class PublicAttribute:
    def __init__(self, dtype):
        """
        A Descriptor for a shared public attribute.

        When accessed from a process which is running the parent System
        locally, the "write array" will be accessed. Otherwise a read-only
        view of the "read array" will be given.

        THERE ARE NO LOCKING MECHANISMS IN PLACE

        Once all systems have signaled that they are done updating
        the main process will copy the write buffer into the read buffer.

        Parameters
        ----------
        dtype : Any
            Any numpy compatibly dtype
        """
        self._dtype = dtype

        # Used when attempting to access..
        # Might be read or write array depending on where access occurs.
        self._array = None

        # Used when updating buffers
        self._read_view = None
        self._write_view = None

    def __set_name__(self, owner, name):
        if not issubclass(owner, BaseComponent) and owner != BaseComponent:
            raise TypeError(
                f"{self.__class__.__name__} is meant to be used on BaseComponent subclasses."
            )
        self._owner = owner
        self._name = name

    def __get__(self, instance, owner):
        """
        Get a view into the shared memory. If the parent System is running on
        this process, then the "write" buffer is used, otherwise the "read" buffer
        is used.

        Returns
        -------
        value : np.ndarray | Any
            A value from the array at index = entity_id if accessed from an instance.
            The whole array if accessed from the component type.

        Raises
        ------
        FileNotFoundError:
            If memory has not been allocated by the time of access.
        """
        if not self.is_open:
            self._open()
        if instance is None:
            return self._array
        index = instance.entity_id
        return self._array[index]

    def __set__(self, obj, value):
        """
        Set an entry in this array to value indexed on obj.entity_id

        Raises
        ------
        FileNotFoundError:
            If shm has not been allocated by time of use.
        """
        if not self.is_open:
            self._open()
        index = obj.entity_id
        self._array[index] = value

    def __repr__(self):
        return f"<PublicAttribute({self._owner.__name__}.{self._name}, open={self.is_open})>"

    def copy_buffer(self):
        """
        Copies the write buffer into the read buffer.

        Note: No synchronization primitives in place.
        """
        if self._read_view is None:
            self._read_view = sharedmem.connect(self._read_spec)
        if self._write_view is None:
            self._write_view = sharedmem.connect(self._write_spec)
        self._read_view[:] = self._write_view[:]

    def close_view(self):
        """
        Close the view to be sure access is not attempted once the shm file is closed.
        """
        self._array = None
        self._read_view = None
        self._write_view = None

    @property
    def is_open(self):
        return self._array is not None

    @property
    def shared_specs(self):
        """
        Get the underlying shared memory specification.

        Returns
        -------
        specs : tuple[ArraySpec]
        """
        return self._read_spec, self._write_spec

    @property
    def _shm_name(self):
        return f"{self._owner.__class__.__name__}__{self._name}"

    @property
    def _read_spec(self):
        return sharedmem.ArraySpec(
            self._shm_name + "_r", self._dtype, Config.MAX_ENTITIES
        )

    @property
    def _write_spec(self):
        return sharedmem.ArraySpec(
            self._shm_name + "_w", self._dtype, Config.MAX_ENTITIES
        )

    def _open(self):
        local = self._owner in Config.local_components
        spec = self._write_spec if local else self._read_spec
        self._array = sharedmem.connect(spec, readonly=(not local))


class ComponentCreated(events.Event):
    __slots__ = ["entity_id", "type", "args"]

    entity_id: int
    type: Type[BaseComponent]
    args: tuple


class ComponentMeta(type):
    instances: Dict[int, BaseComponent]

    def __new__(mcs, name, bases, namespace):
        namespace["instances"] = dict()
        return super().__new__(mcs, name, bases, namespace)

    def __getitem__(self, item):
        return self.instances.get(item, None)


class BaseComponent(metaclass=ComponentMeta):
    _array_attributes: dict
    _public_attributes: dict

    def __init__(self, entity_id):
        type(self).instances[entity_id] = self
        self.entity_id = entity_id

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        cls._array_attributes = {
            k: v for k, v in vars(cls).items() if isinstance(v, ArrayAttribute)
        }
        cls._public_attributes = {
            k: v for k, v in vars(cls).items() if isinstance(v, PublicAttribute)
        }

    def destroy(self):
        del type(self).instances[self.entity_id]
        for name in self._array_attributes.keys():
            setattr(self, name, ma.masked)

    @classmethod
    def destroy_all(cls):
        for component in cls.instances.copy().values():
            component.destroy()

    @classmethod
    def get_array_attributes(cls):
        return cls._array_attributes.values()

    @classmethod
    def get_public_attributes(cls):
        return cls._public_attributes.values()
