from __future__ import annotations

import ctypes
import multiprocessing as mp
from typing import Dict, Optional

import numpy as np
from src.gamelib.ecs import _EcsGlobals, get_static_global, StaticGlobals


class ComponentType(type):
    """The metaclass for Component.

    Adds some helpful properties the Component Type objects.

    Notably allows a Component Type to be used as a context manager
    for synchronizing access to the underlying shared array.
    """
    array: Optional[np.ndarray]
    _mask: Optional[np.ndarray]
    _shm_array: Optional[mp.Array]
    _shm_mask: Optional[mp.Array]

    def __enter__(self):
        self._shm_array.get_lock().acquire()
        self._shm_mask.get_lock().acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._shm_array.get_lock().release()
        self._shm_mask.get_lock().release()

    @property
    def existing(cls):
        return cls.array[~cls._mask]

    @property
    def entities(cls):
        return np.where(~cls._mask)[0]


class NumpyDescriptor:
    """ Descriptor for the data attributes on a Component.

    When attribute access is invoked on an instance of a Component this
    will return/modify the entry in the components shared data array
    at the index corresponding to the instance.entity attribute.

    When used to access an attribute on a Component Type object this
    will return/modify all the unmasked (where a component actually exists)
    entries in the underlying shared array.
    """
    def __set_name__(self, owner, name):
        self._owner = owner
        self._name = name

    def __get__(self, obj, obj_type=None):
        if isinstance(obj, Component):
            if obj.entity is None:
                return None
            return self._owner.array[obj.entity][self._name]
        return obj_type.existing[self._name]

    def __set__(self, instance, value):
        self._owner.array[instance.entity][self._name] = value


class Component(metaclass=ComponentType):
    """ The base type for defining data in the ECS framework.

    Components should be used to define the data to be represented,
    with behaviour defined within Systems.

    Data for Components are allocated in shared memory and a view
    into them is given as a numpy array.

    An instance object of a component or the Type object can both be
    used in the same way to lock the underlying shared array
    and synchronize access to it across multiple processes.

    Components are a lot like dataclasses, such that you can simply
    annotate attributes with any numpy compatible data type.

    Before using a Component in any way the underlying shared memory
    must be allocated with the allocate function. All the annotated
    attributes will be combined into a single numpy.dtype and with
    that dtype a shared array will be allocated with length of
    EcsGlobals.max_entities.

    A call to free should be executed after the Component is not longer
    in use to free up the underling shared array.

    Attempted access to the Component before allocation or after freeing
    will result in an Exception.
    """
    entity: int
    _fields: Dict[str, type]
    _dtype: np.dtype
    _args = None
    _kwargs = None

    def __init__(self, *args, entity=None, **kwargs):
        """ A Component should be initialized with either args or kwargs
        and not both.

        If entity is left as None than the resulting instance will not be
        ready for use and will not be written into the underlying memory.
        To finalize the creation of the Component in this case a call to
        `bind_to_entity` must be included.

        Parameters
        ----------
        args : Any
            Should not be used with kwargs for initializing Component data.
            Args will be assigned to annotated attributes in the order
                that they were annotated.
        entity : int | None
            If entity is not None the component will be bound to this entity
            upon creation.
        kwargs : Any
            Should not be used with args for initializing Component data.
            Keys should correspond to annotated attribute names.

        Raises
        ------
        TypeError:
            If the Component hasn't been properly allocated yet.
        IndexError:
            If entity > EcsGlobals.max_entities.
        """
        self.entity = entity
        self._args = args
        self._kwargs = kwargs
        if entity is not None and (args or kwargs):
            self.bind_to_entity(entity)

    def __init_subclass__(cls, **kwargs):
        cls._fields = cls.__dict__.get("__annotations__", {})
        if not cls._fields:
            raise AttributeError("No attributes have been annotated.")
        cls._dtype = np.dtype([(name, type_) for name, type_ in cls._fields.items()])

        for name in cls._fields.keys():
            descriptor = NumpyDescriptor()
            descriptor.__set_name__(cls, name)
            setattr(cls, name, descriptor)

        cls.allocate()

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.values == other.values

    def __enter__(self):
        self._shm_array.get_lock().acquire()
        self._shm_mask.get_lock().acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._shm_array.get_lock().release()
        self._shm_mask.get_lock().release()

    def bind_to_entity(self, entity):
        """ Actually write this components data into the shared array.

        Parameters
        ----------
        entity : int
            The entity id this component is associated with.

        Raises
        ------
        IndexError:
            If entity > EcsGlobals.max_entities
        """
        self.entity = entity
        if self._args:
            for name, arg in zip(self._fields.keys(), self._args):
                setattr(self, name, arg)
        else:
            for name, value in self._kwargs.items():
                setattr(self, name, value)
        self._mask[entity] = False

    @classmethod
    def get_for_entity(cls, entity):
        """ Return an instance of this component for the given entity.

        Parameters
        ----------
        entity : int

        Returns
        -------
        instance : cls
            If the requested instance doesn't exist returns None.

        Raises
        ------
        TypeError:
            If the memory hasn't yet been properly allocated.
        IndexError:
            If entity > EcsGlobals.max_entities.
        """
        if cls._mask[entity]:
            return None
        return cls(entity=entity)

    @classmethod
    def destroy(cls, entity):
        """ Masks this entities entry into the underlying shared array.

        This is safe to call even if this entity doesn't exist.

        Parameters
        ----------
        entity : int

        Raises
        ------
        TypeError:
            If this Component isn't currently allocated.
        IndexError:
            If entity > EcsGlobals.max_entities.
        """
        cls._mask[entity] = True

    @classmethod
    def destroy_all(cls):
        """ Masks the entire underlying array.

        Raises
        ------
        TypeError:
            If this Component isn't currently allocated.
        """
        cls._mask[:] = True

    @classmethod
    def enumerate(cls):
        """ Mimics the behaviour of built in enumerate, but only returns
        values for existing components.

        Returns
        -------
        entry : Generator
            yields a tuple of (entity, component) for each existing component.

        Raises
        ------
        TypeError:
            If this Component isn't currently allocated.
        """
        indices = np.where(~cls._mask)
        for i in np.nditer(indices):
            yield i, cls.get_for_entity(i)

    @classmethod
    def allocate(cls):
        """ Allocates the underling shared memory array and entity mask.

        The length of the arrays are defined by EcsGlobals.max_entities.
        `free` should be called when the Component is no longer in use.
        An additional call to allocate in the middle of Systems execution
        will most likely result in a SegFault.
        """
        length = get_static_global(StaticGlobals.MAX_ENTITIES)
        cls._shm_array = mp.Array(
            ctypes.c_byte, cls._dtype.itemsize * length
        )
        cls._shm_mask = mp.Array(ctypes.c_bool, [True] * length)
        cls.array = np.ndarray(
            (length,), cls._dtype, cls._shm_array.get_obj()
        )
        cls._mask = np.ndarray(
            (length,), bool, cls._shm_mask.get_obj()
        )

    @classmethod
    def free(cls):
        """ Remove reference to the underlying shared memory.

        This should be called once the Component is no longer being used.
        """
        cls._shm_array = None
        cls._shm_mask = None
        cls._mask = None
        cls.array = None

    @property
    def values(self):
        """ Get a tuple of the attributes annotated on this Component.

        Returns
        -------
        values : tuple | None
            The order of the values is the same order that they were annotated.
            Returns None if called before this Components been bound to an entity.
        """
        if not self.entity:
            return None
        return tuple(self.array[self.entity][name] for name in self._fields.keys())
