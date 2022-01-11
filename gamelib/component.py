# TODO: Some way to get a proxy to a component attribute such that another
#   module could keep reference to the proxy and call it on demand to get
#   to the internal data. Since the internal array is subject to reallocation,
#   simply getting a reference to the underlying array is insufficient.

# TODO: An option to allocate the internal arrays using the structured dtype
#   instead of individual arrays. Using a decorator approach like dataclass
#   uses might offer a more suitable API for optional features like this.

# TODO: A module docstrings with examples after this module is fleshed out a
#   bit more.

# TODO: Eventually I will want the option to allocate the arrays using
#   shared memory. Since they are regularly reallocated this will probably
#   require a new shared memory module so other processes can easily find the
#   internal array after it has been reallocated.

import itertools
import threading

import numpy as np


_STARTING_LENGTH = 10


class _ComponentType(type):
    """Metaclass for Component.

    Responsible for managing access to the underlying ndarrays when
    accessed through the Type object.
    """

    def __getattr__(cls, name):
        """Get the underlying array if name is an annotated field."""

        if name in cls._fields:
            return getattr(cls, "_" + name)[: cls.length]
        raise AttributeError(f"{cls!r} has no attribute {name!r}")

    def __setattr__(cls, name, value):
        """Don't allow values to be bound to annotated attribute names on
        the Type object."""

        if cls._initialized and name in cls._fields:
            # don't allow setting annotated fields as class attributes.
            return
        super().__setattr__(name, value)

    def __enter__(cls):
        """Acquire a lock on the internal arrays."""

        cls._lock.acquire()

    def __exit__(cls, *args, **kwargs):
        """Release the internal lock."""

        cls._lock.release()


class Component(metaclass=_ComponentType):
    """Component is a base class used for laying out data in contiguous
    memory. Attributes should be annotated like a dataclass and appropriate
    internal arrays will be managed with the annotated dtype."""

    _initialized = False

    def __init_subclass__(cls, **kwargs):
        """Initialize the subclass based on what has been annotated."""

        # __init_subclass__ can also be used to reset a component, so it
        # needs to be reset to False here.
        cls._initialized = False
        cls._lock = threading.RLock()

        # find annotated fields
        cls._fields = cls.__dict__.get("__annotations__", {})
        if not cls._fields:
            raise AttributeError("No attributes have been annotated.")

        # keep an array of ids alongside the annotated fields
        cls.ids = np.empty(_STARTING_LENGTH, int)
        cls.ids[:] = -1

        # keep a record of where to find data for a particular component id
        cls.id_lookup = np.zeros(_STARTING_LENGTH, int)
        cls.id_lookup[:] = -1

        # maintain a standard way of assigning ids
        cls._max_length = _STARTING_LENGTH
        cls._counter = itertools.count(0)
        cls._recycled_ids = []
        cls.length = 0

        # create the underlying component arrays and form an aggregate
        # structured dtype allowing the individual arrays created from
        # attribute annotation to be viewed as a single structured array
        structure = []
        for name, dtype in cls._fields.items():
            if not isinstance(dtype, np.dtype):
                dtype = np.dtype(dtype)
            array = np.zeros(_STARTING_LENGTH, dtype)
            setattr(cls, "_" + name, array)
            structure.append((name, dtype))
        cls._structured_dtype = np.dtype(structure)
        cls.itemsize = cls._structured_dtype.itemsize

        # effectively freezes the annotated attributes, disallowing
        # the public-facing attributes to be set with new values.
        cls._initialized = True

    def __new__(cls, *args, id=None, **kwargs):
        """Create some new Component data. If id is given this will load
        access into existing data instead.

        Parameters
        ----------
        *args : Any
            __init__ args
        id : int, optional
            If given, an existing component should be loaded, otherwise
            a new id should be generated and a new component will be created.
        **kwargs : Any
            __init__ kwargs

        Returns
        -------
        Component | None:
            If id is given and isn't found to be an existing component, None
            will be returned instead of a Component instance.
        """

        if id is not None:
            # check if this id exists
            if cls.id_lookup[id] == -1:
                return None

        else:
            id = cls._get_new_id()
            if id >= cls._max_length:
                cls.reallocate(cls._max_length * 1.5)
            cls.id_lookup[id] = cls.length
            cls.ids[cls.length] = id
            cls.length += 1

        instance = super().__new__(cls)
        instance.id = id
        return instance

    def __init__(self, *args, **kwargs):
        """Set the initial values of a component. *args or **kwargs are
        mutually exclusive.

        Parameters
        ----------
        *args : Any
            Args will map to annotated attributes in the order they are given.
        **kwargs : Any
            Keys will map to annotated attribute names.
        """

        if args:
            for name, arg in zip(self._fields, args):
                setattr(self, name, arg)
        else:
            for name, value in kwargs.items():
                setattr(self, name, value)

    def __repr__(self):
        values = ", ".join(
            f"{name}={getattr(self, name)}" for name in self._fields
        )
        return f"<{self.__class__.__name__}({values})>"

    def __eq__(self, other):
        """A component compares for equality based on the values of it's
        annotated attributes."""

        if type(self) == type(other):
            return self.values == other.values
        else:
            return self.values == other

    def __setattr__(self, name, value):
        """Setting an annotated attribute should set the value inside the
        appropriate internal array, rather than binding the given value to
        the instance as would normally happen."""

        if name in self._fields:
            index = self.id_lookup[self.id]
            getattr(self, "_" + name)[index] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        """Getting an annotated attribute from the context of a Component
        instance should index into the internal array to retrieve the
        appropriate value."""

        if name in self._fields:
            index = self.id_lookup[self.id]
            value = getattr(self, "_" + name)[index] if index >= 0 else None
            return value
        else:
            raise AttributeError(f"{self!r} has no attribute {name!r}")

    def __enter__(self):
        """Use the instance as a context manager to lock the internal array."""

        self._lock.acquire()

    def __exit__(self, *args, **kwargs):
        """Release the lock."""

        self._lock.release()

    @property
    def values(self):
        """Get the values for this instance's annotated attributes."""

        return tuple(getattr(self, name) for name in self._fields)

    @classmethod
    def get(cls, id):
        """Gets an existing instance of this Component given an id.

        Parameters
        ----------
        id : int

        Returns
        -------
        Component | None:
            Depending on if a component with this id is accounted for.
        """

        return cls(id=id)

    @classmethod
    def get_raw_arrays(cls):
        """Gets the raw internal arrays (unmasked).

        Returns
        -------
        np.ndarray:
            The resulting array will have a structured dtype which is an
            aggregate of all the annotated attributes of this component.
        """

        combined = np.empty(cls._max_length, cls._structured_dtype)
        for name in cls._fields:
            combined[name] = getattr(cls, "_" + name)
        return combined

    @classmethod
    def destroy(cls, target):
        """Destroys a component given either an instance of `cls` or an
        integer id.

        Parameters
        ----------
        target : Component | int
            This can either be an instance of this type of component or the
            id of the component to be deleted.
        """

        id = target if isinstance(target, int) else target.id
        index = cls.id_lookup[id]

        # component doesn't actually exist
        if index == -1:
            return

        # swap the component to be deleted to the end of the array and
        # decrement the length to delete the component without necessarily
        # having to reallocate the entire array
        last_index = cls.length - 1
        if index != last_index:
            id_of_last_component = cls.ids[last_index]
            cls.id_lookup[id_of_last_component] = index
            arrays = cls.get_raw_arrays()
            for name in cls._fields:
                arrays[name][index] = arrays[name][last_index]

        cls.length -= 1
        cls._recycled_ids.append(id)
        cls.id_lookup[id] = -1
        cls._consider_shrinking()

    @classmethod
    def reset(cls):
        """Resets the component to initial state."""

        cls.__init_subclass__()

    @classmethod
    def reallocate(cls, new_length):
        """Reallocates the internal arrays to the new length. When
        automatically used internally, this will be sure not to delete entries.
        If invoked manually it's possible to destroy data that still in use.

        Parameters
        ----------
        new_length : int
        """

        new_length = max(int(new_length), _STARTING_LENGTH)
        for name in cls._fields:
            name = "_" + name
            array = _reallocate_array(getattr(cls, name), new_length)
            setattr(cls, name, array)
        cls.ids = _reallocate_array(cls.ids, new_length, fill=-1)
        cls._max_length = new_length

    @classmethod
    def _get_new_id(cls):
        """Requests an id to assign to a newly created component."""

        if cls._recycled_ids:
            id = cls._recycled_ids.pop(0)
        else:
            id = next(cls._counter)
        if id >= len(cls.id_lookup):
            cls.id_lookup = _reallocate_array(cls.id_lookup, id * 1.5, fill=-1)
        return id

    @classmethod
    def _consider_shrinking(cls):
        """Considers shrinking the internal arrays based on their current
        size vs their max size."""

        # consider shrinking id lookup
        if len(cls.id_lookup) >= cls.length * 1.8 and cls.length > 0:
            greatest_id_in_use = np.argwhere(cls.id_lookup != -1)[-1]
            if greatest_id_in_use <= cls.length * 1.4:
                cls.id_lookup = _reallocate_array(
                    cls.id_lookup, greatest_id_in_use + 1, -1
                )
                cls._counter = itertools.count(greatest_id_in_use + 1)
                cls._recycled_ids = list(
                    filter(
                        lambda id: id < greatest_id_in_use, cls._recycled_ids
                    )
                )

        # consider shrinking component data arrays
        if cls.length <= cls._max_length * 0.65:
            cls.reallocate(cls.length)


def _reallocate_array(array, new_length, fill=0):
    """Allocates a new array with new_length and copies old data back into
    the array. Empty space created will be filled with fill value."""

    new_length = max(int(new_length), _STARTING_LENGTH)
    new_array = np.empty(new_length, array.dtype)
    new_array[:] = fill
    if len(array) <= new_length:
        new_array[: len(array)] = array
    else:
        new_array[:] = array[:new_length]
    return new_array
