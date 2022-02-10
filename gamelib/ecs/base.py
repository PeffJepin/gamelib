"""This module defines the base classes for an ECS framework.

Examples
--------

Components are used to create contiguous arrays of memory and offer convenient
access to both the elements in the internal arrays and the arrays themselves.

Components should be annotated in the same way dataclasses would be annotated.
They support the VectorType datatypes found in gamelib.core.vectors and also
support the OpenGL datatypes defined in gamelib.gl.

>>> class Physical(Component):
...     pos: gamelib.Vec2
...     mass: float

>>> for i in range(3):
...     Physical.create((i, i), (1 + i))


Access to the internal arrays can be made through the type object.

>>> Physical.position
array([[0., 0.],
       [1., 1.],
       [2., 2.]])

>>> Physical.mass
array([1., 2., 3.])

>>> Physical.ids
array([0, 1, 2])


Access to individual elements into the array can be made through an instances
interface.

>>> obj = Physical.create((10, 10), 5)
>>> obj
<Physical(id=3, pos=[10. 10.], mass=5.0)>

>>> obj.position = (123, 123)
>>> obj
<Physical(id=3, pos=[123. 123.], mass=5.0)>

>>> Physical.position
array([[  0.,   0.],
       [  1.,   1.],
       [  2.,   2.],
       [123., 123.]])

>>> obj.mass += 25
>>> Physical.mass
array([ 1.,  2.,  3., 30.])


The data can also be manipulated through either the array or instance interface
and will remain in sync, as they are both using the same data store.

>>> Physical.mass *= 10
>>> Physical.mass
array([ 10.,  20.,  30., 300.])

>>> obj.mass
300.0

>>> Physical.position -= (50, 50)
>>> Physical.position
array([[-50., -50.],
       [-49., -49.],
       [-48., -48.],
       [ 73.,  73.]])


Data can be accessed and destroyed by the Component ids. Note that when
component data is destroyed, the original order of the data is NOT preserved.

>>> Physical.ids
array([0, 1, 2, 3])

>>> obj1 = Physical.get(1)
>>> obj1
<Physical(id=1, pos=[-49. -49.], mass=20.0)>

>>> Physical.destroy(1)
>>> Physical.ids
array([0, 3, 2])

>>> obj1.position
None
>>> obj1.mass
None
>>> Physical.get(1)
None


The internal allocation and deallocation of memory is handled internally by the
component.

>>> Physical.internal_length
10
>>> len(Physical)
3

>>> for _ in range(10):
...     Physical.create((0, 0), 0)
>>> Physical.internal_length
16
>>> len(Physical)
13

>>> for i in range(10):
...     Physical.destroy(i)
>>> Physical.internal_length
10
>>> len(Physical)
3

>>> Physical.clear()
>>> Physical.ids
array([], dtype=int64)
>>> len(Physical)
0


Entities are used to bind component data together underneath a single identity.
The following examples will use this set of simple example classes:

>>> class Physical(Component):
...     pos: gamelib.Vec2
...     mass: float

>>> class Motion(Component):
...     vel: gamelib.Vec2
...     acc: gamelib.Vec2

>>> class MovingObject(Entity):
...     physical: Physical
...     motion: Motion

>>> class StaticObject(Entity):
...     physical: Physical


Entities are very similar to components in that they are just meant to organize
and facilitate access to data. One important difference to note is that Entity
instances share a global unique id pool. Note below how the StaticObject and
MovingObject don't share any ids, while Physical and Motion do.

>>> for i in range(5):
...     MovingObject.create(Physical((i, i), i), Motion((i, i), (-i, -i)))

>>> for i in range(5):
...     StaticObject.create(Physical((100 * i, 100 * i), 100 * i))

>>> Physical.ids
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> Motion.ids
array([0, 1, 2, 3, 4])
>>> MovingObject.ids
array([0, 1, 2, 3, 4])
>>> StaticObject.ids
array([5, 6, 7, 8, 9])
>>> Entity.ids
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


This distinction means you can load access to an entity from the Entity base
class with only an id value.

>>> Entity.get(0)
<MovingObject(id=0, physical=<Physical(id=0, pos=[0. 0.], mass=0.0)>, motion=<Motion(id=0, vel=[0. 0.], acc=[0. 0.])>)>
>>> Entity.get(5)
<StaticObject(id=5, physical=<Physical(id=5, pos=[0. 0.], mass=0.0)>)>


Like with Component, access into individual elements of the data arrays can
be made through an instance of an EntityType, while array access is available
through the type objects themselves.

The following is an example of direct array element manipulation.

>>> obj1 = Entity.get(1)
>>> obj1.physical
<Physical(id=1, pos=[1. 1.], mass=1.0)>
>>> obj1.motion
<Motion(id=1, vel=[1. 1.], acc=[-1. -1.])>
>>> obj1.physical.mass += 100
>>> obj1
<MovingObject(id=1, physical=<Physical(id=1, pos=[1. 1.], mass=101.0)>, motion=<Motion(id=1, vel=[1. 1.], acc=[-1. -1.])>)>

Access to the component arrays through an entity is a little more complex than
just getting access to the array, since it needs to be masked, as can be seen
by the return types of the following.

>>> StaticObject.physical
<gamelib.ecs._EntityMask at 0x7f875e6c46a0>
>>> MovingObject.motion
<gamelib.ecs._EntityMask at 0x7f875e5bddb0>
>>> MovingObject.physical.position
<gamelib.ecs._MaskedArrayProxy at 0x7f87364d7340>


These mask/proxy objects allow for array manipulation as follows:

>>> Physical.position
array([[  0.,   0.],
       [  1.,   1.],
       [  2.,   2.],
       [  3.,   3.],
       [  4.,   4.],
       [  0.,   0.],
       [100., 100.],
       [200., 200.],
       [300., 300.],
       [400., 400.]])

>>> MovingObject.physical.position += MovingObject.motion.vel
>>> Physical.position
array([[  0.,   0.],
       [  2.,   2.],
       [  4.,   4.],
       [  6.,   6.],
       [  8.,   8.],
       [  0.,   0.],
       [100., 100.],
       [200., 200.],
       [300., 300.],
       [400., 400.]])


Like with Component, Entity data does not preserve it's order through instances
being destroyed. The mask/proxy objects take care of this complexity for you.
(Also note that destroying an entity has destroyed it's bound components.)

>>> Entity.destroy(0)
>>> Entity.destroy(3)
>>> MovingObject.ids
array([4, 1, 2])

>>> Physical.position
array([[400., 400.],
       [  2.,   2.],
       [  4.,   4.],
       [300., 300.],
       [  8.,   8.],
       [  0.,   0.],
       [100., 100.],
       [200., 200.]])

>>> MovingObject.physical.position += 1_000
>>> Physical.position
array([[ 400.,  400.],
       [1002., 1002.],
       [1004., 1004.],
       [ 300.,  300.],
       [1008., 1008.],
       [   0.,    0.],
       [ 100.,  100.],
       [ 200.,  200.]])


Finally there is a distiction to be made between clearing Entity and clearing a
subclass of entity.

>>> Motion.ids
array([4, 1, 2])
>>> MovingObject.clear()
>>> Motion.ids
array([], dtype=int64)
>>> Physical.ids
array([9, 6, 5, 8, 7])
>>> Entity.ids
array([5, 6, 7, 8, 9])


Above we see that clearing MovingObject left StaticObject data untouched. If we
recreate another MovingObject for demonstration purposes, note that
Entity.clear() will clear the entire Entity-Component framework.

>>> MovingObject.create(Physical((0, 0), 1), Motion((1, 1), (-1, -1)))
>>> MovingObject.ids
array([0])
>>> StaticObject.ids
array([5, 6, 7, 8, 9])

>>> Entity.clear()
>>> Entity.ids
array([], dtype=int64)
>>> StaticObject.ids
array([], dtype=int64)
>>> MovingObject.ids
array([], dtype=int64)
>>> Physical.ids
array([], dtype=int64)
>>> Motion.ids
array([], dtype=int64)
"""

import collections
import itertools

from typing import Dict
from typing import Iterable
from typing import Any

import numpy as np

from gamelib.core import vectors


_STARTING_LENGTH = 10


class IdGenerator:
    """A utility for generating unique ids for objects."""

    def __init__(self, start=0):
        """Create a unique id generator.

        Parameters
        ----------
        start : int, optional
            The beginning id.
        """

        self._counter = itertools.count(start)
        self._prev = -1
        self._largest = -1
        self._recycled = []

    def __repr__(self):
        if self._recycled:
            next_id = self._recycled[0]
        else:
            next_id = self._prev + 1
        return f"<IdGenerator(next={next_id})>"

    def __next__(self):
        """Get the next unique id. Will give the lowest id possible.

        Returns
        -------
        int
        """

        if self._recycled:
            id = self._recycled.pop(0)
        else:
            id = next(self._counter)
            self._prev = id

        self._largest = max(self._largest, id)
        return id

    @property
    def largest_active(self):
        """Returns the value of the largest id this generator has created that
        has not yet been recycled.

        Returns
        -------
        int
        """

        return self._largest

    def recycle(self, id):
        """Signal to the generator that you are done with this id, and it can
        be used again.

        Parameters
        ----------
        id : int

        Returns
        -------
        """

        if not self._recycled or self._recycled[-1] < id:
            self._recycled.append(id)
        else:
            for i, v in enumerate(self._recycled[:]):
                if id == v:
                    # dont allow duplicates
                    return
                if id < v:
                    self._recycled.insert(i, id)
                    break
        if id == self._largest:
            self._seek_largest()

    def set_state(self, value):
        """Set the id counter back to this value and recycled id's greater than
        or equal to this value.

        Parameters
        ----------
        value : int
            The id value that the id counter should be reverted to.
        """

        index = 0
        for i, v in enumerate(reversed(self._recycled)):
            if v < value:
                index = len(self._recycled) - i
                break
        self._recycled = self._recycled[:index]
        self._counter = itertools.count(value)

    def clamp(self):
        # convenience method
        self.set_state(self.largest_active + 1)

    def _seek_largest(self):
        # when the largest existing id is deleted we need to check through
        # the recycled ids in reverse until we find a break in the sequence,
        # marking the next largest id
        largest = self._recycled[-1]
        for val in reversed(self._recycled):
            if val == largest:
                largest -= 1
            else:
                break
        self._largest = largest


class _ComponentType(type):
    """Metaclass for Component.

    Responsible for granting access to an entire component array via the
    component type object as opposed to a component instance which deals in
    specific indices into that array.
    """

    # must be implemented on component classes
    fields: Dict[str, Any]  # any numpy compatible dtype
    arrays: Dict[str, np.ndarray]  # field -> internal array mapping
    _data_index: np.ndarray
    _active_length: int
    _initialized: bool = False

    def __len__(cls):
        """How many of this component actually exists."""

        return cls._active_length

    def __setattr__(cls, name, value):
        """Annotated attribute access from the type object represents the whole
        array, so this should set the value for the entire internal array."""

        if cls._initialized and name in cls.fields:
            cls.arrays[name][: cls._active_length] = value
        else:
            super().__setattr__(name, value)

    @property
    def ids(cls):
        """A masked view of the active ids."""

        return cls.arrays["id"][: cls._active_length]

    @property
    def internal_length(cls):
        """Get the full length of the internal data arrays."""

        return len(cls.arrays["id"])

    def get_subclasses(cls):
        """Gets a list of all subclasses defined below this one.

        Returns
        -------
        list[_ComponentType]
        """

        total_subclasses = []
        running_list = list(cls.__subclasses__())
        while running_list:
            c = running_list.pop(0)
            total_subclasses.append(c)
            running_list.extend(c.__subclasses__())
        return total_subclasses

    def get_index(cls, id_):
        """Gets the data index for component with given id.

        Parameters
        ----------
        id_ : int

        Returns
        -------
        int:
            Returns -1 if there is no component with this id.
        """

        return cls._data_index[id_]


class Component(metaclass=_ComponentType):
    """Component is a base class used for laying out data in contiguous
    memory. Attributes should be annotated like a dataclass and appropriate
    internal arrays will be managed with the annotated dtype."""

    fields: Dict[str, Any]  # any numpy compatible dtype
    arrays: Dict[str, np.ndarray]  # field -> internal array mapping
    _data_index: np.ndarray
    _active_length: int
    _initialized: bool = False
    _structured_dtype: np.dtype
    _id_gen: IdGenerator
    itemsize: int

    def __init_subclass__(cls, **kwargs):
        """Initialize the subclass based on what has been annotated."""

        cls._initialized = False
        annotations = cls.__dict__.get("__annotations__", {})
        if getattr(cls, "fields", None):
            cls.fields.update(annotations)
        else:
            cls.fields = annotations
        if not cls.fields:
            raise AttributeError("No attributes have been annotated.")
        for field in cls.fields:
            setattr(cls, field, _ComponentElement(cls, field))
        cls._init_arrays()
        dtypes = []
        for name, dtype in cls.fields.items():
            if dtype in vectors.VectorType.__subclasses__():
                dtypes.append((name, dtype.as_dtype()))
            elif isinstance(dtype, np.dtype):
                dtypes.append((name, dtype))
            else:
                dtypes.append((name, np.dtype(dtype)))
        dtypes.append(("id", int))
        cls._structured_dtype = np.dtype(dtypes)
        cls.itemsize = cls._structured_dtype.itemsize
        cls._initialized = True

    def __new__(cls, *args, _id=None, **kwargs):
        """Create some new Component data. If id is given this will load
        access into existing data instead.

        Parameters
        ----------
        *args : Any
            __init__ args
        _id : int, optional
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

        if _id is not None:
            # check if this id exists
            if cls._data_index[_id] == -1:
                return None
        else:
            _id = cls._create()

        instance = super().__new__(cls)
        instance._id = _id
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

        kwargs.pop("_id", None)  # __new__ kwarg not needed here
        if args:
            for field, value in zip(self.fields, args):
                setattr(self, field, value)
        else:
            for field, value in kwargs.items():
                if field in self.fields:
                    setattr(self, field, value)

    def __repr__(self):
        values = ", ".join(
            f"{field}={getattr(self, field)}" for field in self.fields
        )
        return f"<{self.__class__.__name__}(id={self.id}, {values})>"

    def __eq__(self, other):
        """A component compares for equality based elementwise comparison of
        it's annotated attributes."""

        if type(self) == type(other):
            return self.values == other.values
        else:
            return self.values == other

    @property
    def id(self):
        """Readonly access to the object's id."""

        return self._id

    @property
    def values(self):
        """Get the values for this instance's annotated attributes."""

        return tuple(getattr(self, name) for name in self.fields)

    @classmethod
    def create(cls, *args, **kwargs):
        """Since __new__ and __init__ args are tied closely together, custom
        creation procedures are best done through the create method to avoid
        conflicting with __new__ and __init__ signatures.

        A subclass can implement this signature however they like, and should
        just pass the proper component instances into the super().create call.

        Parameters
        ----------
        *args : Any
            Args will map to annotated attributes in the order they are given.
        **kwargs : Any
            Keys will map to annotated attribute names.

        Returns
        -------
        Component:
            An instance of this class.
        """

        return cls(*args, **kwargs)

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

        return cls(_id=id)

    @classmethod
    def view_raw_arrays(cls):
        """Gets the raw internal arrays (unmasked).

        Returns
        -------
        np.ndarray:
            The resulting array will have a structured dtype which is an
            aggregate of all the annotated attributes of this component.
        """

        combined = np.empty(cls.internal_length, cls._structured_dtype)
        for name in cls.fields:
            combined[name] = cls.arrays[name]
        combined["id"] = cls.arrays["id"]
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

        id = target.id if isinstance(target, cls) else target
        if id >= len(cls._data_index):
            # out of range, not a real component
            return

        index = cls._data_index[id]
        if index == -1:
            # not an active component
            return

        # swap good data on the end of the arrays into this data slot
        cls._active_length -= 1
        swapped_id = cls.arrays["id"][cls._active_length]
        for array in cls.arrays.values():
            array[index] = array[cls._active_length]
        cls._data_index[swapped_id] = index
        cls._data_index[id] = -1

        # maybe shrink arrays
        cls._consider_shrinking()
        cls._id_gen.recycle(id)

    @classmethod
    def clear(cls):
        """Resets the component and all subclasses to initial state."""

        if cls != Component:
            cls._init_arrays()
        for c in cls.get_subclasses():
            c._init_arrays()

    @classmethod
    def indices_from_ids(cls, ids):
        """Gets the indices into the internal arrays for the given ids.

        Parameters
        ----------
        ids : array-like

        Returns
        -------
        np.ndarray
        """

        return cls._data_index[ids]

    @classmethod
    def _consider_shrinking(cls):
        # shrink internal data arrays
        current_active_length = len(cls)
        if cls.internal_length > 1.7 * current_active_length:
            new_length = max(
                int(current_active_length * 1.3), _STARTING_LENGTH
            )
            if new_length == cls.internal_length:
                return
            for field, array in cls.arrays.items():
                array = cls.arrays[field]
                cls.arrays[field] = _reallocate_array(array, new_length, -1)

        # shrink the index
        necessary_length = cls._id_gen.largest_active + 1
        actual_length = len(cls._data_index)
        if actual_length >= 1.7 * necessary_length >= _STARTING_LENGTH:
            cls._data_index = _reallocate_array(
                cls._data_index, necessary_length + 1, -1
            )
            cls._id_gen.clamp()

    @classmethod
    def _init_arrays(cls):
        """Allocate the initial internal arrays."""

        cls.arrays = dict()
        for field, dtype in cls.fields.items():
            if isinstance(dtype, type) and issubclass(
                dtype, vectors.VectorType
            ):
                dtype = dtype.as_dtype()
            cls.arrays[field] = np.zeros(_STARTING_LENGTH, dtype)
        cls.arrays["id"] = np.zeros(_STARTING_LENGTH, int)

        cls._data_index = np.zeros(_STARTING_LENGTH, int)
        cls._data_index[:] = -1
        cls._active_length = 0
        cls._id_gen = IdGenerator()

    @classmethod
    def _create(cls):
        """Responsible for bookkeeping and setting aside storage if necessary
        when new instances are to be created."""

        id = next(cls._id_gen)
        index_length = len(cls._data_index)
        if index_length <= id:
            new_length = index_length * 1.4
            cls._data_index = _reallocate_array(
                cls._data_index, new_length, -1
            )

        index = cls._active_length
        if index >= len(cls):
            new_length = index * 1.4
            for field, arr in cls.arrays.items():
                cls.arrays[field] = _reallocate_array(arr, new_length, -1)
        cls._active_length += 1
        cls._data_index[id] = index
        cls.arrays["id"][index] = id
        return id


class _ComponentElement:
    """Descriptor for an annotated component field."""

    def __init__(self, owner: _ComponentType, field):
        self._owner = owner
        self._field = field
        self._dtype = owner.fields[field]
        self._should_view = False
        if self._dtype in vectors.VectorType.__subclasses__():
            self._should_view = True

    def __get__(self, obj, objtype=None):
        """Returns a value from the internal array when access is attempted
        from the context of an instance, otherwise returns a masked view of the
        entire array."""

        if obj is None:
            return self._owner.arrays[self._field][: len(self._owner)]

        data_index = self._owner.get_index(obj.id)
        if data_index == -1:
            return None

        value = self._owner.arrays[self._field][data_index]
        if self._should_view:
            return value.view(self._dtype)
        return value

    def __set__(self, obj, value):
        """Sets a single value into the internal array."""

        data_index = self._owner.get_index(obj.id)
        if data_index == -1:
            return
        self._owner.arrays[self._field][data_index] = value


class _EntityGlobalState:
    """State global to all entities. A single global instance should be
    accessible from the Entity class."""

    # class attributes
    entity_number_counter = itertools.count(1)  # start at 1, Entity is 0
    subclass_lookup: Dict[int, "_EntityType"] = dict()
    # keeps a set of entity numbers that can be looked up in subclass lookup
    subclasses_by_component_type = collections.defaultdict(set)

    # instance attributes
    data_index: np.ndarray
    type_index: np.ndarray
    id_counter: itertools.count
    existing: int

    def __init__(self):
        self.data_index = np.zeros(_STARTING_LENGTH, int)
        self.type_index = np.zeros(_STARTING_LENGTH, int)
        self.data_index[:] = -1
        self.type_index[:] = -1
        self.id_gen = IdGenerator()
        self.existing = 0

    def grow_index(self):
        new_length = len(self.data_index) * 1.4
        self.data_index = _reallocate_array(self.data_index, new_length, -1)
        self.type_index = _reallocate_array(self.type_index, new_length, -1)


class _EntityType(type):
    """Entity metaclass. Describes properties unique to entity type objects."""

    _global = _EntityGlobalState()

    # per-subclass
    arrays: Dict[str, np.ndarray]
    _length: int
    _entity_number: int

    def __hash__(cls):
        """Assume unique entity numbers."""

        return hash(cls._entity_number)

    def __eq__(cls, other):
        """Assume unique entity numbers."""

        return (
            isinstance(other, _EntityType)
            and cls._entity_number == other._entity_number
        )

    def __len__(cls):
        """Gets the number of Entities of this type that exists."""

        if cls == Entity:
            return cls._global.existing

        return cls._length

    def __iter__(cls):
        for i in cls.ids:
            e = cls.get(i)
            if e is not None:
                yield e

    @property
    def internal_length(cls):
        """Gets the length of the internal data arrays."""

        if cls == Entity:
            return len(cls._global.data_index)

        return len(cls.arrays["id"])

    @property
    def ids_proxy(cls):
        return lambda: cls.ids

    def get_subclasses(cls, components=None):
        """Gets all of the subclasses of Entity if called globally on Entity,
        or just the subclasses for the particular Entity if called on a
        subclass

        Parameters
        ----------
        components : Iterable, optional
            Optionally, you can filter for only entities that have the provided
            component types as fields.

        Returns
        -------
        list[Type[Entity]]
        """

        if components is not None:
            if not isinstance(components, Iterable):
                components = (components,)
            return cls._get_subclasses_by_contained_components(components)
        if cls == Entity:
            return list(cls._global.subclass_lookup.values())
        else:
            total = []
            running_list = [cls]
            while running_list:
                c = running_list.pop(0)
                total.append(c)
                running_list.extend(c.__subclasses__())
            return total

    def _get_subclasses_by_contained_components(cls, components):
        if cls == Entity:
            predicate = lambda e: True
        else:
            predicate = lambda e: issubclass(e, cls)

        first, *the_rest = components
        if predicate(first):
            entity_nums = cls._global.subclasses_by_component_type[first]
        else:
            entity_nums = set()

        for c in the_rest:
            if not predicate(c):
                continue
            these_nums = cls._global.subclasses_by_component_type[c]
            entity_nums = set.intersection(these_nums, entity_nums)
        return list(cls._global.subclass_lookup[n] for n in entity_nums)

    @property
    def ids(cls):
        if cls == Entity:
            return np.where(Entity._global.data_index != -1)[0]
        return cls.arrays["id"][: cls._length]

    @property
    def existing(cls):
        """How many entities are currently active. When called from Entity gets
        ALL entities. When called from a subclass gets just the amount of
        entities in that particular subclass.

        Returns
        -------
        int
        """

        if cls == Entity:
            return cls._global.existing
        return cls._length


class Entity(metaclass=_EntityType):
    """Entities are classes used to relate several component instances together
    underneath a single identity.

    Use like a dataclass, type annotating attributes for components this entity
    type will tie together. Note that all annotated datatypes should be
    subclasses of Component."""

    arrays: Dict[str, np.ndarray]
    fields: Dict[str, _ComponentType]
    _length: int
    _entity_number = 0  # derived automatically for subclasses
    _field_by_type: Dict[_ComponentType, str]

    def __init_subclass__(cls):
        """Inspect the annotated attributes and initialize internals
        accordingly."""

        cls._entity_number = next(cls._global.entity_number_counter)
        cls._global.subclass_lookup[cls._entity_number] = cls
        annotations = cls.__dict__.get("__annotations__", {})
        if getattr(cls, "fields", None):
            cls.fields.update(annotations)
        else:
            cls.fields = annotations
        if not cls.fields:
            raise AttributeError("No attributes have been annotated.")
        cls._field_by_type = dict()
        for field, component_type in cls.fields.items():
            cls._global.subclasses_by_component_type[component_type].add(
                cls._entity_number
            )
            cls._field_by_type[component_type] = field
            setattr(cls, field, _BoundComponent(cls, field, component_type))
        cls._init_arrays()
        cls._length = 0

    def __new__(cls, *args, _id=None, **kwargs):
        """Either load an existing entity or create a new one.

        Parameters
        ----------
        *args : Any
            __init__ args
        _id : int, optional
            Internal for loading an entity that already exists.
        **kwargs : Any
            __init__ kwargs
        """

        if _id is None:
            _id = cls._create()
        elif cls._global.type_index[_id] == -1:
            return None
        obj = super().__new__(cls)
        obj._id = _id
        return obj

    def __init__(self, *args, **kwargs):
        """Initialize values. Args/kwargs are mutually exclusive.

        Parameters
        ----------
        *args : Any
            If args are given they are assigned to annotated attributes in the
            order they were annotated.
        **kwargs : Any
            If kwargs are given then their keys will map to the names given to
            the annotated attributes.
        """

        kwargs.pop("_id", None)  # __new__ kwarg not needed here
        if args:
            for field, arg in zip(self.fields, args):
                setattr(self, field, arg)
        elif kwargs:
            for field, arg in kwargs.items():
                setattr(self, field, arg)

    def __repr__(self):
        components = ", ".join(
            f"{name}={getattr(self, name)}" for name in self.fields
        )
        return f"<{self.__class__.__name__}(id={self._id}, {components})>"

    def __eq__(self, other):
        """Elementwise comparison of each of this instance's components to the
        others. Always false if type(other) != type(self).

        Parameters
        ----------
        other : Any

        Returns
        -------
        bool
        """

        if not isinstance(other, type(self)):
            return False
        else:
            return self.id == other.id

    def __iter__(self):
        """Iterate over the components this entity is bound to."""

        return iter(getattr(self, name) for name in self.fields)

    @property
    def id(self):
        """This entities unique id."""

        return self._id

    def get_component(self, component_type):
        """Gets the first component of the given type found bound to this
        entity

        Parameters
        ----------
        component_type : Type[Component]

        Returns
        -------
        Component:
            An instance of the parameter component type.
        """

        field = self._field_by_type.get(component_type, None)
        if field is None:
            return None
        return getattr(self, field)

    @classmethod
    def create(cls, *args, **kwargs):
        """Since __new__ and __init__ args are tied closely together, custom
        creation procedures are best done through the create method to avoid
        conflicting with __new__ and __init__ signatures.

        A subclass can implement this signature however they like, and should
        just pass the proper component instances into the super().create call.

        Parameters
        ----------
        *args : Any
            Args will map to annotated attributes in the order they are given.
        **kwargs : Any
            Keys will map to annotated attribute names.

        Returns
        -------
        Entity:
            An instance of this class.
        """

        return cls(*args, **kwargs)

    @classmethod
    def has_field(cls, component_type):
        """Check if an entity type has a field of a particular type of
        component.

        Parameters
        ----------
        component_type: Type[Component]

        Returns
        -------
        bool
        """

        return component_type in cls._field_by_type

    @classmethod
    def get_mask(cls, component_type):
        """Gets an annotated field based on component type. Useful when you
        have an entity type you're unsure about and want a particular type of
        component from it.

        Parameters
        ----------
        component_type : Type[Component]

        Returns
        -------
        _EntityMask
        """

        field = cls._field_by_type.get(component_type, None)
        if not field:
            return None
        return getattr(cls, field)

    @classmethod
    def get_component_ids(cls, component_type):
        """Gets the component ids for the given type of component which are
        associated with this type of Entity.

        Returns
        -------
        np.ndarray
        """

        field = ""
        for name, type in cls.fields.items():
            if type is component_type:
                field = name
                break
        if not field:
            raise ValueError(f"Couldn't find a field for {component_type!r}.")
        return cls.arrays[field][: cls._length]

    @classmethod
    def get(cls, id):
        """Load an existing entity instance.

        Parameters
        ----------
        id : int

        Returns
        -------
        Entity | None
            Returns None if no entity with this id was found.
        """

        if id >= len(cls._global.data_index):
            return None
        if cls == Entity:
            entity_num = cls._global.type_index[id]
            if entity_num == -1:
                return None
            entity_type = cls._global.subclass_lookup[entity_num]
        else:
            entity_type = cls
        return entity_type(_id=id)

    @classmethod
    def destroy(cls, target):
        """Destroys this entity and each component bound to it.

        Parameters
        ----------
        target : int | Entity
            Can be either an Entity instance to destroy or the id.
        """

        if isinstance(target, Entity):
            id = target.id
            instance = target
        else:
            id = target
            instance = Entity.get(id)

        # don't modify anything if this isn't actually an existing entity
        if instance is None:
            return

        entity_type = type(instance)

        for comp in instance:
            if comp is not None:
                comp.destroy(comp.id)

        # we want to swap this entry's data to the end and decrement the length
        entity_type._length -= 1
        entity_index = cls._global.data_index[id]
        # if it's already the final element we don't need to swap
        if entity_index != entity_type._length:
            # swap the end entity data into this destroyed entities data slot
            id_at_the_end = entity_type.arrays["id"][entity_type._length]
            for array in entity_type.arrays.values():
                array[entity_index] = array[entity_type._length]
            # update the global index
            cls._global.data_index[id_at_the_end] = entity_index

        # finally, mask of that deleted entity which is now at the end
        cls._global.data_index[id] = -1
        cls._global.type_index[id] = -1
        cls._global.existing -= 1
        cls._global.id_gen.recycle(id)
        entity_type._consider_shrinking()

    @classmethod
    def clear(cls):
        """Shorthand for destroying all of this type of Entity, meaning it will
        also destroy bound components."""

        if cls == Entity:
            _EntityType._global = _EntityGlobalState()
            for e in Entity.get_subclasses():
                e._init_arrays()
            Component.clear()
            return

        for c in [cls] + cls.get_subclasses():
            for field, comp_type in c.fields.items():
                for i in c.arrays[field][: c._length]:
                    comp_type.destroy(i)
            for i in c.ids:
                cls._global.id_gen.recycle(i)

            cls._global.data_index[c.ids] = -1
            cls._global.type_index[c.ids] = -1
            c._init_arrays()

    @classmethod
    def _init_arrays(cls):
        """Allocate the initial internal data storage."""

        cls.arrays = {
            field: np.zeros(_STARTING_LENGTH, int) for field in cls.fields
        }
        cls.arrays["id"] = np.zeros(_STARTING_LENGTH, int)
        cls._length = 0

    @classmethod
    def _create(cls):
        """Ensure there is space for a new entity and manages all the internal
        bookkeeping for creating a new instance."""

        assert cls != Entity, "should be a subclass"
        id_ = next(cls._global.id_gen)

        if id_ >= len(cls._global.data_index):
            cls._global.grow_index()
        data_index = cls._get_new_data_index()
        cls._global.data_index[id_] = data_index
        cls._global.type_index[id_] = cls._entity_number
        cls._global.existing += 1
        cls.arrays["id"][data_index] = id_
        return id_

    @classmethod
    def _get_new_data_index(cls):
        val = cls._length
        cls._length += 1
        if cls._length >= len(cls):
            cls._grow_arrays()
        return val

    @classmethod
    def _grow_arrays(cls):
        new_length = int(len(cls) * 1.4)
        for name, array in cls.arrays.items():
            cls.arrays[name] = _reallocate_array(array, new_length, fill=-1)

    @classmethod
    def _consider_shrinking(cls):
        # shrink internal data arrays
        if len(cls) > 1.7 * cls._length:
            new_length = max(int(cls._length * 1.2), _STARTING_LENGTH)
            if new_length == len(cls):
                return
            for field, array in cls.arrays.items():
                array = cls.arrays[field]
                cls.arrays[field] = _reallocate_array(array, new_length, -1)

        # shrink the index
        necessary_length = cls._global.id_gen.largest_active + 1
        actual_length = len(cls._global.data_index)
        if actual_length >= 1.7 * necessary_length >= _STARTING_LENGTH:
            cls._global.data_index = _reallocate_array(
                cls._global.data_index, necessary_length + 1, -1
            )
            cls._global.id_gen.clamp()


class _EntityMask:
    """Responsible for masking a component so that it only show the indices
    which are associated with a certain Entity type. This is a barebones
    implementation for now."""

    _initialized = False

    def __init__(self, entity_type, component_type):
        """The component and entity types this mask should act upon."""

        self._entity = entity_type
        self._component = component_type
        self._initialized = True

    def __getattr__(self, name):
        """Retrieve the array from the component type specified in __init__,
        but masked to contain only the indices relevant to this entity type.

        Note that this uses advanced numpy indexing, which means this returns
        a copy of the array, not a view. In the future this class may help to
        get around this problem by implementing mathematical dunder methods."""

        if name not in self._component.fields and name != "ids":
            raise AttributeError(f"name not in {self._component.fields!r}.")
        ids = self._entity.get_component_ids(self._component)
        if name == "ids":
            return ids
        indices = self._component.indices_from_ids(ids)
        proxy = _MaskedArrayProxy(getattr(self._component, name), indices)
        return proxy

    def __setattr__(self, name, value):
        if not self._initialized:
            super().__setattr__(name, value)
        elif name in self._component.fields:
            if isinstance(value, _MaskedArrayProxy):
                # the only time an array proxy is passed in here
                # is when is has been called like the following:
                #   _EntityMask.some_component_attribute += 1_000
                # The proxy takes care of the += operation and is
                # trying to be rebound the the _EntityMask object,
                # which we don't want
                return

            # otherwise we might be trying to set the entity mask something
            # like so:
            #   _EntityMask.attr1 = _EntityMask.attr2 + 100
            # Here we are passing in an array like to replace the current
            # array so we can overwrite the storage directly
            ids = self._entity.get_component_ids(self._component)
            indices = self._component.indices_from_ids(ids)
            getattr(self._component, name)[indices] = value
        else:
            super().__setattr__(name, value)

    @property
    def _indices(self):
        ids = self._entity.get_component_ids(self._component)
        return self._component.indices_from_ids(ids)

    def proxy(self, field):
        return lambda: (
            arr[self._indices]
            if len(arr := getattr(self._component, field)) > 0
            else arr
        )


class _MaskedArrayProxy:
    def __init__(self, raw_array, indices):
        self._array = raw_array
        self._indices = indices

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        args = []
        for arg in inputs:
            if isinstance(arg, type(self)):
                args.append(arg._array[arg._indices])
            else:
                args.append(arg)

        return getattr(ufunc, method)(*args, **kwargs)

    def __add__(self, other):
        return self._array[self._indices] + other

    def __iadd__(self, other):
        self._array[self._indices] += other
        return self

    def __sub__(self, other):
        return self._array[self._indices] - other

    def __isub__(self, other):
        self._array[self._indices] -= other
        return self

    def __mul__(self, other):
        return self._array[self._indices] * other

    def __imul__(self, other):
        self._array[self._indices] *= other
        return self

    def __truediv__(self, other):
        return self._array[self._indices] / other

    def __itruediv__(self, other):
        self._array[self._indices] /= other
        return self

    def __floordiv__(self, other):
        return self._array[self._indices] // other

    def __ifloordiv__(self, other):
        self._array[self._indices] //= other
        return self

    def __eq__(self, other):
        return self._array[self._indices] == other

    def __iter__(self):
        return iter(self._array[self._indices])


class _BoundComponent:
    """Descriptor describing a component instance that has been bound to an
    entity."""

    def __init__(self, owner, field, component_type):
        self._component_type = component_type
        self._field = field
        self._cached_name = "__cached_component_" + field
        self._owner = owner

    def __get__(self, obj, objtype=None):
        """Gets an instance of the described component that is bound to the
        given entity if called on an instance of entity. If called on a
        subclass of entity, this will instead return a masked view of the
        described component."""

        if obj is None:
            return _EntityMask(self._owner, self._component_type)
        data_index = _EntityType._global.data_index[obj.id]
        if data_index == -1:
            return None
        cached = getattr(obj, self._cached_name, None)
        if cached is not None:
            return cached
        component_id = obj.arrays[self._field][data_index]
        component_instance = self._component_type.get(component_id)
        setattr(obj, self._cached_name, component_instance)
        return component_instance

    def __set__(self, obj, component):
        """Binds this entity to the given component."""

        data_index = _EntityType._global.data_index[obj.id]
        if data_index == -1:
            return
        obj.arrays[self._field][data_index] = component.id

    def __repr__(self):
        return f"<_ComponentDescriptor(type={self._component_type})>"


def _reallocate_array(array, new_length, fill=0):
    """Allocates a new array with new_length and copies old data back into
    the array. Empty space created will be filled with fill value."""

    new_length = max(int(new_length), _STARTING_LENGTH)
    old_length, *dims = array.shape
    new_array = np.empty((new_length, *dims), array.dtype)
    new_array[:] = fill
    if len(array) <= new_length:
        new_array[: len(array)] = array
    else:
        new_array[:] = array[:new_length]
    return new_array


# TODO: An option to allocate the internal arrays using the structured dtype
#   instead of individual arrays. Using a decorator approach like dataclass
#   uses might offer a more suitable API for optional features like this.

# TODO: Eventually I will want the option to allocate the arrays using
#   shared memory. Since they are regularly reallocated this will probably
#   require a new shared memory module so other processes can easily find the
#   internal array after it has been reallocated.
