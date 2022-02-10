class MethodMarker:
    """This class is meant to be used as a decorator to mark methods. It may be
    later extended to be used on functions as well as bound methods.

    Examples
    --------

    Using MethodMarker as a decorator directly:

    >>> class MyClass:
    ...     @MethodMarker
    ...     def method1(self):
    ...         pass
    ...
    ...     @MethodMarker(type="AnythingHashable", extra="AlsoAnythingHashable")
    ...     def using_kwargs(self):
    ...         pass
    ...
    >>> instance = MyClass()
    >>> MethodMarker.lookup(instance)
    {
        <MethodMarker(owner=MyClass, name='method1')>:
            <bound method MyClass.method1 of <__main__.MyClass object at 0x7fb643988ee0>>,
        <MethodMarker(owner=MyClass, name='using_kwargs')>:
            <bound method MyClass.using_kwargs of <__main__.MyClass object at 0x7fb643988ee0>>
    }


    Filtering markers by type:

    >>> MethodMarker.lookup(instance, type="AnythingHashable")
    {<MethodMarker(owner=MyClass, name='using_kwargs')>:
        <bound method MyClass.using_kwargs of <__main__.MyClass object at 0x7fb643988ee0>>}


    Customizing the decorator by wrapping it:

    >>> def my_customized_decorator(my_param):
    ...     return MethodMarker(type="my_deco", extra=my_param)
    ...
    >>> class MyClass:
    ...     @my_customized_decorator(5)
    ...     def custom(self):
    ...         pass
    ...
    ...     @MethodMarker
    ...     def regular(self):
    ...         pass
    ...
    >>> MethodMarker.lookup(MyClass(), type="my_deco")
    {<MethodMarker(owner=MyClass, name='custom')>:
        <bound method MyClass.custom of <__main__.MyClass object at 0x7fb640543070>>}
    """

    _INJECTION_ATTRIBUTE = "_gamelib_marker_"

    def __new__(cls, func=None, /, *, type=None, extra=None):
        """Checks if the call was done with parenthesis or not and returns
        a wrapper or instance of MethodMarker accordingly.

        Parameters
        ----------
        func : Function
            Should be defined on some class as method.
        type : Any, optional
            Should be hashable. Used to filter markers.
        extra : Any, optional
            Should be hashable. Any extra data to attach.
        """

        def wrapper(f):
            instance = super(MethodMarker, cls).__new__(cls)
            instance.type = type
            instance.extra = extra
            instance.func = f
            return instance

        # called with params, capture the function with wrapper
        if func is None:
            return wrapper

        # called without params, return directly
        return wrapper(func)

    def __set_name__(self, owner, name):
        """When binding the object to a class, instead bind the wrapped
        function and add this marker to a list maintained on the owner."""

        self.name = name
        self.owner = owner

        setattr(owner, name, self.func)
        existing_injection = getattr(owner, self._INJECTION_ATTRIBUTE, None)
        if existing_injection:
            existing_injection.append(self)
        else:
            setattr(owner, self._INJECTION_ATTRIBUTE, [self])

    def __eq__(self, other):
        if not isinstance(other, MethodMarker):
            return False
        return hash(self) == hash(other)

    def __hash__(self):
        return hash((self.func, self.type, self.extra))

    def __repr__(self):
        return (
            f"<MethodMarker(owner={self.owner.__name__}, name"
            f"={self.name!r})>"
        )

    @classmethod
    def lookup(cls, instance, type=None):
        """Return a dict of markers mapped to their bound methods on this
        particular instance. Optionally filter on MethodMarker.type.

        Parameters
        ----------
        instance : object
            Any object, will return {} if no markers exist.
        type : Any, optional
            Filter out markers that don't match this type.

        Returns
        -------
        dict[MethodMarker, BoundMethod]:
            A dictionary mapping markers to the method they are marking.
        """

        return {
            mark: getattr(instance, mark.name, None)
            for mark in getattr(instance, cls._INJECTION_ATTRIBUTE, ())
            if type is None or mark.type == type
        }
