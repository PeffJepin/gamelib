def ensure(condition, error_message):
    """Decorator which ensures a condition is true before the decorated
    function is called. If not, it prints the error message.

    Example
    -------
    @ensure(lamda: window.exists, "You need to initialize the window first!")
    def resize_window():
        ...
    """

    def wrap(func):
        return _Enforcer(func, condition, error_message)

    return wrap


class Ensure:
    """Creates a reusable version of the ensure decorator.

    Example
    -------
    ensure_window = Ensure(
        lambda: window.exists,
        "You need to initialize the window first!"
    )

    @ensure_window
    def resize_window():
        ...
    """

    def __init__(self, condition, error_message):
        self.condition = condition
        self.error_message = error_message

    def __call__(self, func):
        return _Enforcer(func, self.condition, self.error_message)


class _Enforcer:
    def __init__(self, func, condition, error_message):
        self.func = func
        self.ensure = ensure
        self.condition = condition
        self.error_message = error_message
        self._owner = None

    def __call__(self, *args, **kwargs):
        if self.condition():
            if self._owner is not None:
                args = (self._owner, *args)
            return self.func(*args, **kwargs)
        else:
            raise AssertionError(
                f"{self.error_message}\n"
                f"This message resulted from trying to call: {self!r}"
            )

    def __set_name__(self, owner, name):
        self._owner = owner

    def __repr__(self):
        return repr(self.func)
