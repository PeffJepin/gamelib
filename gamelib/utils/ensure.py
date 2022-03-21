def ensure(condition, error_message):
    def wrap(func):
        return _Enforcer(func, condition, error_message)
    return wrap


class Ensure:
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
