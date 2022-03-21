def ensure(condition, error_message):
    def wrap(func):
        return _Enforcer(func, Ensure(condition, error_message))
    return wrap


class Ensure:
    def __init__(self, condition, error_message):
        self.condition = condition
        self.error_message = error_message

    def __call__(self, func):
        return _Enforcer(func, self)


class _Enforcer:
    def __init__(self, func, ensure):
        self.func = func
        self.ensure = ensure
        self._owner = None

    def __call__(self, *args, **kwargs):
        if self.ensure.condition():
            if self._owner is not None:
                args = (self._owner, *args)
            self.func(*args, **kwargs)
        else:
            raise AssertionError(self.ensure.error_message)

    def __set_name__(self, owner, name):
        self._owner = owner
