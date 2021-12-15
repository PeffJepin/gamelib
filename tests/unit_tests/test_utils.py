from gamelib.utils import (
    MethodMarker,
)


class ExampleClass:
    def __init__(self, value):
        self.value = value

    @MethodMarker
    def get_value(self):
        return self.value

    @MethodMarker
    def triple_value(self):
        return 3 * self.value

    @MethodMarker(type="testing", extra=5)
    def with_extra(self):
        return self.value + 5


def test_finding_marked_methods():
    instance = ExampleClass(5)
    expected = {
        MethodMarker(ExampleClass.get_value): instance.get_value,
        MethodMarker(ExampleClass.triple_value): instance.triple_value,
        MethodMarker(ExampleClass.with_extra, type="testing", extra=5):
            instance.with_extra,
    }
    assert MethodMarker.lookup(instance) == expected


def test_finding_marked_methods_by_type_attr():
    instance = ExampleClass(10)
    expected = {
        MethodMarker(ExampleClass.with_extra, type="testing", extra=5):
            instance.with_extra
    }
    assert MethodMarker.lookup(instance, type="testing") == expected


def test_calling_the_methods_regularly():
    instance = ExampleClass(5)

    assert instance.get_value() == 5
    assert instance.triple_value() == 15
    assert instance.with_extra() == 10


def example_customization(string):
    return MethodMarker(type="my_type", extra=string)


class CustomExample:
    @example_customization("hello")
    def custom_decorator(self):
        pass

    @MethodMarker
    def normal_decorator(self):
        pass


def test_creating_a_new_decorator():
    instance = CustomExample()
    expected = {
        MethodMarker(CustomExample.custom_decorator, type="my_type",
                     extra="hello"): instance.custom_decorator
    }
    assert MethodMarker.lookup(instance, type="my_type") == expected