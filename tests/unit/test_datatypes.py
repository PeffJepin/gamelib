import math

import pytest
import numpy as np
from gamelib.core.vectors import Vec2, Vec3, Vec4

from ..conftest import assert_approx


@pytest.fixture(params=[Vec2(1, 2), Vec3(1, 2, 3), Vec4(1, 2, 3, 4)])
def generic_vector(request):
    yield request.param


class TestVectors:
    @pytest.mark.parametrize(
        "vector, as_tuple",
        (
            (Vec2(1, 2), (1, 2)),
            (Vec3(1, 2, 3), (1, 2, 3)),
            (Vec4(1, 2, 3, 4), (1, 2, 3, 4)),
        ),
    )
    def test_iter(self, vector, as_tuple):
        assert tuple(vector) == as_tuple

    @pytest.mark.parametrize(
        "vector, compare_to",
        (
            (Vec2(2, 1), (2, 1)),
            (Vec3(3, 2, 1), (3, 2, 1)),
            (Vec4(4, 3, 2, 1), (4, 3, 2, 1)),
        ),
    )
    def test_equality(self, vector, compare_to):
        assert vector == compare_to
        vals = tuple(vector)
        vector = type(vector)(*reversed(vals))
        assert vector != compare_to

    def test_negation(self):
        assert -Vec3(1, 1, 1) == Vec3(-1, -1, -1)
        assert -Vec3(-4, -2, 3) == Vec3(4, 2, -3)

    def test_getitem(self):
        assert Vec3(1, 2, 3)[0] == 1
        assert Vec3(1, 2, 3)[1] == 2
        assert Vec3(1, 2, 3)[2] == 3
        assert Vec4(1, 2, 3, 6)[3] == 6

        with pytest.raises(IndexError):
            v = Vec3(1, 2, 3)[3]

    def test_setitem(self, generic_vector):
        generic_vector[0] = 111
        assert generic_vector.x == 111
        generic_vector[1] = 222
        assert generic_vector.y == 222

    @pytest.mark.parametrize(
        "vector, other, expected",
        (
            (Vec2(1, 2), Vec2(1, 2), Vec2(2, 4)),
            (Vec3(1, 2, 3), Vec3(1, 2, 4), Vec3(2, 4, 7)),
            (Vec4(1, 2, 3, 4), Vec4(1, 2, 4, 4), Vec4(2, 4, 7, 8)),
            (Vec4(1, 1, 1, 1), 5, Vec4(6, 6, 6, 6)),
            (Vec4(1, 1, 1, 1), (1, 2, 3, 4), Vec4(2, 3, 4, 5)),
        ),
    )
    def test_addition(self, vector, other, expected):
        vec_type = type(vector)

        result = vector + other
        assert result == expected
        assert isinstance(result, vec_type)

        result = other + vector
        assert result == expected
        assert isinstance(result, vec_type)

        vector += other
        assert vector == expected
        assert isinstance(vector, vec_type)

    @pytest.mark.parametrize(
        "vector, other, expected",
        (
            (Vec2(1, 2), Vec2(1, 2), Vec2(0, 0)),
            (Vec3(1, 2, 3), Vec3(1, 2, 4), Vec3(0, 0, -1)),
            (Vec4(1, 2, 3, 10), Vec4(1, 2, 4, 4), Vec4(0, 0, -1, 6)),
            (Vec4(1, 1, 1, 1), 5, Vec4(-4, -4, -4, -4)),
            (Vec4(1, 1, 1, 1), (1, 2, 3, 4), Vec4(0, -1, -2, -3)),
        ),
    )
    def test_subtraction(self, vector, other, expected):
        vec_type = type(vector)

        result = vector - other
        assert result == expected
        assert isinstance(result, vec_type)

        result = other - vector
        assert result == -type(vector)(*expected)
        assert isinstance(result, vec_type)

        vector -= other
        assert vector == expected
        assert isinstance(vector, vec_type)

    @pytest.mark.parametrize(
        "vector, other, expected",
        (
            (Vec2(1, 2), Vec2(1, 2), Vec2(1, 4)),
            (Vec3(1, 2, 3), Vec3(1, 2, 4), Vec3(1, 4, 12)),
            (Vec4(1, 2, 3, 10), Vec4(1, 2, 4, 4), Vec4(1, 4, 12, 40)),
            (Vec4(1, 1, 1, 1), -5, Vec4(-5, -5, -5, -5)),
            (Vec4(1, 1, 1, 1), (1, 2, 3, -4), Vec4(1, 2, 3, -4)),
        ),
    )
    def test_multiplication(self, vector, other, expected):
        vec_type = type(vector)

        result = vector * other
        assert result == expected
        assert isinstance(result, vec_type)

        result = other * vector
        assert result == expected
        assert isinstance(result, vec_type)

        vector *= other
        assert vector == expected
        assert isinstance(vector, vec_type)

    @pytest.mark.parametrize(
        "vector, other, expected",
        (
            (Vec2(1, 2), Vec2(1, 2), Vec2(1, 1)),
            (Vec3(1, 2, 3), Vec3(1, 2, 4), Vec3(1, 1, 3 / 4)),
            (Vec4(1, 2, 3, 10), Vec4(1, 2, 4, 4), Vec4(1, 1, 3 / 4, 10 / 4)),
            (Vec4(1, 1, 1, 1), -5, Vec4(-1 / 5, -1 / 5, -1 / 5, -1 / 5)),
            (Vec4(1, 1, 1, 1), (1, 2, 3, -4), Vec4(1, 1 / 2, 1 / 3, -1 / 4)),
        ),
    )
    def test_division(self, vector, other, expected):
        vec_type = type(vector)

        result = vector / other
        assert result == expected
        assert isinstance(result, vec_type)

        result = other / vector
        assert result == 1 / expected
        assert isinstance(result, vec_type)

        vector /= other
        assert vector == expected
        assert isinstance(vector, vec_type)

    @pytest.mark.parametrize(
        "vector, other, expected",
        (
            (Vec2(1, 2), Vec2(1, 2), 5),
            (Vec3(1, 2, 3), Vec3(1, 2, 4), 17),
            (Vec4(1, 2, 3, 10), Vec4(1, 2, 4, 4), 57),
            (Vec4(1, 1, 1, 1), (1, 2, -3, -4), -4),
        ),
    )
    def test_dot(self, vector, other, expected):
        assert vector.dot(other) == expected

    def test_cross(self):
        vector = Vec3(2, 3, 4)
        vec_type = type(vector)
        expected = (-3, 6, -3)

        result = vector.cross(Vec3(5, 6, 7))
        assert result == expected
        assert isinstance(result, vec_type)

        result = vector.cross((5, 6, 7))
        assert result == expected
        assert isinstance(result, vec_type)

    def test_magnitude(self):
        assert Vec3(1, 0, 0).magnitude == pytest.approx(1)
        assert Vec3(4, 0, 0).magnitude == pytest.approx(4)
        assert Vec2(1, 1).magnitude == pytest.approx(math.sqrt(2))
        assert Vec3(1, 1, 1).magnitude == pytest.approx(math.sqrt(3))
        assert Vec4(1, 1, 1, 1).magnitude == pytest.approx(math.sqrt(4))

    def test_normalize(self, generic_vector):
        ratios_before = (comp / sum(generic_vector) for comp in generic_vector)
        generic_vector.normalize()
        ratios_after = (comp / sum(generic_vector) for comp in generic_vector)

        assert_approx(ratios_before, ratios_after)
        assert generic_vector.magnitude == pytest.approx(1)

    def test_normalize_zero(self):
        for v in (Vec2(0, 0), Vec3(0, 0, 0), Vec4(0, 0, 0, 0)):
            before = tuple(v)
            v.normalize()
            assert v == before

    def test_inverse(self):
        result = Vec3(1, -2, 3).inverse()
        assert result == (1 / 1, -1 / 2, 1 / 3)
        assert isinstance(result, Vec3)

    def test_inverse_zero(self):
        result = Vec3(0, 2, 0).inverse()
        assert result == (math.inf, 1 / 2, math.inf)
        assert isinstance(result, Vec3)

    def test_multiplying_across_an_array_doesnt_pollute_array_type(self):
        array = np.arange(15).reshape(-1, 3)
        array = array * Vec3(1, 1, 1)
        assert not isinstance(array, Vec3)

        array = Vec3(1, 1, 1) * array
        assert not isinstance(array, Vec3)
