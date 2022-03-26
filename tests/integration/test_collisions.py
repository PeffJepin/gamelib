from gamelib.geometry import transforms
from gamelib.geometry import collisions

from ..conftest import assert_approx


def test_transforming_a_ray_into_object_space():
    transform = transforms.Transform(
        (10, 20, 30), (4, 4, 8), (0, 0, 1), 90
    )
    ray = collisions.Ray((10, 0, 30), (0, 1, 0))

    ray.to_object_space(transform)

    assert_approx(ray.origin, (-5, 0, 0))
    assert_approx(ray.direction, (1, 0, 0))


def test_transforming_multiple_times():
    transform1 = transforms.Transform(
        (10, 20, 30), (4, 4, 8), (0, 0, 1), 90
    )
    transform2 = transforms.Transform(
        (10, 40, 30), (4, 4, 8), (0, 0, 1), 90
    )
    ray = collisions.Ray((10, 0, 30), (0, 1, 0))

    ray.to_object_space(transform1)
    ray.to_object_space(transform2)

    # the second transform should still have access to the original ray
    # data and transform according to that.
    assert_approx(ray.origin, (-10, 0, 0))
    assert_approx(ray.direction, (1, 0, 0))


def test_resetting_transform():
    transform = transforms.Transform(
        (10, 20, 30), (4, 4, 8), (0, 0, 1), 90
    )
    ray = collisions.Ray((10, 0, 30), (0, 1, 0))

    ray.to_object_space(transform)
    ray.reset_transform()

    assert_approx(ray.origin, (10, 0, 30))
    assert_approx(ray.direction, (0, 1, 0))
