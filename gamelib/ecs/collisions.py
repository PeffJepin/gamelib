import numpy as np

from gamelib import geometry
from gamelib.ecs import base
from gamelib.ecs import Transform


class Hitbox(base.Component):
    bvh: geometry.BVH

    @classmethod
    def create(cls, model_or_bvh, target_density=64):
        if isinstance(model_or_bvh, geometry.BVH):
            bvh = model_or_bvh
        else:
            bvh = geometry.collisions.BVH.create_tree(
                model_or_bvh, target_density
            )
        return cls(bvh)


def nearest_entity_hit(ray):
    best = None
    best_distance = geometry.Ray.MAX_DISTANCE

    def _skip_check(bvh):
        nearest_point = np.minimum(
            np.abs(ray.origin - bvh.aabb.min),
            np.abs(ray.origin - bvh.aabb.max),
        )
        return np.all(nearest_point > best_distance)

    for entity_type in base.Entity.get_subclasses(components=(Hitbox,)):

        transform_mask = entity_type.get_mask(Transform)
        if transform_mask is not None:
            transform_ids = transform_mask.ids

        for i, bvh in enumerate(entity_type.get_mask(Hitbox).bvh):
            if _skip_check(bvh):
                continue

            if transform_mask is not None:
                transform = Transform.get(transform_ids[i])
                ray.to_object_space(transform)

            dist = ray.collides_bvh(bvh)
            if dist is False:
                continue
            elif dist < best_distance:
                best_distance = dist
                best = base.Entity.get(entity_type.ids[i])

        ray.reset_transform()

    return best
