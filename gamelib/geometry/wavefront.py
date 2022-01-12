"""Internal module for parsing .obj files.

Current limitations:
    Vertices are always made up of 3 components
    Only vertices, vertex_normals and faces are parsed. (v, vn, f)
    Simple triangulation on faces that aren't length 3.
    Does not parse .mtl yet.
"""

import dataclasses

import numpy as np

from gamelib.geometry import transforms
from gamelib.geometry import base
from gamelib import gl


@dataclasses.dataclass
class _PreProcessorData:
    nverts: int
    ntris: int
    normal_lookup: dict


def parse(path) -> base.Model:
    with open(path, "r") as f:
        lines = f.readlines()
        ppd = _PreProcessorData(0, 0, {})

        for line in lines:
            _preprocess_line(line, ppd)

        model = _init_model(ppd)
        _parse_lines(lines, model, ppd)
        return model


def _init_model(ppd) -> base.Model:
    vertices = np.zeros(ppd.nverts * 3, gl.float)
    triangles = np.zeros(ppd.ntris * 3, gl.uint)
    if ppd.normal_lookup:
        normals = np.zeros(ppd.nverts * 3, gl.float)
    else:
        normals = None
    return base.Model(vertices, normals, triangles)


def _parse_lines(lines, model, ppd) -> None:
    triangles_pointer = 0
    vertices_pointer = 0
    normal_counter = 1
    for line in lines:
        spec, *data = line.split(" ")
        if spec == "v":
            values = [float(d) for d in data if d != ""]
            model.vertices[vertices_pointer : vertices_pointer + 3] = values
            vertices_pointer += 3

        elif spec == "vn":
            values = [float(d) for d in data if d != ""]
            normal = transforms.normalize(np.array(values, gl.float))
            index = ppd.normal_lookup[normal_counter] - 1
            start = index * 3
            model.normals[start : start + 3] = normal
            normal_counter += 1

        elif spec == "f":
            values = [d for d in data if d != ""]
            cleaned_values = []
            for value in values:
                if "/" in value:
                    v, *_ = value.split("/")
                else:
                    v = value
                cleaned_values.append(int(v) - 1)
            for i in range(len(cleaned_values) - 2):
                tri = [
                    cleaned_values[0],
                    cleaned_values[i + 1],
                    cleaned_values[i + 2],
                ]
                model.triangles[
                    triangles_pointer : triangles_pointer + 3
                ] = tri
                triangles_pointer += 3


def _preprocess_line(line, ppd) -> None:
    spec, *data = line.split(" ")

    if spec == "v":
        ppd.nverts += 1

    if spec == "f":
        values = [d for d in data if d != ""]
        ppd.ntris += len(values) - 2
        for value in values:
            if "/" in value:
                v, vt, vn = value.split("/")
                if vn != "":
                    ppd.normal_lookup[int(vn)] = int(v)
