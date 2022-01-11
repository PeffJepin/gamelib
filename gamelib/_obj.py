"""Internal module for parsing .obj files.

Current limitations:
    Vertices are always made up of 3 components
    Only vertices, vertex_normals and faces are parsed. (v, vn, f)
    Simple triangulation on faces that aren't length 3.
"""

import dataclasses
from typing import Optional

import numpy as np

from gamelib import geometry
from gamelib.geometry import transforms
from gamelib import gl


@dataclasses.dataclass
class _PreProcessorData:
    nverts: int
    ntris: int
    has_normals: bool
    normal_lookup: Optional[dict]



def parse(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        ppd = _preprocess_lines(lines)
        return _parse_lines(lines, ppd)


def _init_arrays(ppd):
    vertices = np.zeros(ppd.nverts * 3, gl.float)
    triangles = np.zeros(ppd.ntris * 3, gl.uint)
    if ppd.has_normals:
        normals = np.zeros(ppd.nverts * 3, gl.float)
    else:
        normals = None
    return geometry.Geometry(vertices, normals, triangles)


def _parse_lines(lines, ppd):
    geo = _init_arrays(ppd)
    triangles_pointer = 0
    vertices_pointer = 0
    normal_counter = 1
    for line in lines:
        spec, *data = line.split(" ")
        if spec == "v":
            values = [float(d) for d in data if d != ""]
            geo.vertices[vertices_pointer:vertices_pointer+3] = values
            vertices_pointer += 3

        elif spec == "vn":
            values = [float(d) for d in data if d != ""]
            normal = transforms.normalize(np.array(values, gl.float))
            index = ppd.normal_lookup[normal_counter] - 1
            start = index * 3
            geo.normals[start:start+3] = normal
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
                    cleaned_values[i+1],
                    cleaned_values[i+2],
                ]
                geo.triangles[triangles_pointer:triangles_pointer+3] = tri
                triangles_pointer += 3
    return geo


def _preprocess_lines(lines) -> _PreProcessorData:
    nverts = 0
    ntris = 0
    has_normals = False
    normal_lookup = dict()

    for line in lines:
        spec, *data = line.split(" ")
        if spec == "v":
            nverts += 1
        if spec == "f":
            values = [d for d in data if d != ""]
            ntris += (len(values) - 2)
            for value in values:
                if "/" in value:
                    v, vt, vn = value.split("/")
                    if vn != "":
                        has_normals = True
                        normal_lookup[int(vn)] = int(v)
        
    return _PreProcessorData(nverts, ntris, has_normals, normal_lookup)


