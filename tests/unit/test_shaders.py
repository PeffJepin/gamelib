import functools

import pytest

from gamelib.rendering import shaders
from gamelib.core import gl

from ..conftest import compare_glsl

Shader = functools.partial(shaders.Shader, init_gl=False)


def test_parsing_single_string_shader():
    src = r"""
#version 330

#vert
void main()
{
    int i = 1;
}

#tesc
void main()
{
    int i = 2;
}

#tese
void main()
{
    int i = 3;
}

#geom
void main()
{
    int i = 4;
}

#frag
void main()
{
    int i = 5;
}
"""
    shader = Shader(src=src)
    expected_template = """
#version 330

void main()
{{
    int i = {};
}}
"""

    assert compare_glsl(shader.code.vert, expected_template.format(1))
    assert compare_glsl(shader.code.tesc, expected_template.format(2))
    assert compare_glsl(shader.code.tese, expected_template.format(3))
    assert compare_glsl(shader.code.geom, expected_template.format(4))
    assert compare_glsl(shader.code.frag, expected_template.format(5))


def test_parsing_vertex_shader_data():
    src = """
#version 330

#vert
in vec3 v_pos;
in float scale;
out vec3 world_pos;
void main() {}

#frag
in vec3 world_pos;
out vec4 frag;
void main() {}
"""

    shader = Shader(src=src)
    expected_vertex_attributes = (
        shaders.TokenDesc("v_pos", gl.vec3, 1),
        shaders.TokenDesc("scale", gl.float, 1),
    )
    expected_vertex_output = shaders.TokenDesc("world_pos", gl.vec3, 1)

    for tok in expected_vertex_attributes:
        assert shader.meta.attributes[tok.name] == tok
    assert len(shader.meta.attributes) == len(expected_vertex_attributes)

    assert shader.meta.vertex_outputs["world_pos"] == expected_vertex_output
    assert len(shader.meta.vertex_outputs) == 1


def test_parsing_uniforms():
    src = """
#version 330
uniform float u_float;

#vert
in vec3 v_pos;
out vec3 world_pos;
uniform vec4 u_vec4;
void main(){}

#frag
in vec3 world_pos;
out vec4 frag;
uniform vec4 u_vec4;
uniform mat4 u_mat4[2];
void main() {}
"""

    shader = Shader(src=src)
    expected_uniforms = (
        shaders.TokenDesc("u_float", gl.float, 1),
        shaders.TokenDesc("u_vec4", gl.vec4, 1),
        shaders.TokenDesc("u_mat4", gl.mat4, 2),
    )

    for tok in expected_uniforms:
        assert shader.meta.uniforms[tok.name] == tok
    assert len(shader.meta.uniforms) == 3


def test_shader_source_code_hash():
    src = """
#version 330
#vert
in vec3 v_pos;
void main(){}
"""
    shader1 = Shader(src=src)
    shader2 = Shader(src=src)
    shader3 = Shader(src=src + "some difference")

    assert hash(shader1) == hash(shader2)
    assert hash(shader1) != hash(shader3)


def test_error_when_no_source_is_given():
    with pytest.raises(ValueError) as excinfo:
        shader = Shader()

    assert "no source specified" in str(excinfo.value).lower()


def test_error_when_name_and_src_are_both_given():
    with pytest.raises(ValueError) as excinfo:
        shader = Shader("filename", src="my source string")

    assert "mutually exclusive" in str(excinfo.value).lower()
