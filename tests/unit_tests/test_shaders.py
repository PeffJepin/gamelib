from gamelib import gl
from gamelib.rendering.shaders import TokenDesc, parse_source


class TestSourceParsing:
    example = """
        #version 330

        in vec3 v_pos;
        in float scale;

        out uint out_val;

        uniform vec2 uniform_array[2];
        uniform mat3 proj;

        void main() {}
    """

    def test_parsing_in_variables(self):
        parsed = parse_source(self.example)
        assert TokenDesc(name="v_pos", dtype=gl.vec3, len=1) in parsed.inputs
        assert TokenDesc(name="scale", dtype=gl.float, len=1) in parsed.inputs
        assert len(parsed.inputs) == 2

    def test_parsing_out_variables(self):
        parsed = parse_source(self.example)
        assert (
            TokenDesc(name="out_val", dtype=gl.uint, len=1) in parsed.outputs
        )
        assert len(parsed.outputs) == 1

    def test_parsing_uniforms(self):
        parsed = parse_source(self.example)
        assert (
            TokenDesc(name="uniform_array", dtype=gl.vec2, len=2)
            in parsed.uniforms
        )
        assert TokenDesc(name="proj", dtype=gl.mat3, len=1) in parsed.uniforms
        assert len(parsed.uniforms) == 2
