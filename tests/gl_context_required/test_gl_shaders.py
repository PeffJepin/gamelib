import pathlib

import numpy as np
import pytest
import gamelib

from gamelib import gl
from gamelib import shaders


@pytest.fixture(autouse=True, scope="module")
def init_ctx():
    yield gamelib.init(make_window=False)
    gamelib.exit()


class FakeLock:
    def __init__(self):
        self.times_used = 0

    def __enter__(self):
        self.times_used += 1

    def __exit__(self, *args):
        pass


class TestIndexBuffer:
    def test_indices_from_num_entities(self):
        buffer = shaders.IndexBuffer(order=(0, 1, 2, 0, 2, 3), num_entities=2)

        expected = np.array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7])
        assert np.all(buffer.indices.astype(int) == expected)

    def test_changing_number_of_entities(self):
        buffer = shaders.IndexBuffer(order=(0, 1, 2, 0, 2, 3), num_entities=2)
        buffer.num_entities = 3

        expected = np.array(
            [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11]
        )
        assert np.all(buffer.indices.astype(int) == expected)

    def test_indices_from_entities(self):
        buffer = shaders.IndexBuffer(order=(0, 1, 2, 0, 2, 3), entities=(0, 2))

        expected = np.array([0, 1, 2, 0, 2, 3, 8, 9, 10, 8, 10, 11])
        assert np.all(buffer.indices.astype(int) == expected)

    def test_changing_entities(self):
        buffer = shaders.IndexBuffer(order=(0, 1, 2, 0, 2, 3), entities=(0, 1))
        buffer.entities = (0, 1, 2)

        expected = np.array(
            [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11]
        )
        assert np.all(buffer.indices.astype(int) == expected)


class TestVertexBuffer:
    def test_reading_after_sourced(self):
        array = np.zeros(10)
        buffer = shaders.VertexBuffer(array)

        assert all(buffer.read() == array)

    def test_updates_from_given_array(self):
        array = np.zeros(10)
        buffer = shaders.VertexBuffer(array)

        array += 100
        buffer.update()

        assert all(array == buffer.read())

    def test_writing_a_numpy_array(self):
        array = np.zeros(10)
        buffer = shaders.VertexBuffer(array)

        new_data = np.array(list(range(10)), int)
        buffer.write(new_data)

        assert all(new_data == buffer.read())

    def test_writing_bytes(self):
        array = np.zeros(10)
        buffer = shaders.VertexBuffer(array)

        new_data = np.arange(10).tobytes()
        buffer.write(new_data)

        assert new_data == buffer.read(bytes=True)

    def test_dtype_coercion(self):
        array = np.array([1, 2, 3])
        buffer = shaders.VertexBuffer(array, dtype="f4")

        assert array.astype("f4").tobytes() == buffer.read(bytes=True)

    def test_can_use_context_manager_for_writing(self):
        array = np.zeros((10,), int)
        lock = FakeLock()
        buffer = shaders.VertexBuffer(array, lock=lock)

        assert lock.times_used == 0
        buffer.update()
        assert lock.times_used == 1
        buffer.write(array)
        assert lock.times_used == 2
        buffer.read()
        assert lock.times_used == 2


class TestShaderProgram:
    simple_vertex_source = """
        #version 330
        out int value;
        void main() {
            value = gl_VertexID;
        }
    """

    @staticmethod
    def assert_simple_transform(ctx, moderngl_program):
        # using moderngl api to make sure ShaderProgram
        # creates a valid gl shader program
        vao = ctx.vertex_array(moderngl_program, [])
        buffer = ctx.buffer(reserve=10 * 4)
        vao.transform(buffer, vertices=10)

        expected = np.arange(10, dtype="i4").tobytes()
        assert buffer.read() == expected

    def test_loading_a_shader_with_source_string(self):
        program = shaders.ShaderProgram(
            vertex_shader=self.simple_vertex_source,
            varyings=["value"],
        )
        self.assert_simple_transform(program.gl.ctx, program.gl)

    def test_loading_a_shader_with_source_file(self, tmpdir):
        path = pathlib.Path(tmpdir) / "temporary_vertex_shader_source"
        with open(path, "w") as f:
            f.write(self.simple_vertex_source)

        program = shaders.ShaderProgram(vertex_shader=path, varyings=["value"])
        self.assert_simple_transform(program.gl.ctx, program.gl)

    def test_transform_given_vertices(self):
        program = shaders.ShaderProgram(
            vertex_shader="""
                #version 330
                
                out int value;
                
                void main() 
                {
                    value = 123;
                }
            """,
            varyings=["value"],
        )

        result = program.transform(vertices=4)
        expected = np.array([123] * 4, gl.int)
        assert all(result == expected)

    def test_transform_multiple_outputs_given_vertices(self):
        program = shaders.ShaderProgram(
            vertex_shader="""
                #version 330
                
                out int test_int_output;
                out vec2 test_vec2_output;
                
                void main()
                {
                    test_int_output = 321;
                    test_vec2_output = vec2(123, 456);
                }
            """,
            varyings=["test_int_output", "test_vec2_output"],
        )

        result = program.transform(vertices=10)
        expected_ints = np.array([321] * 10, gl.int)
        expected_vec2s = np.array([(123, 456)] * 10, gl.vec2)
        assert result["test_int_output"].tobytes() == expected_ints.tobytes()
        assert result["test_vec2_output"].tobytes() == expected_vec2s.tobytes()

    def test_transform_must_calculate_result_buffer_size(self):
        in_values = np.array([(1, 2), (3, 4), (5, 6)])
        program = shaders.ShaderProgram(
            vertex_shader="""
                #version 330
                
                in vec2 test_inputs;
                out float x;
                out float y;
                
                void main()
                {
                    x = test_inputs.x;
                    y = test_inputs.y;
                } 
            """,
            varyings=["x", "y"],
            buffers={"test_inputs": in_values},
        )

        result = program.transform()
        expected = in_values.astype(gl.float)
        assert result.tobytes() == expected.tobytes()

    def test_inspection(self):
        program = shaders.ShaderProgram(
            vertex_shader="""
                #version 330
                in vec2 v_pos;
                void main() 
                {
                    gl_Position = vec4(v_pos, 0, 1);
                } 
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D atlas;
                out vec4 frag;
                void main() 
                {
                    frag = texture(atlas, vec2(0, 0));
                }
            """,
        )
        assert "v_pos" in program.vertex_attributes
        assert len(program.vertex_attributes) == 1
        assert "atlas" in program.uniforms
        assert len(program.uniforms) == 1

    def test_automatic_uniform_sourcing(self, glsl_dtype_and_input):
        gl_type, input_value = glsl_dtype_and_input
        if gl_type == "sampler2D" or gl_type.startswith("b"):
            # not applicable
            return

        uniform = np.array(input_value)
        program = shaders.ShaderProgram(
            vertex_shader=f"""
                #version 330
                
                uniform {gl_type} test_input;
                out {gl_type} test_output;
                
                void main() 
                {{
                    test_output = test_input; 
                }}
            """,
            varyings=["test_output"],
            uniforms={"test_input": uniform},
        )

        expected = gl.coerce_array(uniform, gl_type)
        assert program.transform(1).tobytes() == expected.tobytes()

        uniform += 1
        expected = gl.coerce_array(uniform, gl_type)
        assert program.transform(1).tobytes() == expected.tobytes()

    def test_uniform_array_support(self, glsl_dtype_and_input):
        gl_type, input_value = glsl_dtype_and_input
        if gl_type == "sampler2D" or gl_type.startswith("b"):
            # not applicable
            return

        arr1 = np.array(input_value)
        arr2 = arr1 + 7
        uniform = np.vstack((arr1, arr2))
        program = shaders.ShaderProgram(
            vertex_shader=f"""
                #version 330

                uniform {gl_type} test_input[2];
                out {gl_type} test_output;

                void main() 
                {{
                    test_output = test_input[gl_VertexID]; 
                }}
            """,
            varyings=["test_output"],
            uniforms={"test_input": uniform},
        )

        expected = gl.coerce_array(uniform, gl_type)
        assert program.transform(vertices=2).tobytes() == expected.tobytes()

    def test_use_uniforms_no_current_uniforms(self):
        program = shaders.ShaderProgram(
            vertex_shader="""
                #version 330

                uniform int input1;
                uniform int input2;
                out int output_value;

                void main() 
                {
                    output_value = input1 + input2;
                }
            """,
            varyings=["output_value"],
        )
        uni1, uni2 = np.array([0], gl.int), np.array([100], gl.int)
        program.use_uniforms(input1=uni1, input2=uni2)
        assert program.transform(1) == uni1 + uni2

        uni1 += 12
        uni2 += 11
        assert program.transform(1) == uni1 + uni2

    def test_use_uniforms_with_existing_uniforms_in_place(self):
        uni1, uni2 = np.array([0], gl.int), np.array([100], gl.int)
        program = shaders.ShaderProgram(
            vertex_shader="""
                #version 330

                uniform int input1;
                uniform int input2;
                out int output_value;

                void main() 
                {
                    output_value = input1 + input2;
                }
            """,
            varyings=["output_value"],
            uniforms={"input1": uni1, "input2": uni2},
        )
        assert program.transform(1) == uni1 + uni2

        uni3, uni4 = np.array([9], gl.int), np.array([120], gl.int)
        program.use_uniforms(input1=uni3, input2=uni4)
        assert program.transform(1) == uni3 + uni4

        uni3 += 12
        uni4 += 11
        assert program.transform(1) == uni3 + uni4

    def test_buffers_auto_coercing_their_data_type(self, glsl_dtype_and_input):
        gl_type, valid_input_value = glsl_dtype_and_input
        if gl_type in ("sampler2D", "double") or gl_type.startswith("b"):
            # skip a few dtypes that don't apply
            # rather than make a ton of test fixtures
            return

        array = np.array(valid_input_value)
        buffer = np.vstack((array, array, array, array))
        program = shaders.ShaderProgram(
            vertex_shader=f"""
                #version 330
                in {gl_type} inval;
                out {gl_type} outval;
                
                void main()
                {{
                    outval = inval;
                }}
            """,
            varyings=["outval"],
            buffers={"inval": buffer},
        )
        assert np.all(program.transform() == gl.coerce_array(buffer, gl_type))

    def test_use_buffers_no_current_buffers(self):
        program = shaders.ShaderProgram(
            vertex_shader="""
                #version 330
                
                in int input1;
                in int input2;
                out int output_value;
                
                void main() 
                {
                    output_value = input1 + input2;
                }
            """,
            varyings=["output_value"],
        )
        array1 = np.arange(6, dtype=gl.int)
        array2 = np.arange(6, 12, dtype=gl.int)

        program.use_buffers(input1=array1, input2=array2)
        assert np.all(program.transform() == array1 + array2)

        array1 += 10
        array2 += 20
        assert np.all(program.transform() == array1 + array2)

    def test_use_buffer_buffers_already_there(self):
        array1 = np.arange(6, dtype=gl.int)
        array2 = np.arange(6, 12, dtype=gl.int)
        program = shaders.ShaderProgram(
            vertex_shader="""
                #version 330

                in int input1;
                in int input2;
                out int output_value;

                void main() 
                {
                    output_value = input1 + input2;
                }
            """,
            varyings=["output_value"],
            buffers={"input1": array1, "input2": array2},
        )
        assert np.all(program.transform() == array1 + array2)

        array3 = np.arange(100, dtype=gl.int)
        array4 = np.arange(100, dtype=gl.int)
        program.use_buffers(input1=array3, input2=array4)
        assert np.all(program.transform() == array3 + array4)

        array3 += 100
        array4 += 33
        assert np.all(program.transform() == array3 + array4)
