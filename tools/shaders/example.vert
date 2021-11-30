#version 330

in vec2 v_pos;
in vec3 v_col;
out vec3 color;

uniform vec2 offset;

void main()
{
    gl_Position = vec4(v_pos + offset, 0, 1);
    color = v_col;
}