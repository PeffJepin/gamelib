#version 330

in vec3 color;
out vec4 frag;

void main()
{
    frag = vec4(color, 1);
}