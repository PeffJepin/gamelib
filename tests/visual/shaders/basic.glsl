#version 330

#vert
out vec3 color;

void main()
{
    if (gl_VertexID == 0)
    {
        gl_Position = vec4(-1, -1, 0, 1);
        color = vec3(1, 0, 0);
    }
    else if (gl_VertexID == 1)
    {
        gl_Position = vec4(0, 1, 0, 1);
        color = vec3(0, 1, 0);
    }
    else
    {
        gl_Position = vec4(1, -1, 0, 1);
        color = vec3(0, 0, 1);
    }
}

#frag
in vec3 color;
out vec4 frag;

void main()
{
    frag = vec4(color, 1);
}

