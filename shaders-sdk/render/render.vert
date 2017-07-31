#version 460

layout ( location = 0 ) in vec2 position;
layout ( location = 0 ) out vec2 texcoord;

void main()
{
    texcoord = fma(position, vec2(0.5f), vec2(0.5f));
    gl_Position = vec4(position, 0.0f, 1.0f);
}
