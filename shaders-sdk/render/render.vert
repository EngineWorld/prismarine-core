#version 450

#extension GL_ARB_separate_shader_objects : require
#extension GL_ARB_shading_language_420pack : require

layout(location = 0) in vec2 position;

out vec2 texcoord;

void main()
{
    texcoord = fma(position, vec2(0.5f), vec2(0.5f));
    gl_Position = vec4(position, 0.0f, 1.0f);
}
