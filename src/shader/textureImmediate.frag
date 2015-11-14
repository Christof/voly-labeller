#version 440
#extension GL_NV_gpu_shader5 : enable
#include "bindlessTexture.hglsl"

in vec2 vertexTexCoord;
in flat int vertexDrawId;

out vec4 color;

layout(std430, binding = 1) buffer CB1
{
  Tex2DAddress texAddress[];
};

void main()
{
  color = Texture(texAddress[vertexDrawId], vertexTexCoord.xy);
}

