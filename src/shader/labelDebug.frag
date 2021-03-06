#version 440
#extension GL_NV_gpu_shader5 : enable
#include "bindlessTexture.hglsl"
#include "layerOutput.hglsl"

in vec2 vertexTexCoord;
in flat int vertexDrawId;

layout(std430, binding = 1) buffer CB1
{
  Tex2DAddress texAddress[];
};

void main()
{
  Tex2DAddress address = texAddress[vertexDrawId];
  vec4 color = Texture(address, vertexTexCoord.xy);
  setColorForLayerDebug(address.dummy, color);
}

