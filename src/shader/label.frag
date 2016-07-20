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

layout(std430, binding = 2) buffer CB2
{
  float alpha[];
};

void main()
{
  Tex2DAddress address = texAddress[vertexDrawId];
  vec4 color = Texture(address, vertexTexCoord.xy);
  color.a *= alpha[vertexDrawId];
  int layerIndex = address.dummy;
  setColorForLayer(layerIndex, color);
}

