#version 440
#extension GL_NV_gpu_shader5 : enable

#include "layerOutput.hglsl"

in vec4 vertexColor;
in flat int vertexDrawId;

layout(std430, binding = 1) buffer CB1
{
  int layerIndex[];
};

layout(std430, binding = 2) buffer CB2
{
  float alpha[];
};

void main()
{
  vec4 color = mix(vec4(0), vertexColor, alpha[vertexDrawId]);
  setColorForLayer(layerIndex[vertexDrawId], color);
}

