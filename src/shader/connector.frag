#version 440
#extension GL_NV_gpu_shader5 : enable

#include "layerOutput.hglsl"

in vec4 vertexColor;
in flat int vertexDrawId;

layout(std430, binding = 1) buffer CB1
{
  int layerIndex[];
};

void main()
{
  setColorForLayer(layerIndex[vertexDrawId], vertexColor);
}

