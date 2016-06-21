#version 440
#extension GL_NV_gpu_shader5 : enable

in vec4 vertexColor;
in flat int vertexDrawId;

layout(location = 0) out vec4 outputColor;
layout(location = 1) out vec4 outputColor2;
layout(location = 2) out vec4 outputColor3;
layout(location = 3) out vec4 outputColor4;
layout(depth_any) out float gl_FragDepth;

layout(std430, binding = 1) buffer CB1
{
  int layerIndex[];
};

void setColorForLayer(int layerIndex, vec4 color)
{
  if (layerIndex == 0)
  {
    outputColor = color;
  }
  else if (layerIndex == 1)
  {
    outputColor2 = color;
  }
  else if (layerIndex == 2)
  {
    outputColor3 = color;
  }
  else
  {
    outputColor4 = color;
  }
}

void main()
{
  setColorForLayer(layerIndex[vertexDrawId], vertexColor);
}

