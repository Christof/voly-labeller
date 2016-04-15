#version 440
#extension GL_NV_gpu_shader5 : enable
#include "bindlessTexture.hglsl"

in vec2 vertexTexCoord;
in flat int vertexDrawId;

layout(location = 0) out vec4 outputColor;
layout(location = 1) out vec4 outputColor2;
layout(location = 2) out vec4 outputColor3;
layout(location = 3) out vec4 outputColor4;
layout(depth_any) out float gl_FragDepth;

layout(std430, binding = 1) buffer CB1
{
  Tex2DAddress texAddress[];
};

void setColorForLayer(int layerIndex, vec4 color)
{
  if (layerIndex == 0)
  {
    outputColor = color;
    outputColor.r = 0;
  }
  else if (layerIndex == 1)
  {
    outputColor2 = color;
    outputColor2.g = 0;
  }
  else if (layerIndex == 2)
  {
    outputColor3 = color;
    outputColor3.b = 0;
  }
  else
  {
    outputColor4 = color;
  }
}

void main()
{
  Tex2DAddress address = texAddress[vertexDrawId];
  vec4 color = Texture(address, vertexTexCoord.xy);
  setColorForLayer(address.dummy, color);
}

