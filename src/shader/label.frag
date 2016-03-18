#version 440
#extension GL_NV_gpu_shader5 : enable
#include "bindlessTexture.hglsl"

in vec2 vertexTexCoord;
in flat int vertexDrawId;

layout(location = 0) out vec4 outputColor;
layout(location = 1) out vec4 position;
layout(location = 2) out vec4 outputColor2;
layout(location = 3) out vec4 position2;
layout(location = 4) out vec4 outputColor3;
layout(location = 5) out vec4 position3;
layout(location = 6) out vec4 outputColor4;
layout(location = 7) out vec4 position4;
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
    outputColor.r = 1;
    position = vec4(1);
  }
  else if (layerIndex == 1)
  {
    outputColor2 = color;
    outputColor2.g = 1;
    position2 = vec4(1);
  }
  else if (layerIndex == 2)
  {
    outputColor3 = color;
    outputColor3.b = 1;
    position3 = vec4(1);
  }
  else
  {
    outputColor4 = color;
    position4 = vec4(1);
  }
}

void main()
{
  Tex2DAddress address = texAddress[vertexDrawId];
  vec4 color = Texture(address, vertexTexCoord.xy);
  setColorForLayer(address.dummy, color);
}

