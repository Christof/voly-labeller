#version 440

#include "HABufferImplementation.hglsl"

in vec4 vertexColor;
in vec4 vertexPosition;

FragmentData computeData()
{
  FragmentData data;
  data.color = vertexColor;
  data.pos = vertexPosition;

  return data;
}

#include "buildHABuffer.hglsl"

