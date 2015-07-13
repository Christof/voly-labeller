#version 440

#include "HABufferImplementation.hglsl"

in vec3 vertexTexCoord3d;
in vec4 vertexPosition;

FragmentData computeData()
{
  FragmentData data;
  data.color.rgb = vertexTexCoord3d;
  data.color.a = 0.5;
  data.pos = vertexPosition;

  return data;
}

#include "buildHABuffer.hglsl"

