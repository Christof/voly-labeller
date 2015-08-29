#version 440

#include "HABufferImplementation.hglsl"

in vec3 vertexTexCoord3d;
in vec4 vertexPosition;
in vec4 vertexEyePosition;

FragmentData computeData()
{
  FragmentData data;
  data.color.rgb = vertexTexCoord3d;
  data.color.a = 0.5;
  data.eyePos = vertexEyePosition;

  return data;
}

#include "buildHABuffer.hglsl"

