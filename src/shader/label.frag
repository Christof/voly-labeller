#version 440

#include "HABufferImplementation.hglsl"

in vec2 vertexTexCoord;
in vec4 vertexPosition;
in vec4 vertexEyePosition;

uniform sampler2D textureSampler;

FragmentData computeData()
{
  FragmentData data;
  data.color = texture(textureSampler, vertexTexCoord);
  data.eyePos = vertexEyePosition;
  data.objectId = 0;

  return data;
}

#include "buildHABuffer.hglsl"

