#version 440

#include "HABufferImplementation.hglsl"

in vec2 vertexTexCoord;
in vec4 vertexPosition;

uniform sampler2D textureSampler;

FragmentData computeData()
{
  FragmentData data;
  data.color = texture(textureSampler, vertexTexCoord);
  data.pos = vertexPosition;

  return data;
}

#include "buildHABuffer.hglsl"

