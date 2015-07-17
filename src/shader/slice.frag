#version 440

#include "HABufferImplementation.hglsl"

in vec2 vertexTexCoord;
in vec4 vertexPosition;

uniform sampler3D textureSampler;

FragmentData computeData()
{
  vec3 texCoord = vec3(vertexTexCoord.x, vertexTexCoord.y, 0);
  float value = texture(textureSampler, texCoord).r;

  FragmentData data;
  data.color = vec4(value, value, value, 1.0f);
  data.pos = vertexPosition;

  return data;
}

#include "buildHABuffer.hglsl"

