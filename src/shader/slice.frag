#version 440

#include "HABufferImplementation.hglsl"

in vec4 vertexPos;
in vec4 vertexEyePos;
in vec2 vertexTexCoord;

uniform sampler3D textureSampler;

FragmentData computeData()
{
  vec3 texCoord = vec3(vertexTexCoord.x, vertexTexCoord.y, 0);
  float value = texture(textureSampler, texCoord).r;

  FragmentData data;
  data.color = vec4(value, value, value, 1.0f);
  data.eyePos = vertexEyePos;

  return data;
}

#include "buildHABuffer.hglsl"

