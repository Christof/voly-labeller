#version 440

#include "HABufferImplementation.hglsl"
#include "bindlessTexture.hglsl"

in vec4 vertexPos;
in vec4 vertexEyePos;
in vec3 vertexNormal;
in vec4 vertexColor;
in vec2 vertexTexCoord;
in flat int vertexDrawId;

layout(std430, binding = 1) buffer CB1
{
  Tex2DAddress texAddress[];
};

FragmentData computeData()
{
  FragmentData data;
  data.color = Texture(texAddress[vertexDrawId], vertexTexCoord.xy);
  //data.color = vec4(0, vertexTexCoord.x, vertexTexCoord.y, 0.5);
  data.eyePos = vertexEyePos;
  data.objectId = 0;

  return data;
}

#include "buildHABuffer.hglsl"

