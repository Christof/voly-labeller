#version 440

#include "HABufferImplementation.hglsl"

in vec4 vertexPos;
in vec3 vertexNormal;
in vec4 vertexColor;
in vec2 vertexTexCoord;
in flat int vertexDrawId;

struct Tex2DAddress
{
  uint64_t Container;
  float Page;
  int dummy;
  vec2 texScale;
};

layout(std430, binding = 1) buffer CB1
{
  Tex2DAddress texAddress[];
};

vec4 Texture(Tex2DAddress addr, vec2 uv)
{
  vec3 texc = vec3(uv.x * addr.texScale.x, uv.y * addr.texScale.y, addr.Page);

  return texture(sampler2DArray(addr.Container), texc);
}

FragmentData computeData()
{
  FragmentData data;
  data.color = Texture(texAddress[vertexDrawId], vertexTexCoord.xy);
  //data.color = vec4(0, vertexTexCoord.x, vertexTexCoord.y, 0.5);
  data.pos = vertexPos;

  return data;
}

#include "buildHABuffer.hglsl"

