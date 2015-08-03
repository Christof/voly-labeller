#version 440

#include "HABufferImplementation.hglsl"

in vec4 vertexPos;
in vec3 vertexNormal;
in vec4 vertexColor;
in vec2 vertexTexCoord;
in flat int vertexDrawId;

FragmentData computeData()
{
  FragmentData data;
  //data.color = vec4(0, vertexTexCoord.x, vertexTexCoord.y, 0.5);
  data.color = vertexColor;
  // data.color = vec4(1, 0, 0, 1);
  data.pos = vertexPos;

  return data;
}

#include "buildHABuffer.hglsl"

