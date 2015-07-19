#version 440

#include "HABufferImplementation.hglsl"

in vec4 vertexPos;
in vec3 vertexNormal;
in vec4 vertexColor;
in vec2 vertexTexCoord;
in int vertexDrawId;

// out vec4 color;

FragmentData computeData()
{
  FragmentData data;
  data.color = vertexColor;
  //data.color = vec4(1, 0, 0, 1);
  data.pos = vertexPos;

  return data;
}

/*
void main()
{
  color = vec4(1, 0, 0, 1);
}
*/

#include "buildHABuffer.hglsl"

