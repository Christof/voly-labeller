#version 440

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 color;
layout(location = 3) in vec2 texCoord;
layout(location = 4) in int drawId;

uniform mat4 viewProjectionMatrix;
uniform mat4 viewMatrix;

out vec4 vertexPos;
out vec4 vertexEyePos;
out vec3 vertexNormal;
out vec4 vertexColor;
out vec2 vertexTexCoord;
out int vertexDrawId;

#include "vertexHelper.hglsl"

void main()
{
  mat4 model = getModelMatrix(drawId);

  vertexPos = viewProjectionMatrix * model * vec4(pos, 1.0f);
  vertexEyePos = viewMatrix * model * vec4(pos, 1.0f);
  gl_Position = vertexPos;
  vertexNormal = mul(model, normal).xyz;
  int id = getObjectId(drawId);
  vertexColor = color;
  vertexTexCoord = texCoord;
  vertexDrawId = drawId;
}
