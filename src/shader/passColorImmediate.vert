#version 440

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 color;
layout(location = 3) in vec2 texCoord;
layout(location = 4) in int drawId;

uniform mat4 viewProjectionMatrix;
uniform mat4 viewMatrix;

out vec4 vertexColor;

#include "vertexHelper.hglsl"

void main()
{
  mat4 model = getModelMatrix(drawId);

  gl_Position = mul(viewProjectionMatrix * model, pos);
  vertexColor = color;
}
