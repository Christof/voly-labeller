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
  mat4 modelMatrix = getModelMatrix(drawId);

  vec3 cameraRight = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
  vec3 cameraUp = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);
  vec3 labelPosition = vec3(modelMatrix[3][0], modelMatrix[3][1], modelMatrix[3][2]);
  vec2 size = vec2(modelMatrix[0][0], modelMatrix[1][1]);
  vec3 pos = labelPosition +
      cameraRight * pos.x * size.x +
      cameraUp * pos.y * size.y;

  vertexPos = viewProjectionMatrix * vec4(pos, 1.0f);
  vertexEyePos = viewMatrix * vec4(pos, 1.0f);
  vertexPos.z -= 0.01f;
  gl_Position = vertexPos;
  vertexNormal = (modelMatrix * vec4(normal, 1.0f)).xyz;
  vertexColor = color;
  vertexTexCoord = texCoord;
  vertexDrawId = drawId;
}
