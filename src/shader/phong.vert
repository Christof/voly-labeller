#version 440

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 color;
layout(location = 3) in vec2 texCoord;
layout(location = 4) in int drawId;

uniform mat4 viewProjectionMatrix;
uniform mat4 viewMatrix;

out vec4 outColor;
out vec3 outNormal;
out vec4 outPosition;
out vec4 outEyePosition;
out vec2 outTextureCoordinate;
out int outDrawId;

#include "vertexHelper.hglsl"

layout(std140, binding = 2) buffer CB2
{
  mat4 normalMatrices[];
};

void main()
{
  mat4 modelMatrix = getModelMatrix(drawId);

  outColor = color;
  outPosition = viewProjectionMatrix * modelMatrix * vec4(pos, 1.0f);
  outEyePosition = viewMatrix * modelMatrix * vec4(pos, 1.0f);
  gl_Position = outPosition;
  outDrawId = drawId;
  outNormal = normalize((normalMatrices[drawId] * vec4(normal, 0)).xyz);
  outTextureCoordinate = texCoord;
}
