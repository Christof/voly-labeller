#version 440

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 color;
layout(location = 3) in vec2 texCoord;
layout(location = 4) in int drawId;

uniform mat4 viewProjectionMatrix;
uniform mat4 viewMatrix;
uniform vec2 maxTextureCoord;

out vec2 vertexTexCoord;
out int vertexDrawId;

#include "vertexHelper.hglsl"

void main()
{
  mat4 modelMatrix = getModelMatrix(drawId);

  const vec3 cameraRight = vec3(1, 0, 0);
  const vec3 cameraUp = vec3(0, 1, 0);
  const vec3 labelPositionNDC = vec3(modelMatrix[3][0], modelMatrix[3][1], modelMatrix[3][2]);
  vec2 sizeNDC = vec2(modelMatrix[0][0], modelMatrix[1][1]);
  vec3 position = labelPositionNDC +
      cameraRight * pos.x * sizeNDC.x +
      cameraUp * pos.y * sizeNDC.y;

  vec4 vertexPos = vec4(position, 1);
  gl_Position = vertexPos;
  vertexTexCoord = texCoord * maxTextureCoord.xy;
  vertexDrawId = drawId;
}
