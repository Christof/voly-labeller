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

struct PhongMaterial
{
  vec4 ambientColor;
  vec4 diffuseColor;
  vec4 specularColor;
  mat4 normalMatrix;
  float shininess;
};

layout(std140, binding = 1) buffer CB1
{
  PhongMaterial phongMaterial[];
  //mat4 normalMatrix[];
};

void main()
{
  mat4 modelMatrix = getModelMatrix(drawId);

  outColor = color;
  outPosition = viewProjectionMatrix * modelMatrix * vec4(pos, 1.0f);
  outEyePosition = viewMatrix * modelMatrix * vec4(pos, 1.0f);
  gl_Position = outPosition;
  outDrawId = drawId;
  outNormal = mul(phongMaterial[drawId].normalMatrix, normal).xyz;
      //normalize(mul(transpose(inverse(viewMatrix * modelMatrix)), normal).xyz);
  outTextureCoordinate = texCoord;
}
