#version 440

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 color;
layout(location = 3) in vec2 texCoord;
layout(location = 4) in int drawId;

uniform mat4 modelViewProjectionMatrix;
uniform mat4 viewMatrix;

out vec4 outColor;
out vec3 outNormal;
out vec4 outPosition;
out vec2 outTextureCoordinate;
out int outDrawId;
out vec3 cameraDirection;

layout(std140, binding = 0) buffer CB0
{
  mat4 Transforms[];
};

vec4 mul(mat4 matrix, vec3 vector)
{
  return matrix * vec4(vector, 1.0f);
}

void main()
{
  mat4 modelMatrix = Transforms[drawId];

  outColor = color;
  outPosition = modelViewProjectionMatrix * modelMatrix * vec4(pos, 1.0f);
  gl_Position = outPosition;
  outNormal = mul(modelMatrix, normal).xyz;
  outTextureCoordinate = texCoord;
  outDrawId = drawId;
  cameraDirection = vec3(viewMatrix[2][0], viewMatrix[2][1], viewMatrix[2][2]);
}
