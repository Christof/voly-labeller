#version 330

in vec3 vertexPosition;
in vec4 vertexColor;
in vec3 vertexNormal;
in vec2 vertexTextureCoordinate;

uniform mat4 modelViewProjectionMatrix;

out vec4 outColor;
out vec3 outNormal;
out vec4 outPosition;
out vec2 outTextureCoordinate;

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

  outColor = vertexColor;
  outPosition = modelViewProjectionMatrix * modelMatrix * vec4(vertexPosition, 1.0f);
  gl_Position = outPosition;
  outNormal = mul(modelMatrix, vertexNormal).xyz;
  outTextureCoordinate = vertexTextureCoordinate;
}
