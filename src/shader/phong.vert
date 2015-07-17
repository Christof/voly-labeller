#version 330

in vec3 vertexPosition;
in vec4 vertexColor;
in vec3 vertexNormal;

uniform mat4 modelViewProjectionMatrix;
uniform mat4 modelMatrix;

out vec4 outColor;
out vec3 outNormal;
out vec4 outPosition;

vec4 mul(mat4 matrix, vec3 vector)
{
  return matrix * vec4(vector, 1.0f);
}
void main()
{
  outColor = vertexColor;
  outPosition = modelViewProjectionMatrix * vec4(vertexPosition, 1.0f);
  gl_Position = outPosition;
  outNormal = mul(modelMatrix, vertexNormal).xyz;
}
