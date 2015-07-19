#version 440

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 color;
layout(location = 3) in vec2 texCoord;
layout(location = 4) in int drawId;

uniform mat4 modelViewProjectionMatrix;
// uniform mat4 modelMatrix;

out vec4 vertexPos;
out vec3 vertexNormal;
out vec4 vertexColor;
out vec2 vertexTexCoord;
out int vertexDrawId;

vec4 mul(mat4 matrix, vec3 vector)
{
  return matrix * vec4(vector, 1.0f);
}

void main()
{
  vertexPos = modelViewProjectionMatrix * vec4(pos, 1.0f);
  gl_Position = vertexPos;
  vertexNormal = normal; //mul(modelMatrix, normal).xyz;
  vertexColor = color;
  vertexTexCoord = texCoord;
  vertexDrawId = drawId;
}
