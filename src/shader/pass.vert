#version 440

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 color;
layout(location = 3) in vec2 texCoord;
layout(location = 4) in int drawId;

uniform mat4 modelViewProjectionMatrix;

out vec4 vertexPos;
out vec3 vertexNormal;
out vec4 vertexColor;
out vec2 vertexTexCoord;
out int vertexDrawId;

vec4 mul(mat4 matrix, vec3 vector)
{
  return matrix * vec4(vector, 1.0f);
}

layout (std140, binding = 0) buffer CB0
{
    mat4 Transforms[];
};

void main()
{
  mat4 model = Transforms[drawId];

  vertexPos = modelViewProjectionMatrix * model * vec4(pos, 1.0f);
  gl_Position = vertexPos;
  vertexNormal = mul(model, normal).xyz;
  vertexColor = color;
  vertexTexCoord = texCoord;
  vertexDrawId = drawId;
}
