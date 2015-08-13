#version 440

layout(location = 0) in vec3 pos;
layout(location = 2) in vec4 color;
layout(location = 4) in int drawId;

uniform mat4 modelViewProjectionMatrix;

out vec4 vPos;
out vec4 vColor;
out mat4 matrix;
out vec3 vTexCoord;

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
  matrix = modelViewProjectionMatrix * model;

  vPos = matrix * vec4(pos, 1.0f);
  gl_Position = vPos;
  vColor = color;
  // cube in the range of [-1, -1, -1] to [1, 1, 1]
  vTexCoord = 2.0f * pos;
}
