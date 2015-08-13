#version 440

layout(location = 0) in vec3 pos;
layout(location = 2) in vec4 color;
layout(location = 4) in int drawId;

uniform mat4 modelViewProjectionMatrix;

out vec4 vPos;
out vec4 vColor;
out vec3 vTexCoord;
out int vDrawId;

vec4 mul(mat4 matrix, vec3 vector)
{
  return matrix * vec4(vector, 1.0f);
}

void main()
{
  vPos = vec4(pos, 1.0f);
  gl_Position = vPos;
  vColor = color;
  // cube in the range of [-1, -1, -1] to [1, 1, 1]
  vTexCoord = 2.0f * pos;
  vDrawId = drawId;
}
