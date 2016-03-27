#version 440

layout(location = 0) in vec2 pos;

uniform mat4 viewProjectionMatrix;
uniform mat4 viewMatrix;
uniform int bitIndex;

out vec4 vertexColor;

void main()
{
  gl_Position = viewProjectionMatrix * vec4(pos, 1.0f, 1.0f);

  int color = 1 << (7 - bitIndex);
  vertexColor = vec4(color / 255.0f);
}
