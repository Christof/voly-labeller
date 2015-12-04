#version 440

layout(location = 0) in vec2 pos;

uniform mat4 viewProjectionMatrix;
uniform mat4 viewMatrix;

out vec4 vertexColor;

void main()
{
  gl_Position = viewProjectionMatrix * vec4(pos, 1.0f, 1.0f);
  vertexColor = vec4(1, 1, 1, 1);
}
