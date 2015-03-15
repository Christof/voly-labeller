#version 330

in vec3 vertexPosition;
in vec4 vertexColor;

uniform mat4 viewProjectionMatrix;

out vec4 outColor;

void main()
{
  outColor = vertexColor;
  gl_Position = viewProjectionMatrix * vec4(vertexPosition, 1.0);
}
