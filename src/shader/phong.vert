#version 330

in vec3 vertexPosition;
in vec4 vertexColor;
in vec3 vertexNormal;

uniform mat4 viewProjectionMatrix;

out vec4 outColor;
out vec3 outNormal;
out vec3 outPosition;

void main()
{
  outColor = vertexColor;
  outPosition = vertexPosition;
  gl_Position = viewProjectionMatrix * vec4(vertexPosition, 1.0f);
  outNormal = vertexNormal;
}
