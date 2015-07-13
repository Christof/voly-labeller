#version 430

in vec3 position;

uniform mat4 modelViewProjectionMatrix;
uniform vec4 color;
uniform float zOffset = 0.0f;

out vec4 vertexColor;
out vec4 vertexPosition;

void main()
{
  vertexColor = color;
  vertexPosition = modelViewProjectionMatrix * vec4(position, 1.0f);
  vertexPosition.z += zOffset;
  gl_Position = vertexPosition;
}
