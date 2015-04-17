#version 430

in vec3 position;

uniform mat4 modelViewProjectionMatrix;
uniform vec4 color;

out vec4 outColor;

void main()
{
  outColor = color;
  gl_Position = modelViewProjectionMatrix * vec4(position, 1.0f);
}
