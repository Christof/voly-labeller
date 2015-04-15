#version 430

in vec3 position;
in vec2 texcoord;

uniform mat4 modelViewProjectionMatrix;
uniform mat4 modelMatrix;

out vec2 outTexcoord;

void main()
{
  outTexcoord = texcoord;
  gl_Position = modelViewProjectionMatrix * vec4(position, 1.0f);
}
