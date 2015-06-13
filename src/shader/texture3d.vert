#version 430

in vec3 position;

uniform mat4 modelViewProjectionMatrix;

out vec3 outTexCoord;

void main()
{
  gl_Position = modelViewProjectionMatrix * vec4(position, 1.0f);
  outTexCoord = position + vec3(0.5f, 0.5f, 0.5f);
}
