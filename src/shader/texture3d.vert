#version 430

in vec3 position;

uniform mat4 modelViewProjectionMatrix;

out vec3 vertexTexCoord3d;
out vec4 vertexPosition;

void main()
{
  vertexPosition = modelViewProjectionMatrix * vec4(position, 1.0f);
  gl_Position = vertexPosition;
  vertexTexCoord3d = position + vec3(0.5f, 0.5f, 0.5f);
}
