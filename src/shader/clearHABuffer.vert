#version 440

in vec4 vertexPosition;

out vec4 fragmentInputPosition;

void main()
{
  fragmentInputPosition = vertexPosition;
  gl_Position = vertexPosition;
}
