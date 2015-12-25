#version 440

in vec3 vertexPosition;

out vec4 fragmentInputPosition;

void main()
{
  fragmentInputPosition = vec4(vertexPosition, 1.0);
  gl_Position = fragmentInputPosition;
}
