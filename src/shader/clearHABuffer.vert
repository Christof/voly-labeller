#version 440

in vec4 vertexPosition;

out vec4 u_Pos;

void main()
{
  u_Pos = vertexPosition;
  gl_Position = vertexPosition;
}
