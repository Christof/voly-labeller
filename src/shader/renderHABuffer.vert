#version 440

in vec3 vertexPosition;

out vec4 u_Pos;

void main()
{
  u_Pos = vec4(vertexPosition, 1.0);
  gl_Position = u_Pos;
}
