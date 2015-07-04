#version 440

in vec3 vertexPos;

uniform mat4 u_Model;
uniform mat4 u_View;
uniform mat4 u_Projection;

out vec4 u_Pos;

void main()
{
  u_Pos       = u_Projection * u_View * u_Model * vec4(vertexPos, 1.0);
  gl_Position = u_Projection * u_View * u_Model * vec4(vertexPos, 1.0);
}
