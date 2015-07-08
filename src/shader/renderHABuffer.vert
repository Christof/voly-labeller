#version 440

in vec3 vertexPosition;

uniform mat4 u_Model;
uniform mat4 u_View;
uniform mat4 u_Projection;

out vec4 u_Pos;

void main()
{
  u_Pos = u_Projection * u_View * u_Model * vec4(vertexPosition, 1.0);
  gl_Position = u_Projection * u_View * u_Model * vec4(vertexPosition, 1.0);
}
