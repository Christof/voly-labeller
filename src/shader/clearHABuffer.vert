#version 440

in vec4 vertexPos;      // LibSL takes care of vertex attributes 'mvf_*' (normal,color0,texcoord0,etc.)

uniform mat4 u_Projection;

out vec4 u_Pos;

void main()
{
  u_Pos       = vertexPos;
  gl_Position = u_Projection * vertexPos;
  //gl_Position = vertexPos;
}
