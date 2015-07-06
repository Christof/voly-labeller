#version 440

in vec4 vertexPosition;  // LibSL takes care of vertex attributes 'mvf_*'
                         // (normal,color0,texcoord0,etc.)

uniform mat4 u_Projection;

out vec4 u_Pos;

void main()
{
  u_Pos = vertexPosition;
  gl_Position = u_Projection * vertexPosition;
  // gl_Position = vertexPosition;
}
