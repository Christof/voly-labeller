/**
 * \brief TODO
 */

#version 440

layout(points) in;
layout(triangle_strip, max_vertices = 113) out;

in int vDrawId[];
in vec4 vConnectorStart[];
in vec4 vConnectorEnd[];
in vec4 vVertexColor[];

out vec4 vertexColor;
out int volumeId;

uniform mat4 viewMatrix;
uniform mat4 viewProjectionMatrix;

void main()
{
  int drawId = vDrawId[0];

  vertexColor = vVertexColor[0];
  gl_Position = gl_in[0].gl_Position;
  EmitVertex();

  gl_Position = vConnectorEnd[0];
  EmitVertex();

  gl_Position = vConnectorStart[0];
  EmitVertex();
  EndPrimitive();
}
