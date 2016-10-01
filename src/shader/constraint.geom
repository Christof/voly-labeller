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

void emitVertex(in vec2 position)
{
  gl_Position = vec4(position, 0, 1);
  EmitVertex();
}

void drawConnectorConstraint(in vec2 anchor, in vec2 connectorStart,
                             in vec2 connectorEnd)
{
  const vec2 anchorToStart = normalize(connectorStart - anchor);
  const vec2 connectorStartShadow = connectorStart + 2.0f * anchorToStart;

  const vec2 anchorToEnd = normalize(connectorEnd - anchor);
  const vec2 connectorEndShadow = connectorEnd + 2.0f * anchorToEnd;

  emitVertex(connectorStart);
  emitVertex(connectorStartShadow);
  emitVertex(connectorEnd);
  EndPrimitive();

  emitVertex(connectorEnd);
  emitVertex(connectorStartShadow);
  emitVertex(connectorEndShadow);
  EndPrimitive();
}

void main()
{
  int drawId = vDrawId[0];

  vertexColor = vVertexColor[0];
  drawConnectorConstraint(gl_in[0].gl_Position.xy, vConnectorEnd[0].xy,
                          vConnectorStart[0].xy);
}
