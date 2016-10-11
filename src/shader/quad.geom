/**
 * \brief Geometry shader which constructs a screen aligned quad
 *
 * The quad is given by the `gl_Position` input and the `halfSize` uniform,
 * which contains half the width and hieght.
 */

#version 440

layout(points) in;
layout(triangle_strip, max_vertices = 6) out;

in int vDrawId[];

out vec4 vertexColor;

uniform float color;
uniform vec2 halfSize = vec2(0.2, 0.05);

void emitVertex(in vec2 position)
{
  gl_Position = vec4(position, 0, 1);
  EmitVertex();
}

void main()
{
  int drawId = vDrawId[0];

  vertexColor = vec4(color);
  vec2 center = gl_in[0].gl_Position.xy;

  vec2 rightBottom = center + vec2(halfSize.x, -halfSize.y);
  emitVertex(rightBottom);

  vec2 rightTop = center + halfSize;
  emitVertex(rightTop);

  vec2 leftBottom = center - halfSize;
  emitVertex(leftBottom);
  EndPrimitive();

  emitVertex(leftBottom);
  emitVertex(rightTop);
  vec2 leftTop = center + vec2(-halfSize.x, halfSize.y);
  emitVertex(leftTop);
  EndPrimitive();
}
