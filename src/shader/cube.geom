#version 440

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec4 vPos[];
in vec4 vColor[];

out vec4 vertexPos;
out vec4 vertexColor;

void main()
{
  vec4 inPos = gl_in[0].gl_Position;
  gl_Position = inPos;
  vertexPos = gl_Position;
  vertexColor = vColor[0];
  EmitVertex();
  gl_Position = inPos + vec4(-0.4, -0.4, 0.0, 0.0);
  vertexPos = gl_Position;
  vertexColor = vColor[0];
  EmitVertex();
  gl_Position = inPos + vec4(0.4, -0.4, 0.0, 0.0);
  vertexPos = gl_Position;
  vertexColor = vColor[0];
  EmitVertex();

  EndPrimitive();
}
