#version 440

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 connectorStart;
layout(location = 2) in vec2 connectorEnd;

uniform mat4 viewProjectionMatrix;
uniform mat4 viewMatrix;
uniform float color;

out vec4 vConnectorStart;
out vec4 vConnectorEnd;
out vec4 vVertexColor;

void main()
{
  gl_Position = viewProjectionMatrix * vec4(pos, 1.0f, 1.0f);

  vConnectorStart = viewProjectionMatrix * vec4(connectorStart, 1.0f, 1.0f);
  vConnectorEnd = viewProjectionMatrix * vec4(connectorEnd, 1.0f, 1.0f);
  vVertexColor = vec4(color);
}
