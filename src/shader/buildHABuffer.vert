#version 440

layout(location = 0) in vec3 vertexPos;
layout(location = 1) in vec3 vertexNormal;
layout(location = 2) in vec4 vertexColor;
layout(location = 3) in vec2 vertexTexCoord;
layout(location = 4) in int drawID;

uniform mat4 u_View;
uniform mat4 u_Projection;

out vec4 v_Pos;
out vec3 v_View;
out vec3 v_Normal;
out vec4 v_Color;
out vec3 v_Vertex;
out vec2 v_Tex;
out flat int v_drawID;

layout(std140, binding = 0) buffer CB0
{
  mat4 Transforms[];
};

void main()
{
  mat4 model = Transforms[drawID];

  v_Vertex    = (model * vec4(vertexPos, 1.0)).xyz;
  v_View      = (u_View * model * vec4(vertexPos, 1.0)).xyz;
  v_Normal    = (u_View * model * vec4(vertexNormal,0.0)).xyz;
  v_Tex       = vertexTexCoord;
  v_Color     = vertexColor;
  v_Pos       = u_Projection * u_View * model * vec4(vertexPos, 1.0);
  gl_Position = u_Projection * u_View * model * vec4(vertexPos, 1.0);
  v_drawID    = drawID;
}
