#version 440

in vec2 vertexTexCoord;

uniform sampler2D textureSampler;

out vec4 color;

void main()
{
  color =
      texture(textureSampler, vec2(vertexTexCoord.x, 1.0 - vertexTexCoord.y));
  color.a = 0.5f;
}
