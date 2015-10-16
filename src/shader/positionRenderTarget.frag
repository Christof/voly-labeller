#version 440

in vec2 vertexTexCoord;

uniform sampler2D textureSampler;

out vec4 color;

void main()
{
  vec4 position =
      texture(textureSampler, vec2(vertexTexCoord.x, 1.0 - vertexTexCoord.y));
  color.rgb = position.rgb * 0.5f + vec3(0.5f, 0.5f, 0.5f);
  color.a = 1.0f;
}
