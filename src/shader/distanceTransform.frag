#version 440

in vec2 vertexTexCoord;

uniform sampler2D textureSampler;

out vec4 color;

void main()
{
  float dist =
      texture(textureSampler, vec2(vertexTexCoord.x, 1.0 - vertexTexCoord.y)).r;
  dist *= 64.0f;
  color.rgb = vec3(dist);
  color.a = 1.0f;
}
