#version 440

in vec2 vertexTexCoord;

uniform sampler2D textureSampler;
uniform vec4 backgroundColor = vec4(1, 1, 1, 1);

out vec4 color;

vec4 blend(vec4 clr, vec4 srf)
{
  return clr + (1.0 - clr.a) * vec4(srf.rgb * srf.a, srf.a);
}

void main()
{
  vec4 texel = texture(textureSampler, vec2(vertexTexCoord.x, 1.0 - vertexTexCoord.y));
  color = blend(texel, backgroundColor);
}
