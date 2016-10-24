#version 440

in vec2 vertexTexCoord;

uniform sampler3D textureSampler;
uniform vec4 backgroundColor = vec4(1, 1, 1, 1);
uniform int slice = 0;

out vec4 color;

vec4 blend(vec4 clr, vec4 srf)
{
  return clr + (1.0 - clr.a) * vec4(srf.rgb * srf.a, srf.a);
}

void main()
{
  vec2 texCoord = vec2(vertexTexCoord.x, 1.0 - vertexTexCoord.y) *
    textureSize(textureSampler, 0).xy;
  vec4 texel = texelFetch(textureSampler, ivec3(texCoord, slice), 0);

  color = blend(texel, backgroundColor);
}
