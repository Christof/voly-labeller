#version 440

in vec2 vertexTexCoord;

uniform sampler2D textureSampler;
uniform sampler2D textureSampler2;

out vec4 outputColor;

// Blending equation for in-order traversal
vec4 blend(vec4 clr, vec4 srf)
{
  return clr + (1.0 - clr.w) * vec4(srf.xyz * srf.w, srf.w);
}

void main()
{
  vec4 color1 = texture(textureSampler, vec2(vertexTexCoord.x, 1.0 - vertexTexCoord.y));
  vec4 color2 = texture(textureSampler2, vec2(vertexTexCoord.x, 1.0 - vertexTexCoord.y));

  outputColor = blend(color1, color2);
}
