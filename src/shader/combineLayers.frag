#version 440

in vec2 vertexTexCoord;

uniform sampler2D layer1;
uniform sampler2D layer2;
uniform sampler2D layer3;
uniform sampler2D layer4;
uniform vec4 backgroundColor = vec4(1, 1, 1, 1);

out vec4 outputColor;

// Blending equation for in-order traversal
vec4 blend(vec4 clr, vec4 srf)
{
  return clr + (1.0 - clr.a) * vec4(srf.rgb * srf.a, srf.a);
}

void main()
{
  vec4 color1 = texture(layer1, vec2(vertexTexCoord.x, 1.0 - vertexTexCoord.y));
  vec4 color2 = texture(layer2, vec2(vertexTexCoord.x, 1.0 - vertexTexCoord.y));
  vec4 color3 = texture(layer3, vec2(vertexTexCoord.x, 1.0 - vertexTexCoord.y));
  vec4 color4 = texture(layer4, vec2(vertexTexCoord.x, 1.0 - vertexTexCoord.y));

  outputColor = blend(blend(blend(blend(color1, color2), color3), color4),
                      backgroundColor);
}
