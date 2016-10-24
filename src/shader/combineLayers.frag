#version 440

in vec2 vertexTexCoord;

uniform sampler3D layers;
uniform int layerCount;
uniform vec4 backgroundColor = vec4(1, 1, 1, 1);

out vec4 outputColor;

// Blending equation for in-order traversal
vec4 blend(vec4 clr, vec4 srf)
{
  return clr + (1.0 - clr.a) * srf;
}

void main()
{
  vec2 texCoord = vec2(vertexTexCoord.x, 1.0 - vertexTexCoord.y) *
    textureSize(layers, 0).xy;
  /*
  vec4 color = texture(layer1, texCoord);
  for (int layerIndex = 1; layerIndex < layerCount; ++layerIndex)
    color = blend(color, texture(layer))
    */

  vec4 color1 = texelFetch(layers, ivec3(texCoord, 0), 0);
  vec4 color2 = texelFetch(layers, ivec3(texCoord, 1), 0);
  vec4 color3 = texelFetch(layers, ivec3(texCoord, 2), 0);
  vec4 color4 = texelFetch(layers, ivec3(texCoord, 3), 0);

  outputColor = blend(blend(blend(blend(color1, color2), color3), color4),
                      backgroundColor);
  //outputColor = blend(color4, backgroundColor);
}
