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

  vec4 color = texelFetch(layers, ivec3(texCoord, 0), 0);
  for (int layerIndex = 1; layerIndex < layerCount; ++layerIndex)
    color = blend(color, texelFetch(layers, ivec3(texCoord, layerIndex), 0));

  outputColor = blend(color, backgroundColor);
}
