#version 440

in vec2 vertexTexCoord;

uniform sampler3D layers;
uniform sampler2D layer1;
uniform sampler2D layer2;
uniform sampler2D layer3;
uniform sampler2D layer4;
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
  vec2 texCoord = vec2(vertexTexCoord.x, 1.0 - vertexTexCoord.y);
  /*
  vec4 color = texture(layer1, texCoord);
  for (int layerIndex = 1; layerIndex < layerCount; ++layerIndex)
    color = blend(color, texture(layer))
    */

  vec4 color1 = texture(layers, vec3(texCoord, 0));
  vec4 color2 = texture(layers, vec3(texCoord, 0.33333333));
  vec4 color3 = texture(layers, vec3(texCoord, 0.6666666));
  vec4 color4 = texture(layers, vec3(texCoord, 1));

  outputColor = blend(blend(blend(blend(color1, color2), color3), color4),
                      backgroundColor);
  //outputColor = blend(color1, backgroundColor);
}
