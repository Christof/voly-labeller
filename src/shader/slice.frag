#version 430 core

in vec2 outTexcoord;

out vec4 color;

uniform sampler3D textureSampler;

void main()
{
  vec3 texCoord = vec3(outTexcoord.x, outTexcoord.y, 0);
  float value = texture(textureSampler, texCoord).r;
  color = vec4(value, value, value, 1.0f);
}
