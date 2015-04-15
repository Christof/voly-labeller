#version 430 core

in vec2 outTexcoord;

out vec4 color;

uniform sampler2D textureSampler;

void main()
{
  color = texture(textureSampler, outTexcoord);
}
