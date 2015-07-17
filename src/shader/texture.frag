#version 440

in vec2 vertexTexCoord;

uniform sampler2D textureSampler;

out vec4 color;

void main()
{
  color = texture(textureSampler, vertexTexCoord);
}

