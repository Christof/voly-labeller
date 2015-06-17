#version 430

in vec3 outTexCoord;
out vec4 color;

void main()
{
  color.rgb = outTexCoord;
  color.a = 1.0f;
}
