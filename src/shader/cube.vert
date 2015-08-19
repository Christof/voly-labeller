#version 440

layout(location = 0) in vec3 pos;
layout(location = 4) in int drawId;

out int vDrawId;

void main()
{
  gl_Position = vec4(pos, 1.0f);
  vDrawId = drawId;
}
