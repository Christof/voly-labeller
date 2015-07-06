#version 440
#extension GL_NV_gpu_shader5 : enable

coherent uniform
    uint64_t *u_Records;  // all fragment records (<depth|pointer> pairs)
coherent uniform uint32_t *u_Counts;  // auxiliary counters

uniform uint u_NumRecords;
uniform uint u_ScreenSz;
uniform uint u_HashSz;

in vec4 u_Pos;
out vec4 o_PixColor;

void main()
{
  o_PixColor = vec4(0);

  // Ignore helper pixels
  if (gl_SampleMaskIn[0] == 0)
    discard;

  uvec2 ij = uvec2(gl_FragCoord.xy);

  // Hashed-lists
  // clear all records
  for (uint o = ij.x + ij.y * u_ScreenSz; o < u_NumRecords;
       o += u_ScreenSz * u_ScreenSz)
  {
    u_Records[o] = uint64_t(0);
  }

  // clear max age table (max age = 0)
  u_Counts[ij.x + ij.y * u_ScreenSz] = 0u;

  if (ij.x == 0 && ij.y == 0)
  {
    u_Counts[u_ScreenSz * u_ScreenSz] = uint32_t(0);
  }

  o_PixColor = vec4(0, 0, 1, 1);
}

