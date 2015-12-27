#version 440
#extension GL_NV_gpu_shader5 : enable

coherent uniform
    uint64_t *records;  // all fragment records (<depth|pointer> pairs)
coherent uniform uint32_t *counters;  // auxiliary counters

uniform uint tableElementCount;
uniform uint u_ScreenSz;

void main()
{
  // Ignore helper pixels
  if (gl_SampleMaskIn[0] == 0)
    discard;

  uvec2 ij = uvec2(gl_FragCoord.xy);

  // Hashed-lists
  // clear all records
  for (uint o = ij.x + ij.y * u_ScreenSz; o < tableElementCount;
       o += u_ScreenSz * u_ScreenSz)
  {
    records[o] = uint64_t(0);
  }

  // clear max age table (max age = 0)
  counters[ij.x + ij.y * u_ScreenSz] = 0u;

  if (ij.x == 0 && ij.y == 0)
  {
    counters[u_ScreenSz * u_ScreenSz] = uint32_t(0);
  }

  discard;
}

