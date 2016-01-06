#version 440
#extension GL_NV_gpu_shader5 : enable

coherent uniform
    uint64_t *records;  // all fragment records (<depth|pointer> pairs)
coherent uniform uint32_t *counters;  // auxiliary counters

uniform uint tableElementCount;
uniform uint screenSize;

void main()
{
  // Ignore helper pixels
  if (gl_SampleMaskIn[0] == 0)
    discard;

  uvec2 ij = uvec2(gl_FragCoord.xy);

  // Hashed-lists
  // clear all records
  for (uint o = ij.x + ij.y * screenSize; o < tableElementCount;
       o += screenSize * screenSize)
  {
    records[o] = uint64_t(0);
  }

  // clear max age table (max age = 0)
  counters[ij.x + ij.y * screenSize] = 0u;

  if (ij.x == 0 && ij.y == 0)
  {
    counters[screenSize * screenSize] = uint32_t(0);
  }

  discard;
}

