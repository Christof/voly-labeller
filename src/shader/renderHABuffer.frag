#version 440
#include "HABufferImplementation.hglsl"

in vec4 u_Pos;

out vec4 o_PixColor;
layout(depth_any) out float gl_FragDepth;

uniform vec3 BkgColor = vec3(1.0, 1.0, 1.0);
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;

struct Tex2DAddress
{
  uint64_t Container;
  float Page;
  int dummy;
  vec2 texScale;
};

struct VolumeData
{
  Tex2DAddress textureAddress;
  mat4 textureMatrix;
  int volumeId;
  mat4 objectToDatasetMatrix;
  int transferFunctionRow;
  // int transferFunctionRowCount;
};

layout(std430, binding = 1) buffer CB1
{
  VolumeData volumes[];
};

// Blending equation for in-order traversal
vec4 blend(vec4 clr, vec4 srf)
{
  return clr + (1.0 - clr.w) * vec4(srf.xyz * srf.w, srf.w);
}

bool fetchFragment(in uvec2 ij, in uint age, out FragmentData fragment)
{
  uvec2 l = (ij + u_Offsets[age]);
  uint64_t h = Haddr(l % uvec2(u_HashSz));

  uint64_t rec = u_Records[h];

  uint32_t key = uint32_t(rec >> uint64_t(32));

  if (HA_AGE(key) == age)
  {
    // clr = blend(clr,RGBA(uint32_t(rec)));
    fragment = u_FragmentData[uint32_t(rec)];
    return true;
  }
  else
  {
    return false;
  }
}


void main()
{
  o_PixColor = vec4(0);

  if (gl_SampleMaskIn[0] == 0)
    discard;

  vec2 pos = (u_Pos.xy * 0.5 + 0.5) * float(u_ScreenSz);

  if (pos.x >= u_ScreenSz || pos.y >= u_ScreenSz || pos.x < 0 || pos.y < 0)
  {
    o_PixColor = vec4(1.0, 1.0, 0.3, 1.0);
    return;
  }

  uvec2 ij = uvec2(pos.xy);
  uint32_t pix = (ij.x + ij.y * u_ScreenSz);

  gl_FragDepth = 0.0;

  uint maxage = u_Counts[Saddr(ij)];

  if (maxage == 0)
  {
    // o_PixColor = vec4(0.0, 0.5, 0.8, 1.0);
    // return;

    discard;  // no fragment, early exit
  }


  int activeobjects = 0;
  int activeobjectcount = 0;
  FragmentData current_fragment;
  FragmentData next_fragment;
  bool current_fragment_read_status = false;
  bool next_fragment_read_status = false;

  vec3 startpos_eye;
  vec3 endpos_eye;
  vec3 dirvec_eye;
  vec4 pos_proj;
  int objectId = -1;

  vec4 clr = vec4(0, 0, 0, 0);
  for (uint a = 1; a < maxage; a++)  // all fragments
  //for (uint a = 1; a <= maxage; a++)  // all fragments
  {

    if (a == 1)
    {
      current_fragment_read_status = fetchFragment(ij, a, current_fragment);
      startpos_eye = current_fragment.eyePos.xyz;
    }
    else
    {
      current_fragment_read_status = next_fragment_read_status;
      current_fragment = next_fragment;
      startpos_eye = endpos_eye;
    }

    // update  active objects

    if (current_fragment.objectId > 0)
    {
      objectId = current_fragment.objectId;
      activeobjects |= 1 << (objectId);

    }
    else if (current_fragment.objectId < 0)
    {
      objectId = -current_fragment.objectId;
      activeobjects &= (~(1 << objectId));
    }

    activeobjectcount = bitCount(activeobjects);


    // fetch next Fragment


    next_fragment_read_status = fetchFragment(ij, a+1, next_fragment);
    if (next_fragment_read_status)
    {
      endpos_eye = next_fragment.eyePos.xyz;
    }
    else
    {
      endpos_eye = startpos_eye;
    }

    // set up segment direction vector

    dirvec_eye = endpos_eye - startpos_eye;

    pos_proj = projectionMatrix*vec4(startpos_eye, 1.0f);
    pos_proj.z /= pos_proj.w;
    // FIXME: posproj.xy ???
    pos_proj += 1.0f;
    pos_proj /= 2.0f;

    if (activeobjectcount > 0) // in frag
    {

      uint ao = activeobjects;
      int aoc = activeobjectcount;

      for (int oi =0; oi < activeobjectcount; oi++)
      {
        int objectID = findLSB(ao);
        ao &= (~(1<<objectID));

        ///FIXME: continue porting from VolyRenderer
       /// vec4 textureStartPos =

        current_fragment.color = volumes[0].textureMatrix * inverse(viewMatrix) * current_fragment.eyePos;
      }

    }
    else
    {
      //current_fragment.color = vec4(0.0, 1.0, 1.0, 0.5);
    }

    if (current_fragment_read_status)
    {
      clr = blend(clr, current_fragment.color);
    }


    // break; // just the first fragment
  }

  if (maxage == 1)
  {
    next_fragment_read_status = fetchFragment(ij, 1, next_fragment);
  }

  if (next_fragment_read_status)
  {
    clr = blend(clr, next_fragment.color);
  }

  o_PixColor = blend(clr, vec4(BkgColor, 1.0));
}

