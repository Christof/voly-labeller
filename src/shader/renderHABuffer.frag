#version 440
#extension GL_NV_shader_buffer_load : enable
#extension GL_NV_gpu_shader5 : enable
#extension GL_NV_shader_atomic_int64 : enable

// --------------------------------------------

#include "HABufferImplementation.hglsl"

// --------------------------------------------

in vec4 u_Pos;

out vec4 o_PixColor;
layout(depth_any) out float gl_FragDepth;

// --------------------------------------------

// uniform float InnerDensity  = 1.0;
// uniform float InnerOpacity  = 1.0;
// uniform float InnerExponent = 1.5;
// uniform vec3  InnerColor    = vec3(0.5,0.5,0.5);
uniform vec3 BkgColor = vec3(1.0, 1.0, 1.0);
// uniform vec3  BkgColor      = vec3(0.0,0.0,0.0);

uniform float ZNear;
uniform float ZFar;

// --------------------------------------------

#define M 64  // Maximum number of fragments for post sort

// --------------------------------------------

// Blending equation for in-order traversal

vec4 blend(vec4 clr, vec4 srf)
{
  return clr + (1.0 - clr.w) * vec4(srf.xyz * srf.w, srf.w);
}

// --------------------------------------------

void main()
{

  o_PixColor = vec4(0);

  if (gl_SampleMaskIn[0] == 0)
  {
    // Ignore helper pixels
    // o_PixColor = vec4(1.0, 1.0, 0.3, 1.0);
  }
  else
  {

    vec2 pos = (u_Pos.xy * 0.5 + 0.5) * float(u_ScreenSz);

    if (pos.x >= u_ScreenSz || pos.y >= u_ScreenSz || pos.x < 0 || pos.y < 0)
    {
      o_PixColor = vec4(1.0, 1.0, 0.3, 1.0);
      return;
    }

    // vec2  pos      = ( u_Pos.xy*0.5  ) * float(u_ScreenSz);
    uvec2 ij = uvec2(pos.xy);
    uint32_t pix = (ij.x + ij.y * u_ScreenSz);

    gl_FragDepth = 0.0;

    vec4 clr = vec4(0, 0, 0, 0);

    uint maxage = u_Counts[Saddr(ij)];

    if (maxage == 0)
    {
      o_PixColor = vec4(0.0, 0.5, 0.8, 1.0);
       return;

      // discard;  // no fragment, early exit
    }
    for (uint a = 1; a <= maxage; a++)  // all fragments
    // for (uint a = 1 ; a <= 1 ; a++ ) // just first fragment
    {
      uvec2 l = (ij + u_Offsets[a]);
      uint64_t h = Haddr(l % uvec2(u_HashSz));

      uint64_t rec = u_Records[h];

      uint32_t key = uint32_t(rec >> uint64_t(32));
      if (HA_AGE(key) == a)
      {
        // clr = blend(clr,RGBA(uint32_t(rec)));
        const FragmentData fragment = u_FragmentData[uint32_t(rec)];
        clr = blend(clr, fragment.color);
      }
    }

    // background
    clr = blend(clr, vec4(BkgColor, 1.0));
    // done
    o_PixColor = clr;
  }
}

// --------------------------------------------
