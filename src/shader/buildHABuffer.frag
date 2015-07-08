#version 440
#extension GL_NV_shader_buffer_load : enable
#extension GL_NV_gpu_shader5 : enable
#extension GL_NV_shader_atomic_int64       : enable

// --------------------------------------------

#define HABuffer 1

#include "HABufferImplementation.hglsl"

// --------------------------------------------

struct Tex2DAddress
{
  uint64_t Container;
  float Page;
  int dummy;
  vec2 texScale;
};

layout(std430, binding = 1) buffer CB1
{
  Tex2DAddress texAddress[];
};

in vec4 v_Pos;
in vec3 v_View;
in vec3 v_Normal;
in vec4 v_Color;
in vec3 v_Vertex;
in vec2 v_Tex;
in flat int v_drawID;

out vec4 o_PixColor;

uniform float u_ZNear;
uniform float u_ZFar;

vec3 shadeStrips(vec3 texcoord)
{
  vec3 col;
  float i = floor(texcoord.x * 6.0f);

  col.rgb = fract(i * 0.5f) == 0.0f ? vec3(0.4f, 0.85f, 0.0f) : vec3(1.0f);
  col.rgb *= texcoord.z;

  return col;
}

// uniform uint u_FlipOrient;

// --------------------------------------------

// uniform vec3 Color;
uniform float Opacity = 0.6;
uniform vec3 LightPos = vec3(0, 2, 0);
// uniform sampler2D Tex;

vec4 Texture(Tex2DAddress addr, vec2 uv)
{
  vec3 texc = vec3(uv.x * addr.texScale.x, uv.y * addr.texScale.y, addr.Page);

  return texture(sampler2DArray(addr.Container), texc);
}

FragmentData computeData()
{
  vec3 nrm = normalize(v_Normal);
  vec3 view = normalize(v_View);
  vec3 light = normalize(LightPos - v_View);
  int drawID = int(v_drawID);
  vec4 clr;
  /*
  if (texAddress[drawID].Container > 0)
  {
    clr = Texture(texAddress[drawID], v_Tex.xy);
  }
  else
  {
    clr = (max(0.2, dot(nrm, light))) * vec4(v_Tex.xy, 0, Opacity);
  }
  */
  clr = vec4(1, 0, 1, 1);
  FragmentData fd;
  fd.color = vec4(clr.xyz, Opacity);
  fd.pos = v_Pos;

  return fd;
}

// --------------------------------------------

void main()
{
  o_PixColor = vec4(0);
  if (gl_SampleMaskIn[0] == 0)
    discard;

  // Detect main buffer overflow
  uint32_t count = atomicAdd(u_Counts + u_ScreenSz * u_ScreenSz, 1);
  if (count > u_NumRecords)
  {
    u_Counts[u_ScreenSz * u_ScreenSz] = u_NumRecords;
    discard;
  }

  // Compute fragment data

  FragmentData fragment = computeData();
  u_FragmentData[count] = fragment;

  vec2 prj = fragment.pos.xy / fragment.pos.w;
  vec3 pos =
      (vec3(prj * 0.5 + 0.5, 1.0 - (fragment.pos.z + u_ZNear) / (u_ZFar + u_ZNear)));
  uint32_t depth = uint32_t(pos.z * MAX_DEPTH);
  uvec2 pix = uvec2(pos.xy * u_ScreenSz);

  bool success = insert_preopen(depth, pix, count);

  o_PixColor = success ? fragment.color : vec4(1, 0, 0, 0);
}

// --------------------------------------------
