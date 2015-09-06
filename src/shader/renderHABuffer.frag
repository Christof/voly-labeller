#version 440
#include "HABufferImplementation.hglsl"
#include "bindlessTexture.hglsl"
#define MAX_SAMPLES 2048
#define STEP_FACTOR 1.5

in vec4 u_Pos;

out vec4 o_PixColor;
layout(depth_any) out float gl_FragDepth;

uniform vec3 BkgColor = vec3(1.0, 1.0, 1.0);
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 inverseViewMatrix;
uniform vec3 textureAtlasSize = vec3(512, 512, 186);
uniform vec3 sampleDistance = vec3(0.49f / 512.0, 0.49f / 512.0, 0.49f / 186.0);

uniform sampler3D volumeSampler;

#define transferFunctionRowCount 64.0f

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

void getVolumeSample(in int objectID, in vec3 texturePos, out float density,
                     out vec3 gradient)
{

  const vec3 gscf = vec3(1.0, 1.0f, 1.0f);

  // density sampling

  density = texture(volumeSampler, texturePos).r;

  // gradient calculation based on dataset values

  gradient = -vec3(texture(volumeSampler, vec3(texturePos.x+sampleDistance.x*gscf.x, texturePos.yz)).x-
                   texture(volumeSampler, vec3(texturePos.x-sampleDistance.x*gscf.y, texturePos.yz)).x,
                   texture(volumeSampler, vec3(texturePos.x,texturePos.y+sampleDistance.y*gscf.y, texturePos.z)).x-
                   texture(volumeSampler, vec3(texturePos.x,texturePos.y-sampleDistance.y*gscf.y, texturePos.z)).x,
                   texture(volumeSampler, vec3(texturePos.xy, texturePos.z+sampleDistance.z*gscf.z)).x-
                   texture(volumeSampler, vec3(texturePos.xy, texturePos.z-sampleDistance.z*gscf.z)).x);
  gradient = normalize(mat3(viewMatrix)*transpose(mat3(volumes[0].textureMatrix))*gradient);
} // getVolumeSample

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

void updateActiveObjects(inout int objectId, out int activeObjects)
{
  if (objectId > 0)
  {
    activeObjects |= 1 << (objectId);
  }
  else if (objectId < 0)
  {
    objectId = -objectId;
    activeObjects &= (~(1 << objectId));
  }
}

vec3 calculateLighting(vec4 color, vec3 startpos_eye, vec3 gradient)
{
  const vec3 lightPos = vec3(0.0f, 0.0f, 0.0f);
  vec3 lightDir = normalize(lightPos - startpos_eye);
  vec3 viewDir = -1.0f * normalize(startpos_eye);
  vec3 nGradient = normalize(gradient);

  // float dotNL = abs(dot(ngradient, lightDir));
  float dotNL = max(dot(nGradient, lightDir), 0.0f);
  vec3 H = normalize(lightDir + viewDir);
  // float dotNH = abs(dot(ngradient, H));
  float dotNH = max(dot(nGradient, H), 0.0f);
  float ka = 0.3;  // gl_LightSource[li].ambient.xyz
  float kd = 0.5;  // gl_LightSource[li].diffuse.xyz
  float ks = 0.5;  // gl_LightSource[li].specular.xyz
  float shininess = 32.0f;
  vec3 specularColor = vec3(1.0, 1.0, 1.0);

  vec3 specular = (shininess + 2.0f) / (2.0f * 3.1415f) * ks * color.a *
                  specularColor * pow(dotNH, shininess);
  return ka * color.rgb + kd * color.rgb * dotNL + specular;
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
  vec3 lastpos_eye;
  vec3 endpos_eye;
  vec3 segment_startpos_eye;
  vec3 dirvec_eye;
  vec4 pos_proj;

  float density = 0.0f;
  vec3 gradient = vec3(0.0f);

  int objectId = -1;

  vec4 finalColor = vec4(0, 0, 0, 0);

  for (uint age = 1; age < maxage; age++)  // all fragments
  {
    vec4 fragmentColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    if (age == 1)
    {
      current_fragment_read_status = fetchFragment(ij, age, current_fragment);
      segment_startpos_eye = current_fragment.eyePos.xyz;
    }
    else
    {
      current_fragment_read_status = next_fragment_read_status;
      current_fragment = next_fragment;
      segment_startpos_eye = endpos_eye;
    }

    objectId = current_fragment.objectId;
    updateActiveObjects(objectId, activeobjects);
    activeobjectcount = bitCount(activeobjects);

    // fetch next Fragment
    next_fragment_read_status = fetchFragment(ij, age + 1, next_fragment);
    if (next_fragment_read_status)
    {
      endpos_eye = next_fragment.eyePos.xyz;
    }
    else
    {
      endpos_eye = segment_startpos_eye;
    }

    // set up segment direction vector

    dirvec_eye = endpos_eye - segment_startpos_eye;

    // FIXME: do we need it?
    // pos_proj = projectionMatrix*vec4(startpos_eye, 1.0f);
    // pos_proj.z /= pos_proj.w;
    // FIXME: posproj.xy ???
    // pos_proj += 1.0f;
    // pos_proj /= 2.0f;


    if (activeobjectcount > 0)  // in frag
    {
      uint ao = activeobjects;
      int aoc = activeobjectcount;

      // calculate length in texture space (needed for step width calculation)
      float segment_texture_length = 0.0;
      for (int oi = 0; oi < activeobjectcount; oi++)
      {
        int objectID = findLSB(ao);
        ao &= (~(1 << objectID));

        vec4 textureStartPos = volumes[0].textureMatrix * inverseViewMatrix *
                               current_fragment.eyePos;
        vec4 textureEndPos = volumes[0].textureMatrix * inverseViewMatrix *
                             next_fragment.eyePos;

        segment_texture_length =
            max(distance(textureStartPos.xyz * textureAtlasSize,
                         textureEndPos.xyz * textureAtlasSize),
                segment_texture_length);
        // float value = texture(volumeSampler, texCoord.xyz).r;
        // fragmentColor.color.rgb = vec3(value);
        // fragmentColor.color.rgb = vec3(textureStartPos.xyz);
        // vec4 transferFunction = Texture(volumes[0].textureAddress,
        //   vec2(textureStartPos.x, volumes[0].transferFunctionRow));
        // fragmentColor = transferFunction;
      }

      activeobjectcount = aoc;
      //FIXME:
      segment_texture_length =
          (segment_texture_length >= 0.0f)
              ? segment_texture_length
              : distance(segment_startpos_eye, endpos_eye) * 100.0f;

      if (activeobjectcount > 0 && segment_texture_length > 0.0f)
      {
        int sample_steps = int(segment_texture_length * STEP_FACTOR);
        sample_steps = clamp(sample_steps, 1, MAX_SAMPLES - 1);
        float stepFactor = 1.0 / float(sample_steps);

        // noise offset
        startpos_eye = segment_startpos_eye;  // + noise offset;

        // FIXME: check code
        // textureStartPos = textureStartPos +
        // noiseoffset*dirvec_eye*stepfactor;
        lastpos_eye = startpos_eye - dirvec_eye * stepFactor;

        // sample ray segment
        for (int step = 0; step < sample_steps; step++)
        {
          vec4 sampleColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);

          uint ao = activeobjects;

          // sampling per non-isosurface object
          for (int objectIndex = 0; objectIndex < activeobjectcount;
               objectIndex++)
          {
            int objectID = findLSB(ao);
            if (objectID < 0)
              break;
            ao &= (~(1 << (objectID)));

            float squareGradientLength = 0.0f;
            vec4 currentColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
            vec4 TFColor;

            vec3 textureSamplePos =
                (volumes[0].textureMatrix * inverseViewMatrix *
                 vec4(startpos_eye, 1.0f)).xyz;

            getVolumeSample(objectID, textureSamplePos, density, gradient);
            squareGradientLength = dot(gradient, gradient);

            // transfer function lookup
            TFColor = Texture(volumes[0].textureAddress,
                              vec2(density, volumes[0].transferFunctionRow));

            // lighting
            if (squareGradientLength > 0.05f)
            {
              currentColor.xyz += calculateLighting(TFColor, startpos_eye, gradient);
            }
            else
            {
              currentColor.xyz += TFColor.xyz;
            }

            // clamp color
            currentColor.xyz = clamp(currentColor.xyz, 0.0f, 1.0f);
            currentColor.w = TFColor.w;

            // we sum up overlapping contributions
            sampleColor += currentColor;
          }  // per active object loop

          // clamp cumulatie sample value
          clamp(sampleColor, vec4(0.0f, 0.0f, 0.0f, 0.0f),
                vec4(1.0f, 1.0f, 1.0f, 1.0f));

          // sample accumulation
          fragmentColor =
              fragmentColor + sampleColor * (1.0f - fragmentColor.w);

          // early ray termination
          if (fragmentColor.w > 0.999)
            break;

          // prepare next segment
          lastpos_eye = startpos_eye;
          startpos_eye += stepFactor * dirvec_eye;

          // FIXME: do we need it?
          // pos_proj = gl_ProjectionMatrix*vec4(startpos_eye,1.0f);
          // pos_proj.z /= pos_proj.w;
          // pos_proj.z += 1.0f;
          // pos_proj.z /=2.0f;
        }  // sampling steps
      }  // if (activeobjectcount > 0) ...
    }
    else
    {
      fragmentColor = current_fragment.color;
      // current_fragment.color = vec4(0.0, 1.0, 1.0, 0.5);
    }

    if (current_fragment_read_status)
    {
      // set Fragment value
      finalColor = finalColor + fragmentColor * (1.0f - finalColor.a);
      //finalColor = blend(finalColor, fragmentColor);
    }

    if (finalColor.a > 0.999)
    {
      break;
    }

    // break; // just the first fragment
  }  // all ages except last ...

  if (maxage == 1)
  {
    next_fragment_read_status = fetchFragment(ij, 1, next_fragment);
  }

  if (next_fragment_read_status)
  {
    finalColor = blend(finalColor, next_fragment.color);
  }

  o_PixColor = blend(finalColor, vec4(BkgColor, 1.0));
}

