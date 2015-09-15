#version 440
#include "HABufferImplementation.hglsl"
#include "bindlessTexture.hglsl"
#define MAX_SAMPLES 4096
#define STEP_FACTOR 2.5

in vec4 u_Pos;

out vec4 o_PixColor;
layout(depth_any) out float gl_FragDepth;

uniform vec3 backgroundColor = vec3(1.0, 1.0, 1.0);
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 inverseViewMatrix;
uniform vec3 textureAtlasSize;
uniform vec3 sampleDistance;

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

float getVolumeSampleDensity(in int objectId, in vec3 texturePos)
{
  return texture(volumeSampler, texturePos).r;
}

vec3 getVolumeSampleGradient(in int objectID, in vec3 texturePos)
{
  const vec3 gscf = vec3(1.0, 1.0f, 1.0f);

  // gradient calculation based on dataset values using central differences
  vec3 gradient = -vec3(texture(volumeSampler, vec3(texturePos.x+sampleDistance.x*gscf.x, texturePos.yz)).x-
                   texture(volumeSampler, vec3(texturePos.x-sampleDistance.x*gscf.y, texturePos.yz)).x,
                   texture(volumeSampler, vec3(texturePos.x,texturePos.y+sampleDistance.y*gscf.y, texturePos.z)).x-
                   texture(volumeSampler, vec3(texturePos.x,texturePos.y-sampleDistance.y*gscf.y, texturePos.z)).x,
                   texture(volumeSampler, vec3(texturePos.xy, texturePos.z+sampleDistance.z*gscf.z)).x-
                   texture(volumeSampler, vec3(texturePos.xy, texturePos.z-sampleDistance.z*gscf.z)).x);
  return normalize(mat3(viewMatrix) *
                   transpose(mat3(volumes[0].textureMatrix)) * gradient);
}

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

void updateActiveObjects(inout int objectId, inout int activeObjects)
{
  if (objectId > 0)
  {
    activeObjects |= 1 << (objectId - 1);
  }
  else if (objectId < 0)
  {
    objectId = -objectId;
    activeObjects &= (~(1 << (objectId - 1)));
  }
}

vec3 calculateLighting(vec4 color, vec3 startPos_eye, vec3 gradient)
{
  const vec3 lightPos = vec3(0.0f, 0.0f, 0.0f);
  vec3 lightDir = normalize(lightPos - startPos_eye);
  vec3 viewDir = -1.0f * normalize(startPos_eye);
  vec3 normalizedGradient = normalize(gradient);

  // float dotNL = abs(dot(ngradient, lightDir));
  float dotNL = max(dot(normalizedGradient, lightDir), 0.0f);
  vec3 H = normalize(lightDir + viewDir);
  // float dotNH = abs(dot(ngradient, H));
  float dotNH = max(dot(normalizedGradient, H), 0.0f);
  float ka = 0.3;  // gl_LightSource[li].ambient.xyz
  float kd = 0.5;  // gl_LightSource[li].diffuse.xyz
  float ks = 0.5;  // gl_LightSource[li].specular.xyz
  float shininess = 32.0f;
  vec3 specularColor = vec3(1.0, 1.0, 1.0);

  vec3 specular = (shininess + 2.0f) / (2.0f * 3.1415f) * ks * color.a *
                  specularColor * pow(dotNH, shininess);
  return ka * color.rgb + kd * color.rgb * dotNL + specular;
}

vec4 transferFunctionLookUp(int volumeId, float density)
{
  return Texture(volumes[volumeId].textureAddress,
                              vec2(density, volumes[volumeId].transferFunctionRow));
}

int calculateNextObjectId(inout uint remainingActiveObjects)
{
  int currentObjectId = findLSB(remainingActiveObjects);
  remainingActiveObjects &= (~(1 << (currentObjectId)));

  return currentObjectId;
}

float calculateSegmentTextureLength(int activeObjectCount, uint activeObjects,
    vec4 currentFragmentPos_eye, vec4 nextFragmentPos_eye)
{
  float segmentTextureLength = 0.0f;
  for (int oi = 0; oi < activeObjectCount; oi++)
  {
    int currentObjectId = calculateNextObjectId(activeObjects);

    vec4 textureStartPos =
        volumes[0].textureMatrix * inverseViewMatrix * currentFragmentPos_eye;
    vec4 textureEndPos =
        volumes[0].textureMatrix * inverseViewMatrix * nextFragmentPos_eye;

    segmentTextureLength = max(distance(textureStartPos.xyz * textureAtlasSize,
                                        textureEndPos.xyz * textureAtlasSize),
                               segmentTextureLength);
  }

  return segmentTextureLength;
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

  uint maxAge = u_Counts[Saddr(ij)];

  if (maxAge == 0)
  {
    discard;  // no fragment, early exit
  }

  int activeObjects = 0;
  int activeObjectCount = 0;
  FragmentData currentFragment;
  FragmentData nextFragment;
  bool nextFragmentReadStatus = false;

  vec3 startPos_eye;
  vec3 lastPos_eye;
  vec3 endPos_eye;
  vec3 segmentStartPos_eye;
  vec3 direction_eye;
  vec4 pos_proj;

  vec3 gradient = vec3(0.0f);

  int objectId = -1;

  vec4 finalColor = vec4(0, 0, 0, 0);

  uint age = 1;
  while (!nextFragmentReadStatus && age < maxAge)
  {
      nextFragmentReadStatus = fetchFragment(ij, age, nextFragment);
      endPos_eye = nextFragment.eyePos.xyz;
      ++age;
  }

  for (--age; age < maxAge; age++)  // all fragments
  {
    currentFragment = nextFragment;
    segmentStartPos_eye = endPos_eye;
    vec4 fragmentColor = currentFragment.color;
    fragmentColor.xyz *= fragmentColor.w;

    objectId = currentFragment.objectId;
    updateActiveObjects(objectId, activeObjects);
    activeObjectCount = bitCount(activeObjects);

    // fetch next Fragment
    nextFragmentReadStatus = false;
    while (!nextFragmentReadStatus && age < maxAge)
    {
      nextFragmentReadStatus = fetchFragment(ij, age + 1, nextFragment);
      ++age;
    }
    --age;

    if (nextFragmentReadStatus)
    {
      endPos_eye = nextFragment.eyePos.xyz;
    }
    else
    {
      endPos_eye = segmentStartPos_eye;
    }

    // set up segment direction vector
    direction_eye = endPos_eye - segmentStartPos_eye;

    // FIXME: do we need it?
    // pos_proj = projectionMatrix*vec4(startPos_eye, 1.0f);
    // pos_proj.z /= pos_proj.w;
    // pos_proj += 1.0f;
    // pos_proj /= 2.0f;

    if (activeObjectCount > 0)
    {
      // float value = texture(volumeSampler, texCoord.xyz).r;
      // fragmentColor.color.rgb = vec3(value);
      // fragmentColor.color.rgb = vec3(textureStartPos.xyz);
      // vec4 transferFunction = Texture(volumes[0].textureAddress,
      //   vec2(textureStartPos.x, volumes[0].transferFunctionRow));
      // fragmentColor = transferFunction;
      float segmentTextureLength  = calculateSegmentTextureLength(activeObjectCount, activeObjects,
        currentFragment.eyePos, nextFragment.eyePos);
      int sampleSteps = int(segmentTextureLength * STEP_FACTOR);
      sampleSteps = clamp(sampleSteps, 1, MAX_SAMPLES - 1);
      float stepFactor = 1.0 / float(sampleSteps);

      // noise offset
      startPos_eye = segmentStartPos_eye;  // + noise offset;

      // FIXME: check code
      // textureStartPos = textureStartPos +
      // noiseoffset*direction_eye*stepfactor;
      lastPos_eye = startPos_eye - direction_eye * stepFactor;

      // sample ray segment
      for (int step = 0; step < sampleSteps; step++)
      {
        vec4 sampleColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);

        uint remainingActiveObjects = activeObjects;

        // sampling per non-isosurface object
        for (int objectIndex = 0; objectIndex < activeObjectCount;
             objectIndex++)
        {
          int currentObjectId = calculateNextObjectId(remainingActiveObjects);
          if (currentObjectId < 0)
            break;

          vec3 textureSamplePos =
              (volumes[0].objectToDatasetMatrix * volumes[0].textureMatrix *
               inverseViewMatrix * vec4(startPos_eye, 1.0f)).xyz;

          float density = getVolumeSampleDensity(currentObjectId, textureSamplePos);
          vec3 gradient = getVolumeSampleGradient(currentObjectId, textureSamplePos);
          float squareGradientLength = dot(gradient, gradient);

          vec4 currentColor = transferFunctionLookUp(0, density);
          if (squareGradientLength > 0.05f)
          {
            currentColor.xyz = calculateLighting(currentColor, startPos_eye, gradient);
          }
          currentColor.xyz = clamp(currentColor.xyz, vec3(0.0f), vec3(1.0f));

          // we sum up overlapping contributions
          sampleColor += currentColor;
        }  // per active object loop

        // clamp cumulatie sample value
        sampleColor = clamp(sampleColor, vec4(0.0f), vec4(1.0f));

        // sample accumulation
        fragmentColor =
            fragmentColor + sampleColor * (1.0f - fragmentColor.w);

        // early ray termination
        if (fragmentColor.w > 0.999)
          break;

        // prepare next segment
        lastPos_eye = startPos_eye;
        startPos_eye += stepFactor * direction_eye;

        // FIXME: do we need it?
        // pos_proj = gl_ProjectionMatrix*vec4(startPos_eye,1.0f);
        // pos_proj.z /= pos_proj.w;
        // pos_proj.z += 1.0f;
        // pos_proj.z /=2.0f;
      }  // sampling steps

      finalColor = finalColor + fragmentColor * (1.0f - finalColor.a);
    }  // if (activeObjectCount > 0) ...
    else
    {
      finalColor = blend(finalColor, currentFragment.color);
    }

    if (finalColor.a > 0.999)
    {
      break;
    }

    // break; // just the first fragment
  }  // all ages except last ...

  if (maxAge == 1)
  {
    nextFragmentReadStatus = fetchFragment(ij, 1, nextFragment);
  }

  if (nextFragmentReadStatus)
  {
    finalColor = blend(finalColor, nextFragment.color);
  }

  finalColor = clamp(finalColor, vec4(0.0), vec4(1.0));

  o_PixColor = blend(finalColor, vec4(backgroundColor, 1.0));
}

