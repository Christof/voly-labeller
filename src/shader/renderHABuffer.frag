#version 440
#include "HABufferImplementation.hglsl"
#include "bindlessTexture.hglsl"
#define MAX_SAMPLES 4096
#define STEP_FACTOR 2.5

in vec4 fragmentInputPosition;

layout(location = 0) out vec4 outputColor;
layout(location = 1) out vec4 position;
layout(depth_any) out float gl_FragDepth;

uniform vec3 backgroundColor = vec3(1.0, 1.0, 1.0);
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 inverseViewMatrix;
uniform vec3 textureAtlasSize;
uniform vec3 sampleDistance;
uniform float alphaThresholdForDepth = 0.1;

uniform sampler3D volumeSampler;

const float POSITION_NOT_SET = -2;

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

float getVolumeSampleDensity(in vec3 texturePos)
{
  return texture(volumeSampler, texturePos).r;
}

vec3 getVolumeSampleGradient(in int objectId, in vec3 texturePos)
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
                   transpose(mat3(volumes[objectId].textureMatrix)) * gradient);
}

// Blending equation for in-order traversal
vec4 blend(vec4 clr, vec4 srf)
{
  return clr + (1.0 - clr.w) * vec4(srf.xyz * srf.w, srf.w);
}

bool fetchFragment(in uvec2 ij, in uint age, out FragmentData fragment)
{
  uvec2 l = (ij + offsets[age]);
  uint64_t h = Haddr(l % uvec2(tableSize));

  uint64_t rec = u_Records[h];

  uint32_t key = uint32_t(rec >> uint64_t(32));

  if (HA_AGE(key) == age)
  {
    fragment = u_FragmentData[uint32_t(rec)];
    return true;
  }

  return false;
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

  float dotNL = max(dot(normalizedGradient, lightDir), 0.0f);
  vec3 H = normalize(lightDir + viewDir);
  float dotNH = max(dot(normalizedGradient, H), 0.0f);
  const float ambient = 0.3;
  const float diffuse = 0.5;
  const float specular = 0.5;
  const float shininess = 32.0f;
  const vec3 specularColor = vec3(1.0, 1.0, 1.0);

  vec3 specularTerm = (shininess + 2.0f) / (2.0f * 3.1415f) * specular *
                      color.a * specularColor * pow(dotNH, shininess);
  return ambient * color.rgb + diffuse * color.rgb * dotNL + specularTerm;
}

vec4 transferFunctionLookUp(int volumeId, float density)
{
  float row = volumes[volumeId].transferFunctionRow;

  return Texture(volumes[volumeId].textureAddress,
                 vec2(density, row / (transferFunctionRowCount - 1.0f)));
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

    vec4 textureStartPos = volumes[currentObjectId].textureMatrix *
                           inverseViewMatrix * currentFragmentPos_eye;
    vec4 textureEndPos = volumes[currentObjectId].textureMatrix *
                         inverseViewMatrix * nextFragmentPos_eye;

    segmentTextureLength = max(distance(textureStartPos.xyz * textureAtlasSize,
                                        textureEndPos.xyz * textureAtlasSize),
                               segmentTextureLength);
  }

  return segmentTextureLength;
}

void setPositionAndDepth(vec4 positionInEyeSpace)
{
  if (position.w == POSITION_NOT_SET)
  {
    vec4 ndcPos = projectionMatrix * positionInEyeSpace;
    ndcPos = ndcPos / ndcPos.w;
    float depth = ndcPos.z;
    gl_FragDepth = depth;
    position.xyz = ndcPos.xyz;
    position.w = 1.0f;
  }
}

vec4 calculateSampleColor(in uint remainingActiveObjects, in int activeObjectCount,
    in vec4 startPos_eye)
{
  vec4 sampleColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);

  // sampling per non-isosurface object
  for (int objectIndex = 0; objectIndex < activeObjectCount;
      objectIndex++)
  {
    int currentObjectId = calculateNextObjectId(remainingActiveObjects);
    if (currentObjectId < 0)
      break;

    vec3 textureSamplePos =
      (volumes[currentObjectId].objectToDatasetMatrix *
       volumes[currentObjectId].textureMatrix * inverseViewMatrix *
       startPos_eye).xyz;

    float density = getVolumeSampleDensity(textureSamplePos);
    vec4 currentColor = transferFunctionLookUp(currentObjectId, density);
    vec3 gradient = getVolumeSampleGradient(currentObjectId, textureSamplePos);
    float squareGradientLength = dot(gradient, gradient);

    if (squareGradientLength > 0.05f)
    {
      currentColor.xyz = calculateLighting(currentColor, startPos_eye.xyz, gradient);
    }
    currentColor.xyz = clamp(currentColor.xyz, vec3(0.0f), vec3(1.0f));

    // we sum up overlapping contributions
    sampleColor += currentColor;
  }  // per active object loop

  // clamp cumulatie sample value
  return clamp(sampleColor, vec4(0.0f), vec4(1.0f));
}

vec4 calculateColorOfVolumes(in int activeObjects, in int activeObjectCount,
    in vec4 segmentStartPos_eye, in vec4 endPos_eye, in vec4 fragmentColor)
{
  float segmentTextureLength  = calculateSegmentTextureLength(activeObjectCount,
      activeObjects, segmentStartPos_eye, endPos_eye);
  int sampleSteps = int(segmentTextureLength * STEP_FACTOR);
  sampleSteps = clamp(sampleSteps, 1, MAX_SAMPLES - 1);
  float stepFactor = 1.0 / float(sampleSteps);

  // set up segment direction vector
  vec4 direction_eye = endPos_eye - segmentStartPos_eye;
  vec4 startPos_eye = segmentStartPos_eye;  // + noise offset;

  // sample ray segment
  for (int step = 0; step < sampleSteps; step++)
  {
    vec4 sampleColor = calculateSampleColor(activeObjects,
        activeObjectCount, startPos_eye);

    // sample accumulation
    fragmentColor = fragmentColor + sampleColor * (1.0f - fragmentColor.w);

    if (fragmentColor.w > alphaThresholdForDepth)
    {
      setPositionAndDepth(startPos_eye);
    }

    // early ray termination
    if (fragmentColor.w > 0.999)
      break;

    // prepare next segment
    startPos_eye += stepFactor * direction_eye;
  }  // sampling steps

  return fragmentColor;
}

void main()
{
  if (gl_SampleMaskIn[0] == 0)
    discard;

  vec2 pos = (fragmentInputPosition.xy * 0.5 + 0.5) * float(u_ScreenSz);

  if (pos.x >= u_ScreenSz || pos.y >= u_ScreenSz || pos.x < 0 || pos.y < 0)
  {
    outputColor = vec4(1.0, 1.0, 0.3, 1.0);
    return;
  }

  uvec2 ij = uvec2(pos.xy);

  gl_FragDepth = 1.0;
  position = vec4(-2, -2, 1, POSITION_NOT_SET);

  uint maxAge = u_Counts[Saddr(ij)];

  if (maxAge == 0)
    discard;  // no fragment, early exit

  int activeObjects = 0;
  FragmentData currentFragment;
  FragmentData nextFragment;
  bool nextFragmentReadStatus = false;

  vec4 endPos_eye;

  vec4 finalColor = vec4(0, 0, 0, 0);

  uint age = 1;
  while (!nextFragmentReadStatus && age <= maxAge)
  {
      nextFragmentReadStatus = fetchFragment(ij, age, nextFragment);
      endPos_eye = nextFragment.eyePos;
      ++age;
  }

  for (--age; age < maxAge; age++)  // all fragments
  {
    currentFragment = nextFragment;
    vec4 segmentStartPos_eye = endPos_eye;
    vec4 fragmentColor = currentFragment.color;
    fragmentColor.xyz *= fragmentColor.w;

    int objectId = currentFragment.objectId;
    updateActiveObjects(objectId, activeObjects);
    int activeObjectCount = bitCount(activeObjects);

    if (objectId == 0 && fragmentColor.w > alphaThresholdForDepth)
    {
      setPositionAndDepth(currentFragment.eyePos);
    }

    // fetch next Fragment
    nextFragmentReadStatus = false;
    while (!nextFragmentReadStatus && age < maxAge)
    {
      nextFragmentReadStatus = fetchFragment(ij, age + 1, nextFragment);
      ++age;
    }
    --age;

    endPos_eye = nextFragmentReadStatus ?
      nextFragment.eyePos : segmentStartPos_eye;

    if (activeObjectCount > 0)
    {
      fragmentColor = calculateColorOfVolumes(activeObjects, activeObjectCount,
          segmentStartPos_eye, endPos_eye, fragmentColor);
      finalColor = finalColor + fragmentColor * (1.0f - finalColor.a);
    }
    else
    {
      finalColor = blend(finalColor, currentFragment.color);
    }

    if (finalColor.a > 0.999)
      break;

    // break; // just the first fragment
  }  // all ages except last ...

  if (nextFragmentReadStatus)
  {
    finalColor = blend(finalColor, nextFragment.color);
    if (nextFragment.color.w > alphaThresholdForDepth)
    {
      setPositionAndDepth(nextFragment.eyePos);
    }
  }

  if (position.w == POSITION_NOT_SET)
    position.w = 1;

  finalColor = clamp(finalColor, vec4(0.0), vec4(1.0));

  outputColor = blend(finalColor, vec4(backgroundColor, 1.0));
}

