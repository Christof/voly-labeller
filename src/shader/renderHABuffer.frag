#version 440
#include "HABufferImplementation.hglsl"
#include "bindlessTexture.hglsl"
#define MAX_SAMPLES 4096
#define STEP_FACTOR 2.5

in vec4 fragmentInputPosition;

layout(location = 0) out vec4 outputColor;
layout(location = 1) out vec4 outputColor2;
layout(location = 2) out vec4 outputColor3;
layout(location = 3) out vec4 outputColor4;
layout(location = 4) out vec4 accumulatedOutputColor;
layout(depth_any) out float gl_FragDepth;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 inverseViewMatrix;
uniform vec3 textureAtlasSize;
uniform int transferFunctionWidth;
uniform vec3 sampleDistance;
uniform float alphaThresholdForDepth = 0.1;
uniform int layerCount;
uniform int planeCount;
const int maxPlaneCount = 3;
uniform vec4 layerPlanes[maxPlaneCount];
uniform float planesZValuesNdc[maxPlaneCount];
uniform vec3 lightPos_eye = vec3(0.0f, 0.0f, 0.0f);

uniform sampler3D volumeSampler;

const float DEPTH_NOT_SET = -10.0f;

#define transferFunctionRowCount 64.0f

struct VolumeData
{
  Tex2DAddress textureAddress;
  mat4 textureMatrix;
  mat4 gradientMatrix;
  mat4 objectToDatasetMatrix;
  int volumeId;
  int transferFunctionRow;
};

layout(std430, binding = 1) buffer CB1
{
  VolumeData volumes[];
};

layout(std140, binding = 2) buffer CB2
{
  Tex2DAddress noiseAddresses[];
};

vec4 clampColor(in vec4 color)
{
  return clamp(color, vec4(0), vec4(1));
}

vec3 clampColor(in vec3 color)
{
  return clamp(color, vec3(0), vec3(1));
}

float getVolumeSampleDensity(in vec3 texturePos)
{
  return texture(volumeSampler, texturePos).r;
}

// gradient calculation based on dataset values using central differences
vec3 getVolumeSampleGradient(in int objectId, in vec3 texturePos)
{
  vec3 gradient = vec3(texture(volumeSampler, vec3(texturePos.x + sampleDistance.x, texturePos.yz)).x -
      texture(volumeSampler, vec3(texturePos.x - sampleDistance.x, texturePos.yz)).x,
      texture(volumeSampler, vec3(texturePos.x, texturePos.y + sampleDistance.y, texturePos.z)).x -
      texture(volumeSampler, vec3(texturePos.x, texturePos.y - sampleDistance.y, texturePos.z)).x,
      texture(volumeSampler, vec3(texturePos.xy, texturePos.z + sampleDistance.z)).x -
      texture(volumeSampler, vec3(texturePos.xy, texturePos.z - sampleDistance.z)).x);

  return normalize(mat3(volumes[objectId].gradientMatrix) * -gradient);
}

// Blending equation for in-order traversal
vec4 blend(vec4 clr, vec4 srf)
{
  return clr + (1.0 - clr.a) * vec4(srf.rgb * srf.a, srf.a);
}

bool fetchFragment(in uvec2 ij, in uint age, out FragmentData fragment)
{
  uvec2 l = (ij + offsets[age]);
  uint64_t h = Haddr(l % uvec2(tableSize));

  uint64_t rec = records[h];

  uint32_t key = uint32_t(rec >> uint64_t(32));

  if (HA_AGE(key) == age)
  {
    fragment = fragmentData[uint32_t(rec)];
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

vec3 calculateLighting(vec4 color, vec3 currentPos_eye, vec3 gradient)
{
  vec3 lightDir = normalize(lightPos_eye - currentPos_eye);
  vec3 viewDir = -1.0f * normalize(currentPos_eye);
  vec3 normalizedGradient = normalize(gradient);

  float dotNL = max(dot(normalizedGradient, lightDir), 0.0f);
  vec3 H = normalize(lightDir + viewDir);
  float dotNH = max(dot(normalizedGradient, H), 0.0f);
  const float ambient = 0.3;
  const float diffuse = 0.5;
  const float specular = 0.5;
  const float shininess = 8.0f;
  const vec3 specularColor = vec3(1.0, 1.0, 1.0);

  vec3 specularTerm = (shininess + 2.0f) / (2.0f * 3.1415f) * specular *
                      color.a * specularColor * pow(dotNH, shininess);
  return ambient * color.rgb + diffuse * color.rgb * dotNL + specularTerm;
}

vec4 transferFunctionLookUp(int volumeId, float density)
{
  return TexelFetch(volumes[volumeId].textureAddress,
                    ivec2(int(density * transferFunctionWidth),
                          volumes[volumeId].transferFunctionRow));
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

vec4 fromEyeToNdcSpace(vec4 positionInEyeSpace)
{
  vec4 ndcPos = projectionMatrix * positionInEyeSpace;
  return ndcPos / ndcPos.w;
}

void setDepthFor(in vec4 positionInEyeSpace)
{
  if (gl_FragDepth != DEPTH_NOT_SET)
    return;

  vec4 ndcPos = fromEyeToNdcSpace(positionInEyeSpace);
  float depth = 0.5f * ndcPos.z + 0.5f;
  gl_FragDepth = depth;
}

void setColorForLayer(int layerIndex, vec4 color)
{
  if (layerIndex == 0)
    outputColor = color;
  else if (layerIndex == 1)
    outputColor2 = color;
  else if (layerIndex == 2)
    outputColor3 = color;
  else
    outputColor4 = color;
}

vec4 calculateSampleColor(in uint remainingActiveObjects, in int activeObjectCount,
    in vec4 currentPos_eye)
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
       currentPos_eye).xyz;

    float density = getVolumeSampleDensity(textureSamplePos);
    vec4 currentColor = transferFunctionLookUp(currentObjectId, density);
    vec3 gradient = getVolumeSampleGradient(currentObjectId, textureSamplePos);
    float squareGradientLength = dot(gradient, gradient);

    if (squareGradientLength > 0.05f)
    {
      currentColor.rgb = calculateLighting(currentColor, currentPos_eye.xyz, gradient);
    }
    currentColor.rgb = clampColor(currentColor.rgb);

    // we sum up overlapping contributions
    sampleColor += currentColor;
  }  // per active object loop

  return clampColor(sampleColor);
}

vec4 calculateColorOfVolumes(in int activeObjects, in int activeObjectCount,
    in vec4 segmentStartPos_eye, in vec4 endPos_eye, in vec4 fragmentColor)
{
  float segmentTextureLength  = calculateSegmentTextureLength(activeObjectCount,
      activeObjects, segmentStartPos_eye, endPos_eye);
  float sampleSteps = segmentTextureLength * STEP_FACTOR;
  sampleSteps = clamp(sampleSteps, 1, MAX_SAMPLES - 1);

  vec4 step_eye = (endPos_eye - segmentStartPos_eye) / sampleSteps;

  vec2 tc = vec2(gl_FragCoord.xy * 4.0f / 1000.0f + 0.5f);
  float noiseOffset = Texture(noiseAddresses[0], tc).x;

  vec4 currentPos_eye = segmentStartPos_eye + noiseOffset * step_eye;

  // sample ray segment
  for (int stepIndex = 0; stepIndex < sampleSteps; stepIndex++)
  {
    vec4 sampleColor = calculateSampleColor(activeObjects,
        activeObjectCount, currentPos_eye);

    // sample accumulation
    fragmentColor = fragmentColor + sampleColor * (1.0f - fragmentColor.a);

    if (fragmentColor.a > alphaThresholdForDepth)
    {
      setDepthFor(currentPos_eye);
    }

    // early ray termination
    if (fragmentColor.a > 0.999)
      break;

    currentPos_eye += step_eye;
  }  // sampling steps

  return fragmentColor;
}

void main()
{
  vec2 pos = (fragmentInputPosition.xy * 0.5 + 0.5) * float(screenSize);

  if (pos.x >= screenSize || pos.y >= screenSize || pos.x < 0 || pos.y < 0)
  {
    outputColor = vec4(1.0, 1.0, 0.3, 1.0);
    return;
  }

  uvec2 ij = uvec2(pos.xy);

  gl_FragDepth = DEPTH_NOT_SET;

  uint maxAge = counters[Saddr(ij)];

  if (maxAge == 0)
    discard;  // no fragment, early exit

  int activeObjects = 0;
  FragmentData currentFragment;
  FragmentData nextFragment;
  bool nextFragmentReadStatus = false;

  vec4 endPos_eye;

  vec4 finalColor = vec4(0);
  accumulatedOutputColor = vec4(0);

  uint age = 1;
  while (!nextFragmentReadStatus && age <= maxAge)
  {
      nextFragmentReadStatus = fetchFragment(ij, age, nextFragment);
      endPos_eye = nextFragment.eyePos;
      ++age;
  }

  int layerIndex = 0;
  float endDistance = dot(endPos_eye, layerPlanes[layerIndex]);
  while (endDistance < 0)
  {
    setColorForLayer(layerIndex, vec4(0));
    ++layerIndex;
    endDistance = dot(endPos_eye, layerPlanes[layerIndex]);

    if (layerIndex == planeCount - 1)
      break;
  }

  for (--age; age < maxAge; age++)  // all fragments
  {
    currentFragment = nextFragment;
    vec4 segmentStartPos_eye = endPos_eye;
    float startDistance = endDistance;
    vec4 fragmentColor = currentFragment.color;
    fragmentColor.rgb *= fragmentColor.a;

    int objectId = currentFragment.objectId;
    updateActiveObjects(objectId, activeObjects);
    int activeObjectCount = bitCount(activeObjects);

    if (objectId == 0 && fragmentColor.a > alphaThresholdForDepth)
    {
      setDepthFor(currentFragment.eyePos);
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

    endDistance = dot(endPos_eye, layerPlanes[layerIndex]);

    while (startDistance >= 0 && endDistance < 0 && layerIndex < planeCount)
    {
      vec3 dir = endPos_eye.xyz - segmentStartPos_eye.xyz;
      float alpha = -dot(segmentStartPos_eye, layerPlanes[layerIndex]) /
                    dot(dir, layerPlanes[layerIndex].xyz);
      vec4 endPosCut_eye = segmentStartPos_eye + alpha * vec4(dir, 0);
      if (activeObjectCount > 0)
      {
        fragmentColor = calculateColorOfVolumes(activeObjects, activeObjectCount,
            segmentStartPos_eye, endPosCut_eye, fragmentColor);
        finalColor = finalColor + fragmentColor * (1.0f - finalColor.a);
        accumulatedOutputColor =
            accumulatedOutputColor +
            fragmentColor * (1.0f - accumulatedOutputColor.a);
      }

      setColorForLayer(layerIndex, finalColor);
      finalColor = vec4(0);
      fragmentColor = vec4(0);
      segmentStartPos_eye = endPosCut_eye;

      ++layerIndex;
      endDistance = dot(endPos_eye, layerPlanes[layerIndex]);
    }

    if (activeObjectCount > 0)
    {
      fragmentColor = calculateColorOfVolumes(activeObjects, activeObjectCount,
          segmentStartPos_eye, endPos_eye, fragmentColor);
      finalColor = finalColor + fragmentColor * (1.0f - finalColor.a);
      accumulatedOutputColor =
          accumulatedOutputColor +
          fragmentColor * (1.0f - accumulatedOutputColor.a);
    }
    else
    {
      finalColor = blend(finalColor, currentFragment.color);
      accumulatedOutputColor = blend(accumulatedOutputColor, currentFragment.color);
    }

    if (finalColor.a > 0.999)
      break;

    // break; // just the first fragment
  }  // all ages except last ...

  if (nextFragmentReadStatus)
  {
    finalColor = blend(finalColor, nextFragment.color);
    accumulatedOutputColor = blend(accumulatedOutputColor, nextFragment.color);
    if (finalColor.a > alphaThresholdForDepth)
    {
      setDepthFor(nextFragment.eyePos);
    }

    float endDistance = dot(nextFragment.eyePos, layerPlanes[layerIndex]);
    while (endDistance < 0 && layerIndex < planeCount)
    {
      setColorForLayer(layerIndex, finalColor);

      finalColor = vec4(0);
      ++layerIndex;
      endDistance = dot(nextFragment.eyePos, layerPlanes[layerIndex]);
    }
  }

  setColorForLayer(layerIndex, finalColor);

  for (++layerIndex; layerIndex < layerCount; ++layerIndex)
  {
    setColorForLayer(layerIndex, vec4(0));
  }

  if (gl_FragDepth == DEPTH_NOT_SET)
    gl_FragDepth = 1.0f;
}

