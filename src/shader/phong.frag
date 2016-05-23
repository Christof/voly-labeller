#version 440

#include "HABufferImplementation.hglsl"

in vec4 outColor;
in vec3 outNormal;
in vec4 outPosition;
in vec4 outEyePosition;
in vec2 outTextureCoordinate;
in flat int outDrawId;

uniform vec3 lightPos_eye = vec3(0.0f, 0.0f, 0.0f);

struct PhongMaterial
{
  vec4 ambientColor;
  vec4 diffuseColor;
  vec4 specularColor;
  float shininess;
};

struct CustomBuffer
{
  PhongMaterial material;
  mat4 normalMatrix;
};

layout(std140, binding = 1) buffer CB1
{
  CustomBuffer customBuffer[];
};

FragmentData computeData()
{
  vec3 dir = normalize(lightPos_eye - outEyePosition.xyz);
  vec3 reflectionDir = normalize(reflect(dir, outNormal));
  vec3 cameraDir = normalize(outEyePosition.xyz);
  vec4 color = outColor;
  PhongMaterial material = customBuffer[outDrawId].material;
  // display normals for debugging
  // color.rgb = outNormal * 0.5f + vec3(0.5f, 0.5f, 0.5f);
  color.rgb = material.diffuseColor.rgb * max(dot(dir, outNormal), 0.0f) +
              material.ambientColor.rgb;
  vec3 specular = pow(max(dot(reflectionDir, cameraDir), 0.0),
                      0.3 * material.shininess) *
                  material.specularColor.rgb;
  color.rgb += specular;
  // color.rg = outTextureCoordinate;
  // color.a = 0.5;

  FragmentData data;
  data.color = color;
  data.eyePos = outEyePosition;
  data.objectId = 0;

  return data;
}

#include "buildHABuffer.hglsl"
