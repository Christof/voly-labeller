#version 440
#extension GL_NV_shader_buffer_load : enable
#extension GL_NV_gpu_shader5 : enable
#extension GL_NV_shader_atomic_int64 : enable

#define HABuffer 1
#include "HABufferImplementation.hglsl"


in vec4 outColor;
in vec3 outNormal;
in vec4 outPosition;

uniform vec3 lightPosition = vec3(2.0f, 10.0f, 0.0f);
uniform vec4 ambientColor = vec4(0.1, 0.4, 0.1f, 1.0f);
uniform vec4 diffuseColor = vec4(0.1, 0.4, 0.8f, 1.0f);
uniform vec4 specularColor = vec4(0.9, 0.8, 0.8f, 1.0f);
uniform vec3 cameraDirection = vec3(0.0f, 0.0f, 0.0f);
uniform float shininess = 256.0f;

FragmentData computeData()
{
  vec3 dir = normalize(outPosition.xyz - lightPosition );
  vec3 reflectionDir = normalize(-reflect(dir, outNormal)); 
  vec4 color = outColor;
  // display normals for debugging
  // color.rgb = outNormal * 0.5f + vec3(0.5f, 0.5f, 0.5f);
  color.rgb = diffuseColor.rgb * max(dot(dir, outNormal), 0.0f) + ambientColor.rgb;
  vec3 specular = pow(max(dot(reflectionDir, -cameraDirection), 0.0), 0.3 * shininess) *
    specularColor.rgb;
  color.rgb += specular;

  FragmentData data;
  data.color = color;
  data.pos = outPosition;

  return data;
}

#include "buildHABuffer.hglsl"
