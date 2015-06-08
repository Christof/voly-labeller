#version 330

in vec4 outColor;
in vec3 outNormal;
in vec3 outPosition;
out vec4 color;

uniform vec3 lightPosition = vec3(2.0f, 10.0f, 0.0f);
uniform vec4 ambientColor = vec4(0.1, 0.4, 0.1f, 1.0f);
uniform vec4 diffuseColor = vec4(0.1, 0.4, 0.8f, 1.0f);
uniform vec4 specularColor = vec4(0.9, 0.8, 0.8f, 1.0f);
uniform vec3 cameraDirection = vec3(0.0f, 0.0f, 0.0f);
uniform float shininess = 256.0f;

void main()
{
  vec3 dir = normalize(outPosition - lightPosition );
  vec3 reflectionDir = normalize(-reflect(dir, outNormal)); 
  color = outColor;
  // display normals for debugging
  // color.rgb = outNormal * 0.5f + vec3(0.5f, 0.5f, 0.5f);
  color.rgb = diffuseColor.rgb * max(dot(dir, outNormal), 0.0f) + ambientColor.rgb;
  vec3 specular = pow(max(dot(reflectionDir, -cameraDirection), 0.0), 0.3 * shininess) *
    specularColor.rgb;
  color.rgb += specular;
}
