#version 330

in vec4 outColor;
in vec3 outNormal;
in vec3 outPosition;
out vec4 color;

uniform vec3 lightPosition = vec3(2.0f, 10.0f, 0.0f);
uniform vec3 ambientColor = vec3(0.1, 0.4, 0.1f);

void main()
{
  vec3 dir = normalize(lightPosition - outPosition);
  color = outColor;
  // display normals for debugging
  // color.rgb = outNormal * 0.5f + vec3(0.5f, 0.5f, 0.5f);
  color.rgb = color.rgb * dot(dir, outNormal) + ambientColor;
}
