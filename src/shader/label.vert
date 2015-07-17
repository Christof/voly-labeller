#version 430

in vec3 position;
in vec2 texcoord;

uniform mat4 modelViewProjectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;

out vec2 vertexTexCoord;
out vec4 vertexPosition;

void main()
{
  vertexTexCoord = texcoord;
  vec3 cameraRight = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
  vec3 cameraUp = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);
  vec3 labelPosition = vec3(modelMatrix[3][0], modelMatrix[3][1], modelMatrix[3][2]);
  vec2 size = vec2(modelMatrix[0][0], modelMatrix[1][1]);
  vec3 pos = labelPosition +
      cameraRight * position.x * size.x +
      cameraUp * position.y * size.y;

  vertexPosition = modelViewProjectionMatrix * vec4(pos, 1.0f);
  vertexPosition.z -= 0.01f;
  gl_Position = vertexPosition;
}
