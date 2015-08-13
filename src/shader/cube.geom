#version 440

layout(triangles) in;
layout(triangle_strip, max_vertices = 6) out;

in vec4 vPos[];
in vec4 vColor[];
in mat4 matrix[];
in vec3 vTexCoord[];
in int vDrawId[];

out vec4 vertexPos;
out vec4 vertexColor;

uniform mat4 modelViewProjectionMatrix;

layout (std140, binding = 0) buffer CB0
{
    mat4 Transforms[];
};

int triangleCount = 0;
bool hasTwoTriangles = false;
bool isFirst = true;
vec4 firstPosition;

void emit(vec4 pos, vec4 color)
{
  if (isFirst)
  {
    firstPosition = pos;
    isFirst = false;
  }

  vertexPos = pos;
  gl_Position = vertexPos;
  vertexColor = color;
  EmitVertex();
  ++triangleCount;

  if (triangleCount == 3)
  {
    EndPrimitive();
    if (hasTwoTriangles)
    {
      vertexPos = pos;
      gl_Position = vertexPos;
      vertexColor = color;
      EmitVertex();
      ++triangleCount;
    }
  }
}

void main()
{
  mat4 model = Transforms[vDrawId[0]];
  mat4 matrix = modelViewProjectionMatrix * model;

  const float cutOffZ = 0.2;
  for (int i = 0; i < 3; ++i)
  {
    vec4 inPos = matrix * gl_in[i].gl_Position;
    if (inPos.z >= cutOffZ)
    {
      emit(inPos, vColor[i]);
    }
    vec4 nextPos = matrix * gl_in[(i + 1) % 3].gl_Position;
    if ((nextPos.z < cutOffZ && inPos.z >= cutOffZ) ||
        (nextPos.z >= cutOffZ && inPos.z < cutOffZ))
    {
      hasTwoTriangles = true;
      vec4 edge = inPos - nextPos;
      float lambda = (cutOffZ - inPos.z) / edge.z;

      vec4 newPos = inPos + lambda * edge;
      emit(newPos, vec4(1, 1, 0, 1));
    }
  }

  if (triangleCount == 5)
    emit(firstPosition, vec4(0, 1, 1, 1));

  EndPrimitive();
}

