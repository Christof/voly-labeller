#version 440

layout(points) in;
layout(triangle_strip, max_vertices = 42) out;

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

void processTriangle(vec4 triangle[3])
{
  const float cutOffZ = 0.2;
  triangleCount = 0;
  isFirst = true;

  for (int i = 0; i < 3; ++i)
  {
    vec4 inPos = triangle[i];
    if (inPos.z >= cutOffZ)
    {
      emit(inPos, vColor[0]);
    }
    vec4 nextPos = triangle[(i + 1) % 3];
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

void main()
{
  mat4 model = Transforms[vDrawId[0]];
  mat4 matrix = modelViewProjectionMatrix * model;
  const vec4 xAxis = vec4(0.5, 0, 0, 0);
  const vec4 yAxis = vec4(0, 0.5, 0, 0);
  const vec4 zAxis = vec4(0, 0, 0.5, 0);

  vec4 center = gl_in[0].gl_Position;
  vec4 triangle[3] = vec4[3](
    matrix * (center + yAxis - xAxis - zAxis),
    matrix * (center + yAxis - xAxis + zAxis),
    matrix * (center + yAxis + xAxis - zAxis));
  processTriangle(triangle);

  triangle = vec4[3](
    matrix * (center + yAxis + xAxis - zAxis),
    matrix * (center + yAxis - xAxis + zAxis),
    matrix * (center + yAxis + xAxis + zAxis));
  processTriangle(triangle);

  triangle = vec4[3](
    matrix * (center + xAxis - yAxis - zAxis),
    matrix * (center + xAxis - yAxis + zAxis),
    matrix * (center + xAxis + yAxis - zAxis));
  processTriangle(triangle);

  triangle = vec4[3](
    matrix * (center + xAxis + yAxis - zAxis),
    matrix * (center + xAxis - yAxis + zAxis),
    matrix * (center + xAxis + yAxis + zAxis));
  processTriangle(triangle);

  triangle = vec4[3](
    matrix * (center + zAxis - xAxis - yAxis),
    matrix * (center + zAxis - xAxis + yAxis),
    matrix * (center + zAxis + xAxis - yAxis));
  processTriangle(triangle);

  triangle = vec4[3](
    matrix * (center + zAxis + xAxis - yAxis),
    matrix * (center + zAxis - xAxis + yAxis),
    matrix * (center + zAxis + xAxis + yAxis));
  processTriangle(triangle);
}

