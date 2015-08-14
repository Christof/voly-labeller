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
vec4 cutPositions[18];
int cutPositionCount = 0;

void justEmit(vec4 pos, vec4 color)
{
  vertexPos = pos;
  gl_Position = vertexPos;
  vertexColor = color;
  EmitVertex();
}

void emit(vec4 pos, vec4 color)
{
  if (isFirst)
  {
    firstPosition = pos;
    isFirst = false;
  }

  justEmit(pos, color);
  ++triangleCount;

  if (triangleCount == 3)
  {
    EndPrimitive();
    if (hasTwoTriangles)
    {
      justEmit(pos, color);
      ++triangleCount;
    }
  }
}

void processTriangle(mat4 matrix, vec4 triangle[3])
{
  const float cutOffZ = 0.4;
  triangleCount = 0;
  isFirst = true;

  mat4 colorMat = inverse(matrix);

  for (int i = 0; i < 3; ++i)
  {
    vec4 inPos = triangle[i];
    if (inPos.z >= cutOffZ)
    {
      vec4 c = (colorMat * inPos) * 0.5f + vec4(0.5, 0.5f, 0.5f, 0);
      c.a = 0.5f;
      emit(inPos, c);  // vColor[0]);
    }
    vec4 nextPos = triangle[(i + 1) % 3];
    if ((nextPos.z < cutOffZ && inPos.z >= cutOffZ) ||
        (nextPos.z >= cutOffZ && inPos.z < cutOffZ))
    {
      hasTwoTriangles = true;
      vec4 edge = inPos - nextPos;
      float lambda = (cutOffZ - inPos.z) / edge.z;

      vec4 newPos = inPos + lambda * edge;
      cutPositions[cutPositionCount++] = newPos;
      vec4 c = (colorMat * newPos) * 0.5f + vec4(0.5, 0.5f, 0.5f, 0);
      c.a = 0.5f;
      emit(newPos, c);  // vec4(1, 1, 0, 1));
    }
  }

  if (triangleCount == 5)
    emit(firstPosition, vec4(0, 1, 1, 1));

  EndPrimitive();
}

void processSide(mat4 matrix, vec4 center, vec4 side, vec4 varying1,
                 vec4 varying2)
{
  vec4 triangle[3] = vec4[3](matrix * (center + side - varying1 - varying2),
                             matrix * (center + side - varying1 + varying2),
                             matrix * (center + side + varying1 - varying2));
  processTriangle(matrix, triangle);

  triangle = vec4[3](matrix * (center + side + varying1 - varying2),
                     matrix * (center + side - varying1 + varying2),
                     matrix * (center + side + varying1 + varying2));
  processTriangle(matrix, triangle);
}

void main()
{
  mat4 model = Transforms[vDrawId[0]];
  mat4 matrix = modelViewProjectionMatrix * model;
  const vec4 xAxis = vec4(0.5, 0, 0, 0);
  const vec4 yAxis = vec4(0, 0.5, 0, 0);
  const vec4 zAxis = vec4(0, 0, 0.5, 0);

  vec4 center = gl_in[0].gl_Position;

  processSide(matrix, center, yAxis, xAxis, zAxis);
  processSide(matrix, center, xAxis, yAxis, zAxis);
  processSide(matrix, center, zAxis, xAxis, yAxis);
  processSide(matrix, center, -yAxis, xAxis, zAxis);
  processSide(matrix, center, -xAxis, yAxis, zAxis);
  processSide(matrix, center, -zAxis, xAxis, yAxis);

  for (int i = 0; i < cutPositionCount; ++i)
  {
    justEmit(cutPositions[i], vColor[0]);
  }
  EndPrimitive();
}

