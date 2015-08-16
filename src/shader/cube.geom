#version 440

layout(points) in;
layout(triangle_strip, max_vertices = 85) out;

in int vDrawId[];

out vec4 vertexPos;
out vec4 vertexColor;

uniform mat4 modelViewProjectionMatrix;

layout(std140, binding = 0) buffer CB0
{
  mat4 Transforms[];
};

vec4 cutPositions[18];
int cutPositionCount = 0;

void emit(const mat4 matrix, const vec4 pos)
{
  vertexPos = matrix * pos;
  gl_Position = vertexPos;
  vertexColor = pos + vec4(0.5, 0.5, 0.5, 0);
  EmitVertex();
}

int emitWithPrimitiveHandling(const mat4 matrix, const vec4 pos,
                              const bool triangleSplittingNecessary,
                              int emittedVertexCount)
{
  emit(matrix, pos);
  ++emittedVertexCount;

  if (emittedVertexCount == 3)
  {
    EndPrimitive();
    if (triangleSplittingNecessary)
    {
      emit(matrix, pos);
      ++emittedVertexCount;
    }
  }

  return emittedVertexCount;
}

void addCutPositionIfNew(const vec4 newPos)
{
  for (int i = 0; i < cutPositionCount; ++i)
  {
    vec4 diff = cutPositions[i] - newPos;
    if (dot(diff, diff) < 0.0001)
      return;
  }

  cutPositions[cutPositionCount++] = newPos;
}

void processTriangle(const mat4 matrix, const vec4 triangle[3])
{
  const float cutOffZ = 0.5;
  int emittedVertexCount = 0;
  vec4 plane = vec4(matrix[0][3] + matrix[0][2], matrix[1][3] + matrix[1][2],
                    matrix[2][3] + matrix[2][2], matrix[3][3] + matrix[3][2]);

  float magnitude = sqrt(dot(plane.xyz, plane.xyz));
  plane = plane / magnitude;

  vec4 firstPosition;
  bool triangleSplittingNecessary = false;

  for (int i = 0; i < 3; ++i)
  {
    vec4 inPos = triangle[i];
    bool isPosInFOV = dot(inPos, plane) > cutOffZ;
    if (isPosInFOV)
    {
      emittedVertexCount = emitWithPrimitiveHandling(
          matrix, inPos, triangleSplittingNecessary, emittedVertexCount);

      if (emittedVertexCount == 1)
        firstPosition = inPos;
    }

    vec4 nextPos = triangle[(i + 1) % 3];
    bool isNextPosInFOV = dot(nextPos, plane) > cutOffZ;
    if ((isPosInFOV && !isNextPosInFOV) || (!isPosInFOV && isNextPosInFOV))
    {
      triangleSplittingNecessary = true;
      vec4 edge = inPos - nextPos;
      float lambda = (cutOffZ - dot(plane, inPos)) / dot(plane, edge);

      vec4 newPos = inPos + lambda * edge;
      addCutPositionIfNew(newPos);
      emittedVertexCount = emitWithPrimitiveHandling(
          matrix, newPos, triangleSplittingNecessary, emittedVertexCount);

      if (emittedVertexCount == 1)
        firstPosition = newPos;
    }
  }

  if (emittedVertexCount == 5)
  {
    emitWithPrimitiveHandling(matrix, firstPosition, triangleSplittingNecessary,
                              emittedVertexCount);
  }

  EndPrimitive();
}

void processSide(const mat4 matrix, const vec4 center, const vec4 side,
                 const vec4 varying1, const vec4 varying2)
{
  vec4 triangle[3] = vec4[3](center + side - varying1 - varying2,
                             center + side - varying1 + varying2,
                             center + side + varying1 - varying2);
  processTriangle(matrix, triangle);

  triangle = vec4[3](center + side + varying1 - varying2,
                     center + side - varying1 + varying2,
                     center + side + varying1 + varying2);
  processTriangle(matrix, triangle);
}

bool hasSmallerAngle(const vec4 center, const vec4 pos1, const vec4 pos2)
{
  float angle1 = atan(pos1.y - center.y, pos1.x - center.x);
  float angle2 = atan(pos2.y - center.y, pos2.x - center.x);

  return angle1 < angle2;
}

void fillHole(const mat4 matrix)
{
  if (cutPositionCount == 3)
  {
    emit(matrix, cutPositions[0]);
    emit(matrix, cutPositions[1]);
    emit(matrix, cutPositions[2]);
    return;
  }

  // sort positions
  vec4 center = vec4(0, 0, 0, 0);
  for (int i = 0; i < cutPositionCount; ++i)
    center += cutPositions[i];
  center = center / cutPositionCount;

  for (int i = 0; i < cutPositionCount; ++i)
  {
    vec4 temp = cutPositions[i];
    int j = i - 1;
    while (hasSmallerAngle(matrix * center, matrix * temp,
                           matrix * cutPositions[j]) &&
           j >= 0)
    {
      cutPositions[j + 1] = cutPositions[j];
      j = j - 1;
    }
    cutPositions[j + 1] = temp;
  }

  cutPositions[cutPositionCount++] = cutPositions[0];
  for (int i = 0; i < cutPositionCount - 1; ++i)
  {
    emit(matrix, center);
    emit(matrix, cutPositions[i]);
    emit(matrix, cutPositions[i + 1]);
    EndPrimitive();
  }
}

void main()
{
  mat4 model = Transforms[vDrawId[0]];
  mat4 matrix = modelViewProjectionMatrix * model;
  const vec4 xAxis = vec4(0.5, 0, 0, 0);
  const vec4 yAxis = vec4(0, 0.5, 0, 0);
  const vec4 zAxis = vec4(0, 0, 0.5, 0);

  vec4 center = gl_in[0].gl_Position;

  // top
  processSide(matrix, center, yAxis, xAxis, zAxis);
  // right
  processSide(matrix, center, xAxis, yAxis, zAxis);
  // front
  processSide(matrix, center, zAxis, xAxis, yAxis);
  // bottom
  processSide(matrix, center, -yAxis, xAxis, zAxis);
  // left
  processSide(matrix, center, -xAxis, yAxis, zAxis);
  // back
  processSide(matrix, center, -zAxis, xAxis, yAxis);

  fillHole(matrix);
}

