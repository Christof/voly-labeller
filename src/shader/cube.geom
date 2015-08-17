/**
 * \brief Geometry shader which constructs a cube from each given point
 *
 * The constructed cube has a size of 1 x 1 x 1 and the center
 * is the input point. The cube is intersected with the near
 * plane, so that nothing is behind the near plane. Additional
 * geometry is generated to fill the created hole.
 */

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

/**
 * \brief Emits the vertex position defined by matrix * pos
 *
 * This consists of setting the vertexPos and vertexColor
 * as well as the gl_Position. The color is calculated
 * from the position so that it can be used as texture coordinate.
 */
void emit(const mat4 matrix, const vec4 pos)
{
  vertexPos = matrix * pos;
  gl_Position = vertexPos;
  vertexColor = pos + vec4(0.5, 0.5, 0.5, 0);
  EmitVertex();
}

/**
 * \brief Emits the given point, ends the primitive and starts a new one
 * with the given point if necessary
 */
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

/**
 * \brief Add given position to cutPositions if its not already present
 * in the collection
 */
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

/**
 * \brief Processes a triangle by performing intersection
 * with the given nearPlane
 *
 * After a call to this function one or two triangles have
 * been emitted (depending on whether the triangle intersects
 * the near plane). Also any calculated intersection positions
 * have been added to the cutPositions global.
 */
void processTriangle(const mat4 matrix, const vec4 nearPlane,
                     const vec4 triangle[3])
{
  const float cutOffZ = 0.5;
  int emittedVertexCount = 0;

  vec4 firstPosition;
  bool triangleSplittingNecessary = false;

  for (int i = 0; i < 3; ++i)
  {
    vec4 inPos = triangle[i];
    bool isPosInFOV = dot(inPos, nearPlane) > cutOffZ;
    if (isPosInFOV)
    {
      emittedVertexCount = emitWithPrimitiveHandling(
          matrix, inPos, triangleSplittingNecessary, emittedVertexCount);

      if (emittedVertexCount == 1)
        firstPosition = inPos;
    }

    vec4 nextPos = triangle[(i + 1) % 3];
    bool isNextPosInFOV = dot(nextPos, nearPlane) > cutOffZ;
    if ((isPosInFOV && !isNextPosInFOV) || (!isPosInFOV && isNextPosInFOV))
    {
      triangleSplittingNecessary = true;
      vec4 edge = inPos - nextPos;
      float lambda = (cutOffZ - dot(nearPlane, inPos)) / dot(nearPlane, edge);

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

/**
 * \brief Generates two triangles from the given data and processes them
 *
 * The side of the cube is calculated by generating a quad in the plane
 * which is determined by the side vector. The plane is has side as
 * normal and is 0.5 units away from the center point.
 */
void processSide(const mat4 matrix, const vec4 nearPlane, const vec4 center,
                 const vec4 side, const vec4 varying1, const vec4 varying2)
{
  vec4 triangle[3] = vec4[3](center + side - varying1 - varying2,
                             center + side - varying1 + varying2,
                             center + side + varying1 - varying2);
  processTriangle(matrix, nearPlane, triangle);

  triangle = vec4[3](center + side + varying1 - varying2,
                     center + side - varying1 + varying2,
                     center + side + varying1 + varying2);
  processTriangle(matrix, nearPlane, triangle);
}

/**
 * \brief Compare the two given positions by their angle in respect to the
 * given center point
 */
bool hasSmallerAngle(const vec4 center, const vec4 pos1, const vec4 pos2)
{
  float angle1 = atan(pos1.y - center.y, pos1.x - center.x);
  float angle2 = atan(pos2.y - center.y, pos2.x - center.x);

  return angle1 < angle2;
}

/**
 * \brief Generate triangles to fill the hole generated if the cube intersects
 * the near plane.
 */
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

  vec4 nearPlane = vec4(matrix[0][3] + matrix[0][2], matrix[1][3] + matrix[1][2],
                    matrix[2][3] + matrix[2][2], matrix[3][3] + matrix[3][2]);

  float magnitude = sqrt(dot(nearPlane.xyz, nearPlane.xyz));
  nearPlane = nearPlane / magnitude;
  vec4 center = gl_in[0].gl_Position;

  // top
  processSide(matrix, nearPlane, center, yAxis, xAxis, zAxis);
  // right
  processSide(matrix, nearPlane, center, xAxis, yAxis, zAxis);
  // front
  processSide(matrix, nearPlane, center, zAxis, xAxis, yAxis);
  // bottom
  processSide(matrix, nearPlane, center, -yAxis, xAxis, zAxis);
  // left
  processSide(matrix, nearPlane, center, -xAxis, yAxis, zAxis);
  // back
  processSide(matrix, nearPlane, center, -zAxis, xAxis, yAxis);

  fillHole(matrix);
}

