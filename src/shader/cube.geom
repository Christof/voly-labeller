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
layout(triangle_strip, max_vertices = 113) out;

in int vDrawId[];

out vec4 vertexPos;
out vec4 vertexEyePos;
out int volumeId;

uniform mat4 viewMatrix;
uniform mat4 viewProjectionMatrix;

layout(std140, binding = 1) buffer CB1
{
  vec4 physicalSize[];
};

#include "vertexHelper.hglsl"

vec4 cutPositions[18];
int cutPositionCount = 0;

/**
 * \brief Emits the vertex position defined by matrix * pos
 *
 * This consists of setting the vertexPos and as well as the gl_Position.
 */
void emit(const mat4 matrix, vec4 pos)
{
  vertexPos = matrix * pos;
  const vec4 size = physicalSize[vDrawId[0]];
  pos.xyz *= size.xyz;
  vertexEyePos = viewMatrix * getModelMatrix(vDrawId[0]) * pos;
  volumeId = floatBitsToInt(size.w);
  gl_Position = vertexPos;
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
  // use positive value to see the cutting in front of the near plane
  const float cutOffZ = 0.000001;
  int emittedVertexCount = 0;

  vec4 firstPosition;
  bool triangleSplittingNecessary = false;

  for (int i = 0; i < 3; ++i)
  {
    const vec4 inPos = triangle[i];
    const bool isPosInFOV = dot(inPos, nearPlane) > cutOffZ;
    if (isPosInFOV)
    {
      emittedVertexCount = emitWithPrimitiveHandling(
          matrix, inPos, triangleSplittingNecessary, emittedVertexCount);

      if (emittedVertexCount == 1)
        firstPosition = inPos;
    }

    const vec4 nextPos = triangle[(i + 1) % 3];
    const bool isNextPosInFOV = dot(nextPos, nearPlane) > cutOffZ;
    if ((isPosInFOV && !isNextPosInFOV) || (!isPosInFOV && isNextPosInFOV))
    {
      triangleSplittingNecessary = true;
      const vec4 edge = inPos - nextPos;
      const float lambda = (cutOffZ - dot(nearPlane, inPos)) / dot(nearPlane, edge);

      const vec4 newPos = inPos + lambda * edge;
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
 * which is determined by the side vector. The plane has side as
 * normal and is 0.5 units away from the center point.
 */
void processSide(const mat4 matrix, const vec4 nearPlane, const vec4 center,
                 const vec4 side, const vec4 varying1, const vec4 varying2)
{
  vec4 triangle[3] = vec4[3](center + side - varying1 - varying2,
      center + side + varying1 - varying2,
      center + side - varying1 + varying2);
  processTriangle(matrix, nearPlane, triangle);

  triangle = vec4[3](center + side + varying1 - varying2,
      center + side + varying1 + varying2,
      center + side - varying1 + varying2);
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
 * \brief Sort all positions generated from intersections by
 * their angle in respect to their center
 */
vec4 sortCutPositions(const mat4 matrix)
{
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

  return center;
}

/**
 * \brief Generate triangles to fill the hole generated if the cube intersects
 * the near plane.
 */
void fillHole(const mat4 matrix)
{
  if (cutPositionCount < 3)
    return;

  if (cutPositionCount == 3)
  {
    emit(matrix, cutPositions[0]);
    emit(matrix, cutPositions[1]);
    emit(matrix, cutPositions[2]);
    EndPrimitive();
    return;
  }

  const vec4 center = sortCutPositions(matrix);
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
  int drawId = vDrawId[0];
  vec4 size = physicalSize[drawId];
  mat4 model = getModelMatrix(drawId);
  mat4 scaleMatrix = mat4(size.x, 0, 0, 0,
                          0, size.y, 0, 0,
                          0, 0, size.z, 0,
                          0, 0, 0, 1);
  mat4 matrix = viewProjectionMatrix * model * scaleMatrix;
  const vec4 xAxis = vec4(0.5, 0, 0, 0);
  const vec4 yAxis = vec4(0, 0.5, 0, 0);
  const vec4 zAxis = vec4(0, 0, 0.5, 0);

  vec4 nearPlane = vec4(matrix[0][3] + matrix[0][2], matrix[1][3] + matrix[1][2],
                    matrix[2][3] + matrix[2][2], matrix[3][3] + matrix[3][2]);

  float magnitude = sqrt(dot(nearPlane.xyz, nearPlane.xyz));
  nearPlane = nearPlane / magnitude;
  vec4 center = gl_in[0].gl_Position;

  // top
  processSide(matrix, nearPlane, center, yAxis, -xAxis, zAxis);
  // right
  processSide(matrix, nearPlane, center, xAxis, yAxis, zAxis);
  // front
  processSide(matrix, nearPlane, center, zAxis, xAxis, yAxis);
  // bottom
  processSide(matrix, nearPlane, center, -yAxis, xAxis, zAxis);
  // left
  processSide(matrix, nearPlane, center, -xAxis, -yAxis, zAxis);
  // back
  processSide(matrix, nearPlane, center, -zAxis, -xAxis, yAxis);

  fillHole(matrix);
}

