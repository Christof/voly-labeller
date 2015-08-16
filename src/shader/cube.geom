#version 440

layout(points) in;
layout(triangle_strip, max_vertices = 85) out;

in vec4 vPos[];
in vec4 vColor[];
in mat4 matrix[];
in vec3 vTexCoord[];
in int vDrawId[];

out vec4 vertexPos;
out vec4 vertexColor;

uniform mat4 modelViewProjectionMatrix;

layout(std140, binding = 0) buffer CB0
{
  mat4 Transforms[];
};

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

int emit(vec4 pos, vec4 color, int emittedVertexCount)
{
  if (isFirst)
  {
    firstPosition = pos;
    isFirst = false;
  }

  justEmit(pos, color);
  ++emittedVertexCount;

  if (emittedVertexCount == 3)
  {
    EndPrimitive();
    if (hasTwoTriangles)
    {
      justEmit(pos, color);
      ++emittedVertexCount;
    }
  }

  return emittedVertexCount;
}

void addCutPositionIfNew(vec4 newPos)
{
  for (int i = 0; i < cutPositionCount; ++i)
  {
    vec4 diff = cutPositions[i] - newPos;
    if (dot(diff, diff) < 0.0001)
      return;
  }

  cutPositions[cutPositionCount++] = newPos;
}

void processTriangle(mat4 matrix, vec4 triangle[3])
{
  const float cutOffZ = 0.5;
  int emittedVertexCount = 0;
  isFirst = true;
  hasTwoTriangles = false;
  vec4 plane = vec4(matrix[0][3] + matrix[0][2], matrix[1][3] + matrix[1][2],
                    matrix[2][3] + matrix[2][2], matrix[3][3] + matrix[3][2]);

  float magnitude = sqrt(dot(plane.xyz, plane.xyz));
  plane = plane / magnitude;

  vec4 c = triangle[0] + triangle[1] + triangle[2];
  c = c * 0.25 + vec4(0.8, 0.8, 0.8, 0);
  c.a = 1;
  for (int i = 0; i < 3; ++i)
  {
    vec4 inPos = triangle[i];
    bool isPosInFOV = dot(inPos, plane) > cutOffZ;
    if (isPosInFOV)
    {
      emittedVertexCount = emit(matrix * inPos, c, emittedVertexCount);  // vColor[0]);
    }
    vec4 nextPos = triangle[(i + 1) % 3];
    bool isNextPosInFOV = dot(nextPos, plane) > cutOffZ;
    if ((isPosInFOV && !isNextPosInFOV) ||
        (!isPosInFOV && isNextPosInFOV))
    {
      hasTwoTriangles = true;
      vec4 edge = inPos - nextPos;
      float lambda = (cutOffZ - dot(plane, inPos)) / dot(plane, edge);

      vec4 newPos = inPos + lambda * edge;
      addCutPositionIfNew(newPos);
      //vec4 c = newPos * 0.5f + vec4(0.5, 0.5f, 0.5f, 0);
      // c.a = 0.5f;
      emittedVertexCount = emit(matrix * newPos, c, emittedVertexCount);  // vec4(1, 1, 0, 1));
    }
  }

  if (emittedVertexCount == 5)
  {
    emit(firstPosition, c, emittedVertexCount);
  }

  EndPrimitive();
}

void processSide(mat4 matrix, vec4 center, vec4 side, vec4 varying1,
                 vec4 varying2)
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

bool hasSmallerAngle(vec4 center, vec4 pos1, vec4 pos2)
{
  float angle1 = atan(pos1.y - center.y, pos1.x - center.x);
  float angle2 = atan(pos2.y - center.y, pos2.x - center.x);

  return angle1 < angle2;
}

void fillHole(mat4 matrix)
{
  if (cutPositionCount == 3)
  {
    justEmit(matrix * cutPositions[0], vec4(1, 0, 0, 0.8));
    justEmit(matrix * cutPositions[1], vec4(1, 0, 0, 0.8));
    justEmit(matrix * cutPositions[2], vec4(1, 0, 0, 0.8));
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
    while(hasSmallerAngle(matrix * center, matrix * temp, matrix * cutPositions[j]) && j >= 0)
    {
      cutPositions[j+1] = cutPositions[j];
      j = j - 1;
    }
    cutPositions[j + 1] = temp;
  }

  cutPositions[cutPositionCount++] = cutPositions[0];
  for (int i = 0; i < cutPositionCount - 1; ++i)
  {
    justEmit(matrix * center, vec4(0, 0, 1, 0.8));
    justEmit(matrix * cutPositions[i], vec4(1, 0, 0, 0.8));
    justEmit(matrix * cutPositions[i + 1], vec4(0, 1, 0, 0.8));
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

