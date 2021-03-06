/**
 * \brief Geometry shader which constructs a dilated shadow
 *
 * The shadow is created by the `gl_Position` input as source and a line
 * between vConnectorStart and vConnectorEnd.
 *
 * The dilation is adapted from Hasselgren, J., Akenine-Möller, T.,
 * & Ohlsson, L. (2005). Conservative rasterization. GPU Gems, 2, 677–690.
 * (http://http.developer.nvidia.com/GPUGems2/gpugems2_chapter42.html)
 */

#version 440

#define THRESHOLD 0.0000125

layout(points) in;
layout(triangle_strip, max_vertices = 113) out;

in int vDrawId[];
in vec4 vConnectorStart[];
in vec4 vConnectorEnd[];
in vec4 vVertexColor[];

out vec4 vertexColor;

uniform vec2 halfSize = vec2(0.2, 0.05);

vec2 positions[18];
int positionsCount = 0;

void emitVertex(in vec2 position)
{
  gl_Position = vec4(position, 0, 1);
  EmitVertex();
}

void drawDilatedLine(in vec2 prevPos, in vec2 currentPos, in vec2 nextPos)
{
  vec3 plane[2];

  vec2 prevToCurrent = currentPos - prevPos;
  // cross(prevToCurrent.xyw, prevPos.xyw)
  plane[0] = vec3(prevToCurrent.y, -prevToCurrent.x,
                  prevToCurrent.x * prevPos.y - prevToCurrent.y * prevPos.x);

  vec2 currentToNext = nextPos - currentPos;
  // cross(currentToNext.xyw, currentPos.xyw)
  plane[1] =
      vec3(currentToNext.y, -currentToNext.x,
           currentToNext.x * currentPos.y - currentToNext.y * currentPos.x);

  // Compute the semi diagonals in the same quadrants as the plane normal. Note
  // that the use of the step function here is the same as sign(), but do not
  // return zero for an input value of zero (which sign() unfortunately does)
  vec2 semiDiagonal[2];
  semiDiagonal[0] = (step(vec2(0, 0), plane[0].xy) - 0.5) * 2.0;
  semiDiagonal[1] = (step(vec2(0, 0), plane[1].xy) - 0.5) * 2.0;

  vec2 finalPos = currentPos;

  float dp = dot(semiDiagonal[0], semiDiagonal[1]);

  vec2 diag;
  if (dp > THRESHOLD)
  {
    // The plane normals are in the same quadrant -> One vertex generated.
    positions[positionsCount++] = currentPos + semiDiagonal[0] * halfSize;
  }
  else if (dp >= -THRESHOLD)
  {
    // The plane normals are in neighboring quadrants -> Two vertices generated
    positions[positionsCount++] = currentPos + semiDiagonal[0] * halfSize;
    positions[positionsCount++] = currentPos + semiDiagonal[1] * halfSize;
  }
  else
  {
    // The plane normals are in opposite quadrants -> Three vertices generated
    positions[positionsCount++] = currentPos + semiDiagonal[0] * halfSize;
    vec2 diag = vec2(semiDiagonal[0].x * semiDiagonal[0].y * semiDiagonal[1].x,
                     semiDiagonal[0].y * semiDiagonal[1].x * semiDiagonal[1].y);
    positions[positionsCount++] = currentPos + diag * halfSize;
    positions[positionsCount++] = currentPos + semiDiagonal[1] * halfSize;
  }
}

/**
 * \brief Compare the two given positions by their angle in respect to the
 * given center point
 */
bool hasSmallerAngle(const vec2 center, const vec2 pos1, const vec2 pos2)
{
  float angle1 = atan(pos1.y - center.y, pos1.x - center.x);
  float angle2 = atan(pos2.y - center.y, pos2.x - center.x);

  return angle1 < angle2;
}

/**
 * \brief Calculate the center of all positions
 *
 */
vec2 calculateCenter()
{
  vec2 sum = vec2(0);
  for (int i = 0; i < positionsCount; ++i)
    sum += positions[i];

  return sum / positionsCount;
}

void drawPositions()
{
  const vec2 center = calculateCenter();

  positions[positionsCount++] = positions[0];
  for (int i = 0; i < positionsCount - 1; ++i)
  {
    emitVertex(center);
    emitVertex(positions[i]);
    emitVertex(positions[i + 1]);
    EndPrimitive();
  }
}

void drawConnectorConstraint(in vec2 anchor, in vec2 connectorStart,
                             in vec2 connectorEnd)
{
  const vec2 anchorToStart = normalize(connectorStart - anchor);
  const vec2 connectorStartShadow = connectorStart + 200.0f * anchorToStart;

  const vec2 anchorToEnd = normalize(connectorEnd - anchor);
  const vec2 connectorEndShadow = connectorEnd + 200.0f * anchorToEnd;

  // Triangle: connectorStart, connectorEnd, connectorStartShadow
  vec2 v = connectorEnd - connectorStart;
  float areaOfFirstTriangle = v.x * anchorToStart.y - v.y * anchorToStart.x;
  const bool isCCW = areaOfFirstTriangle > 0;
  if (isCCW)
  {
    drawDilatedLine(connectorStartShadow, connectorStart, connectorEnd);
    drawDilatedLine(connectorStart, connectorEnd, connectorEndShadow);
    drawDilatedLine(connectorEnd, connectorEndShadow, connectorStartShadow);
    drawDilatedLine(connectorEndShadow, connectorStartShadow, connectorStart);
    /*
    emitVertex(connectorStart);
    emitVertex(connectorEnd);
    emitVertex(connectorStartShadow);
    EndPrimitive();

    emitVertex(connectorStartShadow);
    emitVertex(connectorEnd);
    emitVertex(connectorEndShadow);
    EndPrimitive();
    */
  }
  else
  {
    drawDilatedLine(connectorEndShadow, connectorEnd, connectorStart);
    drawDilatedLine(connectorEnd, connectorStart, connectorStartShadow);
    drawDilatedLine(connectorStart, connectorStartShadow, connectorEndShadow);
    drawDilatedLine(connectorStartShadow, connectorEndShadow, connectorEnd);
    /*
    emitVertex(connectorStart);
    emitVertex(connectorStartShadow);
    emitVertex(connectorEnd);
    EndPrimitive();

    emitVertex(connectorEnd);
    emitVertex(connectorStartShadow);
    emitVertex(connectorEndShadow);
    EndPrimitive();
    */
  }

  drawPositions();
}

void main()
{
  int drawId = vDrawId[0];

  vertexColor = vVertexColor[0];
  drawConnectorConstraint(gl_in[0].gl_Position.xy, vConnectorEnd[0].xy,
                          vConnectorStart[0].xy);
}
