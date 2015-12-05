#include "./constraint_updater.h"
#include <Eigen/Geometry>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/register/point.hpp>
#include <vector>
#include <cmath>
#include <utility>
#include <list>
#include "../graphics/vertex_array.h"
#include "../graphics/render_data.h"
BOOST_GEOMETRY_REGISTER_POINT_2D(Eigen::Vector2i, int, cs::cartesian, x(), y())
#include "./boost_polygon_concepts.h"

ConstraintUpdater::ConstraintUpdater(
    Graphics::Gl *gl, std::shared_ptr<Graphics::ShaderManager> shaderManager,
    int width, int height)
  : gl(gl), shaderManager(shaderManager), width(width), height(height)
{
  shaderId = shaderManager->addShader(":/shader/constraint.vert",
                                      ":/shader/colorImmediate.frag");

  Eigen::Affine3f pixelToNDCTransform(
      Eigen::Translation3f(Eigen::Vector3f(-1, 1, 0)) *
      Eigen::Scaling(Eigen::Vector3f(2.0f / width, -2.0f / height, 1)));
  pixelToNDC = pixelToNDCTransform.matrix();
}

polygon createBoxPolygon(Eigen::Vector2i center, Eigen::Vector2i size)
{
  polygon p;
  p.outer().push_back(
      Eigen::Vector2i(center.x() - size.x(), center.y() - size.y()));
  p.outer().push_back(
      Eigen::Vector2i(center.x() + size.x(), center.y() - size.y()));
  p.outer().push_back(
      Eigen::Vector2i(center.x() + size.x(), center.y() + size.y()));
  p.outer().push_back(
      Eigen::Vector2i(center.x() - size.x(), center.y() + size.y()));

  return p;
}

template <typename edge>
void ConstraintUpdater::convolveTwoSegements(polygon &polygon, const edge &a,
                                             const edge &b)
{
  auto p = a.first;
  boost::polygon::convolve(p, b.second);
  polygon.outer().push_back(Eigen::Vector2i(p.x(), p.y()));

  p = a.first;
  boost::polygon::convolve(p, b.first);
  polygon.outer().push_back(Eigen::Vector2i(p.x(), p.y()));

  p = a.second;
  boost::polygon::convolve(p, b.first);
  polygon.outer().push_back(Eigen::Vector2i(p.x(), p.y()));

  p = a.second;
  boost::polygon::convolve(p, b.second);
  polygon.outer().push_back(Eigen::Vector2i(p.x(), p.y()));
}

template <typename itrT1, typename itrT2>
void ConstraintUpdater::convolveTwoPointSequences(itrT1 ab, itrT1 ae, itrT2 bb,
                                                  itrT2 be)
{
  if (ab == ae || bb == be)
    return;

  polygon poly;
  auto prev_a = *ab;
  ++ab;
  for (; ab != ae; ++ab)
  {
    auto prev_b = *bb;
    itrT2 tmpb = bb;
    ++tmpb;
    for (; tmpb != be; ++tmpb)
    {
      convolveTwoSegements(poly, std::make_pair(prev_b, *tmpb),
                           std::make_pair(prev_a, *ab));
      prev_b = *tmpb;
    }
    prev_a = *ab;
  }

  polygon convexHull;
  boost::geometry::convex_hull(poly, convexHull);
  drawPolygon(convexHull.outer());
}

template <typename Polygon>
void ConstraintUpdater::convolveTwoPolygons(const Polygon &a, const Polygon &b)
{
  convolveTwoPointSequences(
      boost::polygon::begin_points(a), boost::polygon::end_points(a),
      boost::polygon::begin_points(b), boost::polygon::end_points(b));

  Polygon tmp_poly = a;
  addPolygonToPositions(
      boost::polygon::convolve(tmp_poly, *(boost::polygon::begin_points(b))));
  tmp_poly = b;
  addPolygonToPositions(
      boost::polygon::convolve(tmp_poly, *(boost::polygon::begin_points(a))));
}

bool hasSmallerAngle(float dir1X, float dir1Y, float dir2X, float dir2Y)
{
  float angle1 = std::atan2(dir1Y, dir1X);
  float angle2 = std::atan2(dir2Y, dir2X);

  return angle1 <= angle2;
}

template <typename Polygon>
void ConstraintUpdater::addPolygonToPositions(const Polygon &polygon)
{
  auto iteratorBegin = boost::polygon::begin_points(polygon);
  float referenceX = iteratorBegin->x();
  positions.push_back(referenceX);
  float referenceY = iteratorBegin->y();
  positions.push_back(height - referenceY);
  positions.push_back(referenceX);
  positions.push_back(height - referenceY);

  std::list<std::pair<float, float>> temp;
  for (auto iterator = ++iteratorBegin;
       iterator != boost::polygon::end_points(polygon); ++iterator)
  {
    float diffX = iterator->x() - referenceX;
    float diffY = iterator->y() - referenceY;
    auto inner = temp.begin();
    while (inner != temp.end() &&
           hasSmallerAngle(inner->first - referenceX,
                           inner->second - referenceY, diffX, diffY))
      ++inner;

    temp.insert(inner, 1, std::make_pair(iterator->x(), iterator->y()));
  }

  for (auto iterator = temp.cbegin(); iterator != temp.cend(); ++iterator)
  {
    positions.push_back(iterator->first);
    positions.push_back(height - iterator->second);
  }
}

void ConstraintUpdater::addLabel(Eigen::Vector2i anchorPosition,
                                 Eigen::Vector2i labelSize,
                                 Eigen::Vector2i lastAnchorPosition,
                                 Eigen::Vector2i lastLabelPosition,
                                 Eigen::Vector2i lastLabelSize)
{
  polygon oldLabel = createBoxPolygon(lastLabelPosition, lastLabelSize / 2);

  polygon oldLabelExtruded(oldLabel);
  for (auto point : oldLabel.outer())
  {
    Eigen::Vector2i p = anchorPosition + 1000 * (point - anchorPosition);
    oldLabelExtruded.outer().push_back(p);
  }

  polygon oldLabelExtrudedConvexHull;
  boost::geometry::convex_hull(oldLabelExtruded, oldLabelExtrudedConvexHull);

  int border = 2;
  polygon newLabel = createBoxPolygon(
      Eigen::Vector2i(0, 0), labelSize / 2 + Eigen::Vector2i(border, border));

  positions.clear();
  convolveTwoPolygons(oldLabelExtrudedConvexHull, newLabel);

  polygon connectorPolygon;
  Eigen::Vector2i throughLastAnchor =
      anchorPosition + 1000 * (lastAnchorPosition - anchorPosition);
  Eigen::Vector2i throughLastLabel =
      anchorPosition + 1000 * (lastLabelPosition - anchorPosition);

  connectorPolygon.outer().push_back(lastAnchorPosition);
  connectorPolygon.outer().push_back(throughLastAnchor);
  connectorPolygon.outer().push_back(throughLastLabel);
  connectorPolygon.outer().push_back(lastLabelPosition);

  convolveTwoPolygons(connectorPolygon, newLabel);

  drawElementVector(positions);
}

void ConstraintUpdater::clear()
{
  gl->glViewport(0, 0, width, height);
  gl->glClearColor(0, 0, 0, 0);
  gl->glClear(GL_COLOR_BUFFER_BIT);
}

template <class T> void ConstraintUpdater::drawPolygon(std::vector<T> polygon)
{
  std::vector<float> positions;
  if (polygon.size() > 0)
  {
    auto point = polygon[0];
    positions.push_back(point.x());
    positions.push_back(height - point.y());
  }
  for (auto point : polygon)
  {
    positions.push_back(point.x());
    positions.push_back(height - point.y());
  }

  drawElementVector(positions);
}

void ConstraintUpdater::drawElementVector(std::vector<float> positions)
{
  Graphics::VertexArray *vertexArray =
      new Graphics::VertexArray(gl, GL_TRIANGLE_FAN, 2);
  vertexArray->addStream(positions, 2);

  RenderData renderData;
  renderData.modelMatrix = Eigen::Matrix4f::Identity();
  renderData.viewMatrix = pixelToNDC;
  renderData.projectionMatrix = Eigen::Matrix4f::Identity();
  shaderManager->bind(shaderId, renderData);
  vertexArray->draw();
}

