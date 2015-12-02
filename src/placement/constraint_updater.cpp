#include "./constraint_updater.h"
#include <Eigen/Geometry>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/register/point.hpp>
#include <vector>
#include <utility>
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

typedef boost::polygon::point_data<int> point;
typedef boost::polygon::polygon_set_data<int> polygon_set;
typedef boost::polygon::polygon_with_holes_data<int> ppolygon;
typedef std::pair<point, point> edge;

void ConstraintUpdater::convolveTwoSegements(const edge &a, const edge &b)
{
  point p = a.first;
  convolve(p, b.second);
  positions.push_back(p.x());
  positions.push_back(height - p.y());
  positions.push_back(p.x());
  positions.push_back(height - p.y());

  p = a.first;
  convolve(p, b.first);
  positions.push_back(p.x());
  positions.push_back(height - p.y());

  p = a.second;
  convolve(p, b.first);
  positions.push_back(p.x());
  positions.push_back(height - p.y());

  p = a.second;
  convolve(p, b.second);
  positions.push_back(p.x());
  positions.push_back(height - p.y());
}

template <typename itrT1, typename itrT2>
void ConstraintUpdater::convolveTwoPointSequences(itrT1 ab, itrT1 ae, itrT2 bb,
                                                  itrT2 be)
{
  if (ab == ae || bb == be)
    return;
  point prev_a = *ab;
  ++ab;
  for (; ab != ae; ++ab)
  {
    point prev_b = *bb;
    itrT2 tmpb = bb;
    ++tmpb;
    for (; tmpb != be; ++tmpb)
    {
      convolveTwoSegements(std::make_pair(prev_b, *tmpb),
                           std::make_pair(prev_a, *ab));
      prev_b = *tmpb;
    }
    prev_a = *ab;
  }
}

void ConstraintUpdater::convolveTwoPolygons(const ppolygon &a,
                                            const ppolygon &b)
{
  convolveTwoPointSequences(begin_points(a), end_points(a), begin_points(b),
                            end_points(b));

  ppolygon tmp_poly = a;
  addPolygonToPositions(convolve(tmp_poly, *(begin_points(b))));
  tmp_poly = b;
  addPolygonToPositions(convolve(tmp_poly, *(begin_points(a))));
}

void ConstraintUpdater::addPolygonToPositions(const ppolygon &polygon)
{
  std::vector<boost::polygon::point_data<int>> p(polygon.begin(), polygon.end());
  auto iteratorBegin = p.begin();
  positions.push_back(iteratorBegin->x());
  positions.push_back(height - iteratorBegin->y());
  for (auto iterator = iteratorBegin; iterator != p.end(); ++iterator)
  {
    positions.push_back(iterator->x());
    positions.push_back(height - iterator->y());
  }
}

void ConstraintUpdater::minkowskiSum(const polygon &a, const polygon &b)
{
  boost::polygon::polygon_with_holes_data<int> aPoly;
  boost::polygon::set_points(aPoly, a.outer().begin(), a.outer().end());

  boost::polygon::polygon_with_holes_data<int> bPoly;
  boost::polygon::set_points(bPoly, b.outer().begin(), b.outer().end());

  convolveTwoPolygons(aPoly, bPoly);
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
  minkowskiSum(oldLabelExtrudedConvexHull, newLabel);

  polygon connectorPolygon;
  connectorPolygon.outer().push_back(lastAnchorPosition);
  Eigen::Vector2i throughLastAnchor =
      anchorPosition + 1000 * (lastAnchorPosition - anchorPosition);
  connectorPolygon.outer().push_back(throughLastAnchor);

  Eigen::Vector2i throughLastLabel =
      anchorPosition + 1000 * (lastLabelPosition - anchorPosition);
  connectorPolygon.outer().push_back(throughLastLabel);
  connectorPolygon.outer().push_back(lastLabelPosition);

  minkowskiSum(connectorPolygon, newLabel);

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

