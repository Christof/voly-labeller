#include "./constraint_updater.h"
#include <Eigen/Geometry>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/register/point.hpp>
#include <boost/polygon/polygon.hpp>
#include <vector>
#include "../graphics/vertex_array.h"
#include "../graphics/render_data.h"
BOOST_GEOMETRY_REGISTER_POINT_2D(Eigen::Vector2i, int, cs::cartesian, x(), y())
#include "./boost_polygon_concepts.h"

namespace bg = boost::geometry;

// ccw, open polygon
typedef bg::model::polygon<Eigen::Vector2i, false, false> polygon;

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

std::vector<boost::polygon::polygon_with_holes_data<int>>
minkowskiSum(const polygon &a, const polygon &b)
{
  boost::polygon::polygon_with_holes_data<int> aPoly;
  boost::polygon::set_points(aPoly, a.outer().begin(), a.outer().end());

  boost::polygon::polygon_with_holes_data<int> bPoly;
  boost::polygon::set_points(bPoly, b.outer().begin(), b.outer().end());

  boost::polygon::polygon_set_data<int> dilated;
  boost::polygon::polygon_set_data<int> aPolys;
  aPolys.insert(aPoly);
  boost::polygon::polygon_set_data<int> bPolys;
  bPolys.insert(bPoly);
  boost::polygon::detail::minkowski_offset<int>::convolve_two_polygon_sets(
      dilated, aPolys, bPolys);

  std::vector<boost::polygon::polygon_with_holes_data<int>> polys;
  dilated.get(polys);

  return polys;
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
  bg::convex_hull(oldLabelExtruded, oldLabelExtrudedConvexHull);

  int border = 2;
  polygon newLabel = createBoxPolygon(
      Eigen::Vector2i(0, 0), labelSize / 2 + Eigen::Vector2i(border, border));

  auto dilatedOldLabelExtruded =
      minkowskiSum(oldLabelExtrudedConvexHull, newLabel);
  for (auto &p : dilatedOldLabelExtruded)
  {
    std::vector<boost::polygon::point_data<int>> points(p.begin(), p.end());
    drawPolygon(points);
  }

  polygon connectorPolygon;
  connectorPolygon.outer().push_back(lastAnchorPosition);
  Eigen::Vector2i throughLastAnchor =
      anchorPosition + 1000 * (lastAnchorPosition - anchorPosition);
  connectorPolygon.outer().push_back(throughLastAnchor);

  Eigen::Vector2i throughLastLabel =
      anchorPosition + 1000 * (lastLabelPosition - anchorPosition);
  connectorPolygon.outer().push_back(throughLastLabel);
  connectorPolygon.outer().push_back(lastLabelPosition);

  auto dilatedConnector = minkowskiSum(connectorPolygon, newLabel);
  for (auto &p : dilatedConnector)
  {
    std::vector<boost::polygon::point_data<int>> pointsConnector(p.begin(),
                                                                 p.end());
    drawPolygon(pointsConnector);
  }
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
    positions.push_back(0.0f);
  }
  for (auto point : polygon)
  {
    positions.push_back(point.x());
    positions.push_back(height - point.y());
    positions.push_back(0.0f);
  }

  Graphics::VertexArray *vertexArray =
      new Graphics::VertexArray(gl, GL_TRIANGLE_FAN);
  vertexArray->addStream(positions);

  RenderData renderData;
  renderData.modelMatrix = Eigen::Matrix4f::Identity();
  renderData.viewMatrix = pixelToNDC;
  renderData.projectionMatrix = Eigen::Matrix4f::Identity();
  shaderManager->bind(shaderId, renderData);
  vertexArray->draw();
}

