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

typedef boost::polygon::point_data<int> point;
typedef boost::polygon::polygon_set_data<int> polygon_set;
typedef boost::polygon::polygon_with_holes_data<int> ppolygon;
typedef std::pair<point, point> edge;

static void convolve_two_segments(std::vector<point> &figure, const edge &a,
                                  const edge &b)
{
  figure.clear();
  figure.push_back(point(a.first));
  figure.push_back(point(a.first));
  figure.push_back(point(a.second));
  figure.push_back(point(a.second));
  convolve(figure[0], b.second);
  convolve(figure[1], b.first);
  convolve(figure[2], b.first);
  convolve(figure[3], b.second);
}

template <typename itrT1, typename itrT2>
static void convolve_two_point_sequences(polygon_set &result, itrT1 ab,
                                         itrT1 ae, itrT2 bb, itrT2 be)
{
  if (ab == ae || bb == be)
    return;
  point prev_a = *ab;
  std::vector<point> vec;
  ppolygon poly;
  ++ab;
  for (; ab != ae; ++ab)
  {
    point prev_b = *bb;
    itrT2 tmpb = bb;
    ++tmpb;
    for (; tmpb != be; ++tmpb)
    {
      convolve_two_segments(vec, std::make_pair(prev_b, *tmpb),
                            std::make_pair(prev_a, *ab));
      set_points(poly, vec.begin(), vec.end());
      result.insert(poly);
      prev_b = *tmpb;
    }
    prev_a = *ab;
  }
}

template <typename itrT>
static void convolve_point_sequence_with_polygon(polygon_set &result, itrT b,
                                                 itrT e,
                                                 const ppolygon &polygon)
{
  convolve_two_point_sequences(result, b, e, begin_points(polygon),
                               end_points(polygon));
}

static void convolve_two_polygons(polygon_set &result, const ppolygon &a,
                                  const ppolygon &b)
{
  result.clear();
  convolve_point_sequence_with_polygon(result, begin_points(a), end_points(a),
                                       b);
  ppolygon tmp_poly = a;
  result.insert(convolve(tmp_poly, *(begin_points(b))));
  tmp_poly = b;
  result.insert(convolve(tmp_poly, *(begin_points(a))));
}

std::vector<boost::polygon::polygon_with_holes_data<int>>
minkowskiSum(const polygon &a, const polygon &b)
{
  boost::polygon::polygon_with_holes_data<int> aPoly;
  boost::polygon::set_points(aPoly, a.outer().begin(), a.outer().end());

  boost::polygon::polygon_with_holes_data<int> bPoly;
  boost::polygon::set_points(bPoly, b.outer().begin(), b.outer().end());

  boost::polygon::polygon_set_data<int> dilated;
  convolve_two_polygons(dilated, aPoly, bPoly);

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

