#include "./constraint_updater.h"
#include <Eigen/Geometry>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/register/point.hpp>
#include <vector>
#include <cmath>
#include <utility>
#include <list>
BOOST_GEOMETRY_REGISTER_POINT_2D(Eigen::Vector2i, int, cs::cartesian, x(), y())
#include "./boost_polygon_concepts.h"

ConstraintUpdater::ConstraintUpdater(std::shared_ptr<Graphics::Drawer> drawer,
                                     int width, int height)
  : drawer(drawer), width(width), height(height)
{
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

void ConstraintUpdater::drawConstraintRegionFor(
    Eigen::Vector2i anchorPosition, Eigen::Vector2i labelSize,
    Eigen::Vector2i lastAnchorPosition, Eigen::Vector2i lastLabelPosition,
    Eigen::Vector2i lastLabelSize)
{
  int border = 2;
  polygon newLabel = createBoxPolygon(
      Eigen::Vector2i(0, 0), labelSize / 2 + Eigen::Vector2i(border, border));

  positions.clear();

  drawLabelShadowRegion(anchorPosition, lastLabelPosition, lastLabelSize,
                        newLabel);
  drawConnectorShadowRegion(anchorPosition, lastAnchorPosition,
                            lastLabelPosition, newLabel);

  drawer->drawElementVector(positions);
}

void ConstraintUpdater::clear()
{
  drawer->clear();
}

void ConstraintUpdater::useConnectorShadowRegion(bool enable)
{
  isConnectorShadowRegionEnabled = enable;
}

void ConstraintUpdater::drawLabelShadowRegion(Eigen::Vector2i anchorPosition,
                                              Eigen::Vector2i lastLabelPosition,
                                              Eigen::Vector2i lastLabelSize,
                                              const polygon &newLabel)
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

  convolveTwoPolygons(oldLabelExtrudedConvexHull, newLabel);
}

void ConstraintUpdater::drawConnectorShadowRegion(
    Eigen::Vector2i anchorPosition, Eigen::Vector2i lastAnchorPosition,
    Eigen::Vector2i lastLabelPosition, const polygon &newLabel)
{
  if (!isConnectorShadowRegionEnabled)
    return;

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

  drawer->drawElementVector(positions);
}

