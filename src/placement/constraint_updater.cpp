#include "./constraint_updater.h"
#include <QLoggingCategory>
#include <Eigen/Geometry>
#include <vector>
#include <chrono>
#include <cmath>
#include <utility>
#include <list>
#include <polyclipping/clipper.hpp>
#include "./placement.h"

QLoggingCategory cuChan("Placement.ConstraintUpdater");

ConstraintUpdater::ConstraintUpdater(std::shared_ptr<Graphics::Drawer> drawer,
                                     int width, int height)
  : drawer(drawer), width(width), height(height)
{
  labelShadowColor = Placement::labelShadowValue / 255.0f;
  connectorShadowColor = Placement::connectorShadowValue / 255.0f;
}

ClipperLib::IntPoint toClipper(Eigen::Vector2i v)
{
  return ClipperLib::IntPoint(v.x(), v.y());
}

ClipperLib::Path createBoxPolygon(Eigen::Vector2i center, Eigen::Vector2i size)
{
  ClipperLib::Path p;
  p.push_back(
      toClipper(Eigen::Vector2i(center.x() - size.x(), center.y() - size.y())));
  p.push_back(
      toClipper(Eigen::Vector2i(center.x() + size.x(), center.y() - size.y())));
  p.push_back(
      toClipper(Eigen::Vector2i(center.x() + size.x(), center.y() + size.y())));
  p.push_back(
      toClipper(Eigen::Vector2i(center.x() - size.x(), center.y() + size.y())));

  return p;
}

void ConstraintUpdater::drawConstraintRegionFor(
    Eigen::Vector2i anchorPosition, Eigen::Vector2i labelSize,
    Eigen::Vector2i lastAnchorPosition, Eigen::Vector2i lastLabelPosition,
    Eigen::Vector2i lastLabelSize)
{
  auto startTime = std::chrono::high_resolution_clock::now();

  int border = 2;
  ClipperLib::Path newLabel = createBoxPolygon(
      Eigen::Vector2i(0, 0), labelSize / 2 + Eigen::Vector2i(border, border));

  positions.clear();

  drawConnectorShadowRegion(anchorPosition, lastAnchorPosition,
                            lastLabelPosition, newLabel);

  drawLabelShadowRegion(anchorPosition, lastLabelPosition, lastLabelSize,
                        newLabel);

  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> diff = endTime - startTime;

  qCDebug(cuChan) << "drawConstraintRegionFor without drawing took"
                  << diff.count() << "ms";

  drawer->drawElementVector(positions, labelShadowColor);
}

void ConstraintUpdater::clear()
{
  drawer->clear();
}

void ConstraintUpdater::setIsConnectorShadowEnabled(bool enabled)
{
  isConnectorShadowEnabled = enabled;
}

void ConstraintUpdater::drawLabelShadowRegion(Eigen::Vector2i anchorPosition,
                                              Eigen::Vector2i lastLabelPosition,
                                              Eigen::Vector2i lastLabelSize,
                                              const ClipperLib::Path &newLabel)
{
  ClipperLib::Path oldLabel =
      createBoxPolygon(lastLabelPosition, lastLabelSize / 2);

  ClipperLib::Path oldLabelExtruded(oldLabel);
  for (auto point : oldLabel)
  {
    oldLabelExtruded.push_back(ClipperLib::IntPoint(
        anchorPosition.x() + 1000 * (point.X - anchorPosition.x()),
        anchorPosition.y() + 1000 * (point.Y - anchorPosition.y())));
  }

  ClipperLib::Paths shadow;
  ClipperLib::MinkowskiSum(newLabel, oldLabelExtruded, shadow, false);

  ClipperLib::Clipper clipper;
  clipper.AddPaths(shadow, ClipperLib::PolyType::ptSubject, true);

  ClipperLib::Paths solution;
  clipper.Execute(ClipperLib::ClipType::ctUnion, solution,
                  ClipperLib::PolyFillType::pftNonZero);
  for (auto &polygon : solution)
    drawPolygon(polygon);
}

void ConstraintUpdater::drawConnectorShadowRegion(
    Eigen::Vector2i anchorPosition, Eigen::Vector2i lastAnchorPosition,
    Eigen::Vector2i lastLabelPosition, const ClipperLib::Path &newLabel)
{
  if (!isConnectorShadowEnabled)
    return;

  ClipperLib::Path connectorPolygon;
  Eigen::Vector2i throughLastAnchor =
      anchorPosition + 1000 * (lastAnchorPosition - anchorPosition);
  Eigen::Vector2i throughLastLabel =
      anchorPosition + 1000 * (lastLabelPosition - anchorPosition);

  connectorPolygon.push_back(toClipper(lastAnchorPosition));
  connectorPolygon.push_back(toClipper(throughLastAnchor));
  connectorPolygon.push_back(toClipper(throughLastLabel));
  connectorPolygon.push_back(toClipper(lastLabelPosition));

  ClipperLib::Paths shadow;
  ClipperLib::MinkowskiSum(newLabel, connectorPolygon, shadow, true);

  ClipperLib::Clipper clipper;
  clipper.AddPaths(shadow, ClipperLib::PolyType::ptSubject, true);

  ClipperLib::Paths solution;
  clipper.Execute(ClipperLib::ClipType::ctUnion, solution,
                  ClipperLib::PolyFillType::pftNonZero);

  for (auto &polygon : solution)
    drawPolygon(polygon);
  drawer->drawElementVector(positions, connectorShadowColor);

  positions.clear();
}

void ConstraintUpdater::drawPolygon(ClipperLib::Path polygon)
{
  if (polygon.size() > 0)
  {
    auto point = polygon[0];
    positions.push_back(point.X);
    positions.push_back(height - point.Y);
  }
  for (auto point : polygon)
  {
    positions.push_back(point.X);
    positions.push_back(height - point.Y);
  }
}

