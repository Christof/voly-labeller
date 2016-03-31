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

  Eigen::Vector2i anchorToOldLabelInteger = lastLabelPosition - anchorPosition;
  Eigen::Vector2f anchorToOldLabel =
      anchorToOldLabelInteger.cast<float>().normalized();

  ClipperLib::Path oldLabelExtruded(oldLabel);
  float smallestCosA = 1.0f;
  Eigen::Vector2i pointForSmallestCosA;
  float smallestCosB = 1.0f;
  Eigen::Vector2i pointForSmallestCosB;
  for (auto point : oldLabel)
  {
    Eigen::Vector2f probe =
        Eigen::Vector2f(point.X, point.Y) - anchorPosition.cast<float>();
    probe.normalize();

    float cosOfAngle = anchorToOldLabel.dot(probe);
    float perpDot =
        -anchorToOldLabel.y() * probe.x() + anchorToOldLabel.x() * probe.y();
    if (perpDot < 0.0f && cosOfAngle < smallestCosA)
    {
      smallestCosA = cosOfAngle;
      pointForSmallestCosA = Eigen::Vector2i(point.X, point.Y);
    }
    else if (perpDot >= 0.0f && cosOfAngle < smallestCosB)
    {
      smallestCosB = cosOfAngle;
      pointForSmallestCosB = Eigen::Vector2i(point.X, point.Y);
    }
  }

  ClipperLib::Path testPath;
  testPath.push_back(toClipper(anchorPosition));
  testPath.push_back(toClipper(pointForSmallestCosA));
  testPath.push_back(toClipper(pointForSmallestCosB));

  drawPolygon(testPath);
  /*
  ClipperLib::Paths shadow;
  ClipperLib::MinkowskiSum(newLabel, oldLabelExtruded, shadow, false);

  ClipperLib::Clipper clipper;
  clipper.AddPaths(shadow, ClipperLib::PolyType::ptSubject, true);

  ClipperLib::Paths solution;
  clipper.Execute(ClipperLib::ClipType::ctUnion, solution,
                  ClipperLib::PolyFillType::pftNonZero);
  for (auto &polygon : solution)
    drawPolygon(polygon);
    */
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

