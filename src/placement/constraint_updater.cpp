#include "./constraint_updater.h"
#include <QLoggingCategory>
#include <Eigen/Geometry>
#include <polyclipping/clipper.hpp>
#include <vector>
#include <chrono>
#include <cmath>
#include <utility>
#include <list>
#include "./placement.h"

QLoggingCategory cuChan("Placement.ConstraintUpdater");

ConstraintUpdater::ConstraintUpdater(std::shared_ptr<Graphics::Drawer> drawer,
                                     int width, int height)
  : drawer(drawer), width(width), height(height)
{
  labelShadowColor = Placement::labelShadowValue / 255.0f;
  connectorShadowColor = Placement::connectorShadowValue / 255.0f;
  anchorConstraintColor = Placement::anchorConstraintValue / 255.0f;
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

  drawAnchorRegion(anchorPosition, labelSize);
}

void ConstraintUpdater::clear()
{
  drawer->clear();
}

void ConstraintUpdater::setIsConnectorShadowEnabled(bool enabled)
{
  isConnectorShadowEnabled = enabled;
}

int squaredDistance(Eigen::Vector2i v)
{
  return v.x() * v.x() + v.y() * v.y();
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

  float smallestCosA = 1.0f;
  int smallestCosAIndex = -1;
  Eigen::Vector2i pointForSmallestCosA;
  float smallestCosB = 1.0f;
  int smallestCosBIndex = -1;
  Eigen::Vector2i pointForSmallestCosB;

  int index = 0;
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
      smallestCosAIndex = index;
    }
    else if (perpDot >= 0.0f && cosOfAngle < smallestCosB)
    {
      smallestCosB = cosOfAngle;
      pointForSmallestCosB = Eigen::Vector2i(point.X, point.Y);
      smallestCosBIndex = index;
    }

    ++index;
  }

  bool foundCloserPoint = false;
  Eigen::Vector2i closerPoint;
  int compareDistance = squaredDistance(pointForSmallestCosA - anchorPosition);
  for (int i = 0; i < 4; ++i)
  {
    if (i == smallestCosAIndex || i == smallestCosBIndex)
      continue;

    auto point = oldLabel[i];
    Eigen::Vector2i probe = Eigen::Vector2i(point.X, point.Y) - anchorPosition;

    if (squaredDistance(probe) < compareDistance)
    {
      foundCloserPoint = true;
      closerPoint = Eigen::Vector2i(point.X, point.Y);
      break;
    }
  }

  ClipperLib::Path oldLabelExtruded;
  oldLabelExtruded.push_back(toClipper(pointForSmallestCosA));
  oldLabelExtruded.push_back(toClipper(
      anchorPosition + 1000 * (pointForSmallestCosA - anchorPosition)));
  oldLabelExtruded.push_back(toClipper(
      anchorPosition + 1000 * (pointForSmallestCosB - anchorPosition)));
  oldLabelExtruded.push_back(toClipper(pointForSmallestCosB));

  if (foundCloserPoint)
  {
    oldLabelExtruded.push_back(toClipper(closerPoint));
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

void ConstraintUpdater::drawAnchorRegion(Eigen::Vector2i anchorPosition,
                                         Eigen::Vector2i labelSize)
{
  positions.clear();
  drawPolygon(createBoxPolygon(anchorPosition, 2 * labelSize));
  drawer->drawElementVector(positions, anchorConstraintColor);
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

