#include "./constraint_updater.h"
#include <Eigen/Geometry>
#include <vector>
#include <string>
#include <memory>
#include "./placement.h"
#include "./shadow_constraint_drawer.h"
#include "./anchor_constraint_drawer.h"

ConstraintUpdater::ConstraintUpdater(
    int bufferWidth, int bufferHeight,
    std::shared_ptr<AnchorConstraintDrawer> anchorConstraintDrawer,
    std::shared_ptr<ShadowConstraintDrawer> connectorShadowDrawer,
    std::shared_ptr<ShadowConstraintDrawer> shadowConstraintDrawer,
    float scaleFactor)
  : width(bufferWidth), height(bufferHeight),
    anchorConstraintDrawer(anchorConstraintDrawer),
    connectorShadowDrawer(connectorShadowDrawer),
    shadowConstraintDrawer(shadowConstraintDrawer), scaleFactor(scaleFactor)
{
  labelShadowColor = Placement::labelShadowValue / 255.0f;
  connectorShadowColor = Placement::connectorShadowValue / 255.0f;
  anchorConstraintColor = Placement::anchorConstraintValue / 255.0f;

  borderPixel = scaleFactor * Eigen::Vector2f(4, 4);
}

ConstraintUpdater::~ConstraintUpdater()
{
}

void ConstraintUpdater::drawConstraintRegionFor(
    Eigen::Vector2i anchorPosition, Eigen::Vector2i labelSize,
    Eigen::Vector2i lastAnchorPosition, Eigen::Vector2i lastLabelPosition,
    Eigen::Vector2i lastLabelSize)
{
  this->labelSize = labelSize.cast<float>();

  if (isConnectorShadowEnabled)
    addConnectorShadow(anchorPosition, lastAnchorPosition, lastLabelPosition);

  Eigen::Vector2f anchor = anchorPosition.cast<float>();
  Eigen::Vector2f lastHalfSize = 0.5f * lastLabelSize.cast<float>();
  addLabelShadow(anchor, lastLabelPosition, lastHalfSize);
}

void ConstraintUpdater::drawRegionsForAnchors(
    std::vector<Eigen::Vector2f> anchorPositions, Eigen::Vector2i labelSize)
{
  std::vector<float> positions(anchorPositions.size() * 2);
  size_t index = 0;
  for (auto &anchorPosition : anchorPositions)
  {
    positions[index++] = anchorPosition.x();
    positions[index++] = anchorPosition.y();
  }
  assert(positions.size() == index);

  anchorConstraintDrawer->update(positions);

  Eigen::Vector2f constraintSize = borderPixel + labelSize.cast<float>();
  Eigen::Vector2f halfSize =
      constraintSize.cwiseQuotient(Eigen::Vector2f(width, height));

  anchorConstraintDrawer->draw(anchorConstraintColor, halfSize);
}

void ConstraintUpdater::clear()
{
  shadowConstraintDrawer->clear();

  sources.clear();
  starts.clear();
  ends.clear();

  anchors.clear();
  connectorStart.clear();
  connectorEnd.clear();
}

void ConstraintUpdater::finish()
{
  shadowConstraintDrawer->update(sources, starts, ends);
  connectorShadowDrawer->update(anchors, connectorStart, connectorEnd);

  Eigen::Vector2f sizeWithBorder = labelSize.cast<float>() + borderPixel;
  Eigen::Vector2f halfSize =
      sizeWithBorder.cwiseQuotient(Eigen::Vector2f(width, height));

  shadowConstraintDrawer->draw(labelShadowColor, halfSize);
  connectorShadowDrawer->draw(connectorShadowColor, halfSize);
}

void ConstraintUpdater::setIsConnectorShadowEnabled(bool enabled)
{
  isConnectorShadowEnabled = enabled;
}

void ConstraintUpdater::addConnectorShadow(Eigen::Vector2i anchor,
                                           Eigen::Vector2i start,
                                           Eigen::Vector2i end)
{
  anchors.push_back(anchor.x());
  anchors.push_back(anchor.y());

  connectorStart.push_back(start.x());
  connectorStart.push_back(start.y());

  connectorEnd.push_back(end.x());
  connectorEnd.push_back(end.y());
}

void ConstraintUpdater::addLineShadow(Eigen::Vector2f source,
                                      Eigen::Vector2f start,
                                      Eigen::Vector2f end)
{
  sources.push_back(source.x());
  sources.push_back(source.y());

  starts.push_back(start.x());
  starts.push_back(start.y());

  ends.push_back(end.x());
  ends.push_back(end.y());
}

void ConstraintUpdater::addLabelShadow(Eigen::Vector2f anchor,
                                       Eigen::Vector2i lastLabelPosition,
                                       Eigen::Vector2f lastHalfSize)
{
  std::vector<Eigen::Vector2f> corners =
      getCornersFor(lastLabelPosition, lastHalfSize);
  std::vector<float> cornerAnchorDistances;
  for (auto corner : corners)
    cornerAnchorDistances.push_back((corner - anchor).squaredNorm());

  int maxIndex = std::distance(cornerAnchorDistances.begin(),
                               std::max_element(cornerAnchorDistances.begin(),
                                                cornerAnchorDistances.end()));

  // Just add a line shadow for two edges of the label.
  // The other two, which are connected to the farthest corner
  // would produce an area which is included in the first one.
  // The following table illustrates which lines must be drawn,
  // between which corners (given by index)
  //
  // maxIndex  line 1  line 2
  // 0         1 to 2  2 to 3
  // 1         2 to 3  3 to 0
  // 2         0 to 1  3 to 0
  // 3         0 to 1  1 to 2
  if (maxIndex == 2 || maxIndex == 3)
    addLineShadow(anchor, corners[0], corners[1]);

  if (maxIndex == 0 || maxIndex == 3)
    addLineShadow(anchor, corners[1], corners[2]);

  if (maxIndex == 0 || maxIndex == 1)
    addLineShadow(anchor, corners[2], corners[3]);

  if (maxIndex == 1 || maxIndex == 2)
    addLineShadow(anchor, corners[3], corners[0]);
}

std::vector<Eigen::Vector2f>
ConstraintUpdater::getCornersFor(Eigen::Vector2i position,
                                 Eigen::Vector2f halfSize)
{
  std::vector<Eigen::Vector2f> corners = {
    Eigen::Vector2f(position.x() + halfSize.x(), position.y() + halfSize.y()),
    Eigen::Vector2f(position.x() - halfSize.x(), position.y() + halfSize.y()),
    Eigen::Vector2f(position.x() - halfSize.x(), position.y() - halfSize.y()),
    Eigen::Vector2f(position.x() + halfSize.x(), position.y() - halfSize.y()),
  };

  return corners;
}

void ConstraintUpdater::save(std::string filename)
{
  anchorConstraintDrawer->saveBufferTo(filename);
}
