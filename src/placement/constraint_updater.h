#ifndef SRC_PLACEMENT_CONSTRAINT_UPDATER_H_

#define SRC_PLACEMENT_CONSTRAINT_UPDATER_H_

#include <Eigen/Core>
#include <polyclipping/clipper.hpp>
#include <vector>
#include <utility>
#include <memory>
#include "../graphics/drawer.h"

/**
 * \brief Updates the constraint buffer by drawing occupied regions for already
 * placed labels
 *
 * Each placed label makes two regions unusable for future labels:
 * - A dilated shadow region which is created by using a virtual light position
 *   at the anchor point of the next placed label and using already placed
 *   label boxes as shadow casters.
 * - A dilated shadow region which is also created by using a virtual light
 *   position at the anchor point and using the line between an already placed
 *   label's anchor and the label box center.
 * The dilation size is determined by the new label size and a border.
 *
 *
 * For each newly placed label the already bound ConstraintBufferObject must
 * be cleared by calling #clear.
 * Afterwards #drawConstraintRegionFor must be called for each already placed
 * label.
 */
class ConstraintUpdater
{
 public:
  ConstraintUpdater(std::shared_ptr<Graphics::Drawer> drawer, int width,
                    int height);

  void drawConstraintRegionFor(Eigen::Vector2i anchorPosition,
                               Eigen::Vector2i labelSize,
                               Eigen::Vector2i lastAnchorPosition,
                               Eigen::Vector2i lastLabelPosition,
                               Eigen::Vector2i lastLabelSize);

  void clear();
  void setIsConnectorShadowEnabled(bool enabled);

 private:
  std::shared_ptr<Graphics::Drawer> drawer;
  int width;
  int height;
  bool isConnectorShadowEnabled = true;
  float labelShadowColor;
  float connectorShadowColor;

  std::vector<float> positions;

  void drawConnectorShadowRegion(Eigen::Vector2i anchorPosition,
                                 Eigen::Vector2i lastAnchorPosition,
                                 Eigen::Vector2i lastLabelPosition,
                                 const ClipperLib::Path &newLabel);
  void drawLabelShadowRegion(Eigen::Vector2i anchorPosition,
                             Eigen::Vector2i lastLabelPosition,
                             Eigen::Vector2i lastLabelSize,
                             const ClipperLib::Path &newLabel);
  void drawPolygon(ClipperLib::Path polygon);
};

#endif  // SRC_PLACEMENT_CONSTRAINT_UPDATER_H_
