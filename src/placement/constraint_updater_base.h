#ifndef SRC_PLACEMENT_CONSTRAINT_UPDATER_BASE_H_

#define SRC_PLACEMENT_CONSTRAINT_UPDATER_BASE_H_

#include <Eigen/Core>
#include <vector>

/**
 * \brief Interface for updating the constraint buffer by drawing occupied
 * regions for already placed labels
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
 * label. Following that, #finish be invoked.
 */
class ConstraintUpdaterBase
{
 public:
  virtual void drawConstraintRegionFor(Eigen::Vector2i anchorPosition,
                                       Eigen::Vector2i labelSize,
                                       Eigen::Vector2i lastAnchorPosition,
                                       Eigen::Vector2i lastLabelPosition,
                                       Eigen::Vector2i lastLabelSize) = 0;
  virtual void
  drawRegionsForAnchors(std::vector<Eigen::Vector2i> anchorPositions,
                        Eigen::Vector2i labelSize) = 0;

  virtual void clear() = 0;
  virtual void finish()
  {
  };
  virtual void setIsConnectorShadowEnabled(bool enabled) = 0;
};

#endif  // SRC_PLACEMENT_CONSTRAINT_UPDATER_BASE_H_
