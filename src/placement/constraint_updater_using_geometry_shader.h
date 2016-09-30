#ifndef SRC_PLACEMENT_CONSTRAINT_UPDATER_USING_GEOMETRY_SHADER_H_

#define SRC_PLACEMENT_CONSTRAINT_UPDATER_USING_GEOMETRY_SHADER_H_

#include <Eigen/Core>
#include <vector>
#include "./constraint_updater_base.h"

/**
 * \brief
 *
 *
 */
class ConstraintUpdaterUsingGeometryShader : public ConstraintUpdaterBase
{
 public:
  ConstraintUpdaterUsingGeometryShader();
  virtual ~ConstraintUpdaterUsingGeometryShader();

  void drawConstraintRegionFor(Eigen::Vector2i anchorPosition,
                               Eigen::Vector2i labelSize,
                               Eigen::Vector2i lastAnchorPosition,
                               Eigen::Vector2i lastLabelPosition,
                               Eigen::Vector2i lastLabelSize);
  void drawRegionsForAnchors(std::vector<Eigen::Vector2i> anchorPositions,
                             Eigen::Vector2i labelSize);

  void clear();
  void setIsConnectorShadowEnabled(bool enabled);

 private:
  /* data */
};

#endif  // SRC_PLACEMENT_CONSTRAINT_UPDATER_USING_GEOMETRY_SHADER_H_
