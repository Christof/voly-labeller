#ifndef SRC_FORCES_ANCHOR_DEPTH_FORCE_H_

#define SRC_FORCES_ANCHOR_DEPTH_FORCE_H_

#include "./force.h"

namespace Forces
{
/**
 * \brief
 *
 *
 */
class AnchorDepthForce : public Force
{
 public:
  AnchorDepthForce();

  Eigen::Vector3f calculate(LabelState &label, std::vector<LabelState> &labels,
                            const LabellerFrameData &frameData);
};
}  // namespace Forces

#endif  // SRC_FORCES_ANCHOR_DEPTH_FORCE_H_
