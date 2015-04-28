#include "./anchor_depth_force.h"
#include "./label_state.h"
#include <Eigen/LU>

namespace Forces
{
AnchorDepthForce::AnchorDepthForce() : Force(0.01f)
{
}

Eigen::Vector4f toVector4f(Eigen::Vector3f vector)
{
  return Eigen::Vector4f(vector.x(), vector.y(), vector.z(), 1);
}

Eigen::Vector3f AnchorDepthForce::calculate(LabelState &label,
                                            std::vector<LabelState> &labels,
                                            const LabellerFrameData &frameData)
{
  auto anchorProjected =
      frameData.viewProjection * toVector4f(label.anchorPosition);
  auto labelProjected =
      frameData.viewProjection * toVector4f(label.labelPosition);

  Eigen::Vector4f zDiffProjected(0, 0, anchorProjected.z() - labelProjected.z(),
                                 1);

  Eigen::Vector4f zDiff = frameData.viewProjection.inverse() * zDiffProjected;

  return Eigen::Vector3f(zDiff.x(), zDiff.y(), zDiff.z()) / zDiff.w() * weight;
}
}  // namespace Forces
