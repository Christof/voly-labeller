#ifndef SRC_FORCES_LABELLER_H_

#define SRC_FORCES_LABELLER_H_

#include <Eigen/Core>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "./label_state.h"
#include "./force.h"
#include "./labeller_frame_data.h"

namespace Forces
{

/**
 * \brief
 *
 *
 */
class Labeller
{
 public:
  Labeller();

  void addLabel(int id, std::string text, Eigen::Vector3f anchorPosition,
                Eigen::Vector2f size);

  std::map<int, Eigen::Vector3f> update(const LabellerFrameData &frameData);

  std::vector<LabelState> getLabels();

 private:
  std::vector<std::unique_ptr<Force>> forces;
  std::vector<LabelState> labels;

  template <class T> void addForce(T *force);
  void enforceAnchorDepthForLabel(LabelState &label,
                                  const Eigen::Matrix4f &viewMatrix);
};
}  // namespace Forces

#endif  // SRC_FORCES_LABELLER_H_
