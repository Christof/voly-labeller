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

/**
 * \brief Contains classes for the force simulation part of label placement
 *
 */
namespace Forces
{

/**
 * \brief Facade for force simulation, which provides methods to add
 * labels
 *
 * It stores all forces and runs the simulation when Labeller::Update is called.
 */
class Labeller
{
 public:
  Labeller();

  void addLabel(int id, std::string text, Eigen::Vector3f anchorPosition,
                Eigen::Vector2f size);
  void updateLabel(int id, Eigen::Vector3f anchorPosition);

  std::map<int, Eigen::Vector3f> update(const LabellerFrameData &frameData);

  std::vector<LabelState> getLabels();

  std::vector<std::unique_ptr<Force>> forces;

  bool updatePositions = true;

 private:
  std::vector<LabelState> labels;

  template <class T> void addForce(T *force);
  void enforceAnchorDepthForLabel(LabelState &label,
                                  const Eigen::Matrix4f &viewMatrix);
};
}  // namespace Forces

#endif  // SRC_FORCES_LABELLER_H_
