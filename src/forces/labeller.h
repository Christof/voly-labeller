#ifndef SRC_FORCES_LABELLER_H_

#define SRC_FORCES_LABELLER_H_

#include <Eigen/Core>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "./label_state.h"
#include "./force.h"
#include "../labelling/labeller_frame_data.h"
#include "../labelling/labels.h"

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
 * It stores all forces and runs the simulation when Labeller::update is called.
 */
class Labeller
{
 public:
  explicit Labeller(std::shared_ptr<Labels> labels);
  ~Labeller();

  void resize(int width, int height);

  void updateLabel(int id, Eigen::Vector3f anchorPosition);

  std::map<int, Eigen::Vector3f> update(const LabellerFrameData &frameData,
      std::map<int, Eigen::Vector3f> placementPositions);

  std::vector<LabelState> getLabels();

  std::vector<std::unique_ptr<Force>> forces;

  bool updatePositions = true;

 private:
  std::shared_ptr<Labels> labels;
  std::vector<LabelState> labelStates;
  std::function<void()> unsubscribeLabelChanges;

  template <class T> void addForce(T *force);
  void enforceAnchorDepthForLabel(LabelState &label,
                                  const Eigen::Matrix4f &viewMatrix);
  void setLabel(Labels::Action action, const Label &label);

  Eigen::Vector2f size;

  float epsilon = 0.05f;
};
}  // namespace Forces

#endif  // SRC_FORCES_LABELLER_H_
