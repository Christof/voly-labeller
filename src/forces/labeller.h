#ifndef SRC_FORCES_LABELLER_H_

#define SRC_FORCES_LABELLER_H_

#include <Eigen/Core>
#include <string>
#include <vector>
#include <map>
#include <deque>
#include <memory>
#include <functional>
#include "./label_state.h"
#include "./force.h"
#include "../labelling/labeller_frame_data.h"
#include "../labelling/labels.h"
#include "../labelling/label_positions.h"

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

  LabelPositions update(const LabellerFrameData &frameData,
                        const LabelPositions &placementPositions);

  void setPositions(const LabellerFrameData &frameData,
      LabelPositions placementPositions);

  std::vector<LabelState> getLabels();

  std::vector<std::unique_ptr<Force>> forces;

  bool updatePositions = true;

  float overallForceFactor = 3.0f;

 private:
  std::shared_ptr<Labels> labels;
  std::vector<LabelState> labelStates;
  std::function<void()> unsubscribeLabelChanges;

  std::map<int, std::deque<Eigen::Vector2f>> oldPositions;

  template <class T> void addForce(T *force, bool enabled = true);
  void setLabel(Labels::Action action, const Label &label);

  Eigen::Vector2f size;

  float epsilon = 1e-7f;
};
}  // namespace Forces

#endif  // SRC_FORCES_LABELLER_H_
